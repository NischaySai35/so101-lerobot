"""Joint-space control of the follower: smooth interpolated moves streamed at CONTROL_HZ.

Talks to the Feetech bus directly (like go_home.py) instead of SOFollower.connect(), because connect() toggles
torque without first syncing Goal_Position to the present position, which can make the arm jump. Units are
the same as SOFollower.send_action with use_degrees=True: arm joints in calibrated degrees, gripper 0..100
(0 closed, 100 open).

FakeArm has the same interface and just tracks the commanded state, for dry runs without hardware.
"""
import json
import threading
import time

import numpy as np


from chess_player import config as C

CAL_PATH = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"


def min_jerk(a):
    return a**3 * (10 - 15 * a + 6 * a * a)


def load_pose(path):
    """Pose file from read_positions.py --save -> {joint: value} in calibrated units (from raw counts, so it
    survives recalibration)."""
    raw = json.loads(path.read_text())["raw"]
    cal = json.loads(CAL_PATH.read_text())
    out = {}
    for j in C.ALL_JOINTS:
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]
        if j == "gripper":
            out[j] = (min(hi, max(lo, raw[j])) - lo) / (hi - lo) * 100
        else:
            out[j] = (raw[j] - (lo + hi) / 2) * 360 / 4095
    return out


def look_pose():
    return load_pose(C.LOOK_POSE if C.LOOK_POSE.exists() else C.HOME_POSE)


class ArmStopped(RuntimeError):
    """The emergency stop was pressed: the arm froze where it was."""


class _Base:
    """Shared trajectory logic. Subclasses implement _read() and _write(goal_dict).

    Emergency stop: set `stop_event` from any thread; every motion loop checks it each control tick (~33 ms),
    freezes the arm where it is and raises ArmStopped."""

    log = print
    torque_on = True

    @property
    def stop_event(self):
        if "_stop_event" not in self.__dict__:
            self._stop_event = threading.Event()
        return self._stop_event

    def _check_stop(self):
        if self.stop_event.is_set():
            self.freeze()
            raise ArmStopped("emergency stop")

    def freeze(self):
        """Hold exactly where the arm is right now."""

    def joints(self) -> dict:
        return self._read()

    def move_to(self, target: dict, duration=None):
        """Min-jerk interpolation in joint space. `target` may omit joints (they stay put)."""
        start = self._read()
        goal = {**start, **target}
        span = max(abs(goal[j] - start[j]) for j in C.ARM_JOINTS)
        dur = max(duration or C.MIN_MOVE_S, span / C.MAX_JOINT_SPEED)
        n = max(1, int(dur * C.CONTROL_HZ))
        t0 = time.perf_counter()
        for k in range(1, n + 1):
            self._check_stop()
            a = min_jerk(k / n)
            self._write({j: start[j] + (goal[j] - start[j]) * a for j in C.ALL_JOINTS})
            self._sleep_until(t0 + k / C.CONTROL_HZ)
        return goal

    def move_path(self, waypoints, duration, check=None):
        """Follow a dense list of joint dicts (e.g. IK of a straight Cartesian line) over `duration` seconds,
        with min-jerk timing along the list. `check()` is called every few ticks; if it returns a reason, the arm
        stops where it is and that reason is returned (else None)."""
        n = max(1, int(duration * C.CONTROL_HZ))
        m = len(waypoints) - 1
        t0 = time.perf_counter()
        for k in range(1, n + 1):
            self._check_stop()
            if check is not None and k % 4 == 0:
                why = check()
                if why:
                    return why
            s = min_jerk(k / n) * m
            i = min(int(s), m - 1)
            f = s - i
            a, b = waypoints[i], waypoints[i + 1]
            self._write({j: a[j] + (b[j] - a[j]) * f for j in C.ALL_JOINTS})
            self._sleep_until(t0 + k / C.CONTROL_HZ)

    def gripper(self, percent, duration=C.GRIP_S):
        self.move_to({"gripper": float(percent)}, duration)

    def settle(self, iters=C.SAG_ITERS):
        """Hold still and correct for sag: under load the servos stop short of the goal (gravity, weak P gain),
        so measure the error and add it to the command. Returns the remaining worst arm-joint error (deg)."""
        return 0.0

    def dwell(self, s=C.DWELL_S):
        t_end = time.perf_counter() + s
        while time.perf_counter() < t_end:
            self._check_stop()
            time.sleep(min(0.03, max(0.0, t_end - time.perf_counter())))

    @staticmethod
    def _sleep_until(t):
        d = t - time.perf_counter()
        if d > 0:
            time.sleep(d)


class Arm(_Base):
    def __init__(self, port=C.FOLLOWER_PORT, use_sag_model=C.USE_SAG_MODEL):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        cal = {k: MotorCalibration(**v) for k, v in json.loads(CAL_PATH.read_text()).items()}
        motors = {
            j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
            for i, j in enumerate(C.ALL_JOINTS)
        }
        self.bus = FeetechMotorsBus(port, motors, calibration=cal)
        self.lock = threading.RLock()  # the UI thread may freeze / release the arm while the worker moves it
        self.bus.connect(handshake=True)
        # Hold the present position before enabling torque so nothing jumps.
        present = self.bus.sync_read("Present_Position", normalize=False)
        self.bus.sync_write("Goal_Position", present, normalize=False)
        for j in C.ALL_JOINTS:
            self.bus.write("Torque_Enable", j, 1)
        self._last = self._read_present()
        self.comp = {}  # residual sag correction (deg) on top of the gravity model, updated by settle()
        from chess_player.motion.sag import SagModel

        self.sag = SagModel.load() if use_sag_model else None

    def _read_present(self):
        with self.lock:
            return {j: float(v) for j, v in self.bus.sync_read("Present_Position", num_retry=2).items()}

    def _read(self):
        # The last *intended* goal is the trajectory start: smoother than the (sagging) present position.
        return dict(self._last)

    def present(self):
        return self._read_present()

    def correction(self, goal):
        """Total correction for `goal`: gravity-model prediction plus what settle() learned on top."""
        pred = self.sag.predict(goal) if self.sag else {}
        return {j: pred.get(j, 0.0) + self.comp.get(j, 0.0) for j in goal}

    def _write(self, goal):
        if not self.torque_on:  # torque was switched off: re-engage where the arm is, then continue
            self.hold()
            return
        corr = self.correction(goal)
        cmd = {j: goal[j] + corr[j] for j in goal}
        with self.lock:
            self.bus.sync_write("Goal_Position", cmd)
        self._last = dict(goal)

    def freeze(self):
        """Hold exactly where the arm physically is right now (goal = present position)."""
        with self.lock:
            raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            self.bus.sync_write("Goal_Position", raw, normalize=False)
        self._last = self._read_present()
        self.comp = {}

    def hold(self):
        """Torque back on at the present position (nothing jumps)."""
        with self.lock:
            raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            self.bus.sync_write("Goal_Position", raw, normalize=False)
            for j in C.ALL_JOINTS:
                self.bus.write("Torque_Enable", j, 1)
        self.torque_on = True
        self._last = self._read_present()
        self.comp = {}

    def gripper_load(self):
        """Gripper motor load, % of max torque (pressing on something = high)."""
        with self.lock:
            return abs(self.bus.sync_read("Present_Load", ["gripper"], normalize=False, num_retry=2)["gripper"]) / 10

    def load_status(self):
        """({joint: load % of max torque}, {joint: status byte, nonzero = error}) for the arm joints."""
        with self.lock:
            load = self.bus.sync_read("Present_Load", list(C.ARM_JOINTS), normalize=False, num_retry=2)
            status = self.bus.sync_read("Status", list(C.ARM_JOINTS), normalize=False, num_retry=2)
        return ({j: abs(v) / 10 for j, v in load.items()}, {j: v for j, v in status.items() if v})

    def settle(self, iters=C.SAG_ITERS):
        worst = 0.0
        for _ in range(iters):
            self.dwell(C.SAG_WAIT_S)
            now = self._read_present()
            worst = 0.0
            for j in C.ARM_JOINTS:
                err = self._last[j] - now[j]
                worst = max(worst, abs(err))
                if abs(err) > C.SAG_DEADBAND_DEG:
                    self.comp[j] = float(np.clip(self.comp.get(j, 0.0) + err, -C.SAG_MAX_DEG, C.SAG_MAX_DEG))
            if worst <= C.SAG_DEADBAND_DEG:
                break
            self._write(self._last)
        return worst

    def reset_residual(self):
        self.comp = {}

    def release(self):
        """Torque off. The arm drops: support it."""
        with self.lock:
            for j in C.ALL_JOINTS:
                self.bus.write("Torque_Enable", j, 0)
        self.torque_on = False

    def close(self, release=False):
        if release:
            self.release()
        self.bus.port_handler.closePort()


class FakeArm(_Base):
    def __init__(self, start=None, realtime=False):
        self._q = dict(start or look_pose())
        self.realtime = realtime
        self.history = []

    def _read(self):
        return dict(self._q)

    present = _read

    def _write(self, goal):
        self._q = dict(goal)
        self.history.append(dict(goal))

    def _sleep_until(self, t):
        if self.realtime:
            super()._sleep_until(t)

    def dwell(self, s=C.DWELL_S):
        self._check_stop()
        if self.realtime:
            time.sleep(s)

    def release(self):
        self.torque_on = False

    def hold(self):
        self.torque_on = True

    def gripper_load(self):
        return 0.0  # simulated: always free

    def close(self, release=False):
        pass
