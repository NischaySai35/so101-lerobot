"""Inverse kinematics for the SO-101 with the jaws pointing straight down.

Wraps placo directly rather than lerobot.model.kinematics.RobotKinematics so we can (a) fix wrist_roll,
(b) ask only for the approach axis to point down (5 DOF can't take a full 6-D pose), and (c) iterate to
convergence and report the residual.

Joint angles in and out are LeRobot calibrated degrees (use_degrees=True). LeRobot's 0 deg is the middle of
the range swept during calibration, while so101_new_calib.urdf's zero is the homing pose (raw 2047), so each
joint is shifted by (range_mid - 2047) * 360/4095 before going into the URDF (urdf_offsets_deg). Without this
the board-corner fit was 15 mm RMS; with it 3.9 mm (2026-10-17).
"""
import json

import numpy as np
import placo

from chess_player import config as C

DOWN = np.array([0.0, 0.0, -1.0])
TOOL_AXIS = np.array([0.0, 0.0, 1.0])  # gripper_frame_link z points out of the jaws (checked with FK at zero)


CAL_PATH = C.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{C.FOLLOWER_ID}.json"


def urdf_offsets_deg():
    """URDF angle = LeRobot degrees + offset, per arm joint (0 for wrist_roll, whose range is the full turn)."""
    if not CAL_PATH.exists():
        return {j: 0.0 for j in C.ARM_JOINTS}
    cal = json.loads(CAL_PATH.read_text())
    out = {j: ((cal[j]["range_min"] + cal[j]["range_max"]) / 2 - 2047) * 360 / 4095 for j in C.ARM_JOINTS}
    out["wrist_roll"] -= C.WRIST_ROLL_ZERO_SHIFT_DEG  # its zero moved when the follower was matched to the leader
    return out


def calibrated_limits_deg():
    """Joint ranges recorded by LeRobot calibration, in LeRobot degrees, shrunk by JOINT_MARGIN_DEG."""
    if not CAL_PATH.exists():
        return {}
    cal = json.loads(CAL_PATH.read_text())
    out = {}
    for j in C.ARM_JOINTS:
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]
        mid = (lo + hi) / 2
        out[j] = ((lo - mid) * 360 / 4095 + C.JOINT_MARGIN_DEG, (hi - mid) * 360 / 4095 - C.JOINT_MARGIN_DEG)
    return out


class IK:
    def __init__(self, wrist_roll_deg=C.WRIST_ROLL_DEG):
        self.robot = placo.RobotWrapper(str(C.URDF), placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.mask_dof("gripper")
        self.solver.mask_dof("wrist_roll")
        self.wrist_roll = wrist_roll_deg
        self.off = urdf_offsets_deg()
        # Intersect URDF limits with the real calibrated range so IK never asks for an unreachable angle.
        for j, (lo, hi) in calibrated_limits_deg().items():
            ulo, uhi = self.robot.get_joint_limits(j)
            self.robot.set_joint_limits(j, max(ulo, self._rad(j, lo)), min(uhi, self._rad(j, hi)))
        self.pos_task = self.solver.add_position_task(C.EE_FRAME, np.zeros(3))
        self.pos_task.configure("pos", "soft", 1.0)
        self.align_task = self.solver.add_axisalign_task(C.EE_FRAME, TOOL_AXIS, DOWN)
        self.align_task.configure("align", "soft", 0.3)
        self.solver.add_regularization_task(1e-4)

    def _rad(self, j, lerobot_deg):
        return np.deg2rad(lerobot_deg + self.off[j])

    def _deg(self, j):
        return float(np.rad2deg(self.robot.get_joint(j))) - self.off[j]

    def _set(self, q_deg):
        for j, v in zip(C.ARM_JOINTS, q_deg):
            self.robot.set_joint(j, self._rad(j, v))
        self.robot.set_joint("wrist_roll", self._rad("wrist_roll", self.wrist_roll))
        self.robot.update_kinematics()

    def fk(self, q_deg):
        """Pose (4x4, meters) of the jaw tip for arm joint angles [pan, lift, elbow, flex, roll] in LeRobot degrees."""
        for j, v in zip(C.ARM_JOINTS, q_deg):
            self.robot.set_joint(j, self._rad(j, v))
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(C.EE_FRAME)

    def _solve_once(self, target_m, seed_deg, align_w, iters):
        self._set(seed_deg)
        self.pos_task.target_world = np.asarray(target_m, float)
        self.align_task.configure("align", "soft", align_w)
        for _ in range(iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
            if np.linalg.norm(self.robot.get_T_world_frame(C.EE_FRAME)[:3, 3] - target_m) < 2e-4:
                break
        T = self.robot.get_T_world_frame(C.EE_FRAME)
        q = [self._deg(j) for j in C.ARM_JOINTS]
        q[4] = self.wrist_roll
        err = float(np.linalg.norm(T[:3, 3] - target_m) * 1000)
        tilt = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ DOWN, -1, 1))))
        return np.array(q), err, tilt

    def solve(self, target_m, seed_deg, max_tilt_deg=90.0, iters=80):
        """Joint angles (deg, 5 values) putting the jaw tip at target_m (robot frame, meters), pointing down.

        The wrist can't point straight down everywhere (wrist_flex saturates when the tip is high or far), so
        the down-alignment is relaxed step by step until position is met. Returns (q_deg, pos_err_mm,
        tilt_deg) of the most vertical solution within IK_MAX_ERR_MM, or the best-position one if none is.
        Seed with the previous solution to avoid elbow flips.
        """
        target_m = np.asarray(target_m, float)
        best = None
        for w in (0.3, 0.03, 3e-3, 3e-4, 0.0):
            q, err, tilt = self._solve_once(target_m, seed_deg, w, iters)
            if err <= C.IK_MAX_ERR_MM and tilt <= max_tilt_deg:
                return q, err, tilt
            if best is None or err < best[1]:
                best = (q, err, tilt)
        return best


class Unreachable(RuntimeError):
    pass


def solve_or_raise(ik, target_m, seed_deg, max_tilt_deg=90.0, what=""):
    q, err, tilt = ik.solve(target_m, seed_deg, max_tilt_deg)
    if err > C.IK_MAX_ERR_MM or tilt > max_tilt_deg:
        raise Unreachable(
            f"{what}: IK error {err:.1f} mm, tilt {tilt:.0f} deg at {np.round(np.asarray(target_m) * 1000)} mm"
        )
    return q
