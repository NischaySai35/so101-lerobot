"""Predict joint sag anywhere from gravity, so commands are corrected before the arm gets there.

Each position-controlled servo stops short of its goal roughly in proportion to the load it holds. The URDF
gives the gravity torque on every joint for any pose (placo generalized_gravity), so per joint

    correction_deg = k * gravity_torque_Nm + b

with k, b fitted from a few measured poses (calib/sag_calibration.py). The wrist camera and finger pads aren't
in the URDF; k absorbs that. Saved in data/sag_model.json.
"""
import json

import numpy as np

from chess_player import config as C
from chess_player.motion.ik import IK

SAG_MODEL = C.DATA / "sag_model.json"
FIT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


class GravityTorques:
    def __init__(self):
        self.ik = IK()  # own placo instance (never shares state with the motion IK)
        self.v = {j: self.ik.robot.get_joint_v_offset(j) for j in C.ARM_JOINTS}

    def __call__(self, q: dict) -> dict:
        self.ik._set([q[j] for j in C.ARM_JOINTS])
        self.ik.robot.set_joint("wrist_roll", self.ik._rad("wrist_roll", q["wrist_roll"]))
        self.ik.robot.update_kinematics()
        g = self.ik.robot.generalized_gravity()
        return {j: float(g[self.v[j]]) for j in C.ARM_JOINTS}


class SagModel:
    def __init__(self, k: dict, b: dict):
        self.k, self.b = k, b
        self.tau = GravityTorques()

    def predict(self, q: dict) -> dict:
        t = self.tau(q)
        return {j: self.k.get(j, 0.0) * t[j] + self.b.get(j, 0.0) for j in FIT_JOINTS}

    @classmethod
    def load(cls, path=SAG_MODEL):
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(d["k"], d["b"])

    @staticmethod
    def fit(samples):
        """samples: [(joint dict, total correction dict)]. Returns (k, b, per-joint RMS residual deg)."""
        tau = GravityTorques()
        T = [tau(q) for q, _ in samples]
        k, b, rms = {}, {}, {}
        for j in FIT_JOINTS:
            y = np.array([c.get(j, 0.0) for _, c in samples])
            x = np.array([t[j] for t in T])
            if np.ptp(x) < 0.02:  # no torque variation (pan): bias only
                k[j], b[j] = 0.0, float(y.mean())
            else:
                A = np.c_[x, np.ones_like(x)]
                (k[j], b[j]), *_ = np.linalg.lstsq(A, y, rcond=None)
                k[j], b[j] = float(k[j]), float(b[j])
            rms[j] = float(np.sqrt(np.mean((k[j] * x + b[j] - y) ** 2)))
        return k, b, rms

    @staticmethod
    def save(k, b, rms, samples, path=SAG_MODEL):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"k": k, "b": b, "rms_deg": rms,
                                    "samples": [{"q": q, "corr": c} for q, c in samples]}, indent=2))
