"""Pixel <-> board-mm mapping from the four ArUco markers (or from manually clicked board corners).

The homography is recomputed on every capture so small look-pose differences don't matter. With camera
intrinsics (calib/camera_intrinsics.py) points are undistorted first and a full camera pose (solvePnP) is
available, which the auto-labeler uses to project 3-D piece boxes. Without intrinsics a guessed pinhole
model is used for that.
"""
import json

import cv2
import numpy as np

from chess_player import config as C


def intrinsics(size=None, calibrated_file=C.INTRINSICS):
    """(K, dist, calibrated). `size` = (w, h) of the frame; the intrinsics file only applies to the wrist cam."""
    if calibrated_file is not None and calibrated_file.exists():
        d = np.load(calibrated_file)
        return d["K"], d["dist"], True
    w, h = size or (C.CAM_W, C.CAM_H)
    f = 0.9 * w  # rough guess for a ~60 deg webcam
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), np.zeros(5), False


def focal_from_homography(H, cx, cy):
    """Focal length (px) from a plane homography, assuming square pixels and the principal point at (cx, cy).

    The board's x and y axes are perpendicular and equally scaled in 3-D, which gives two equations in 1/f^2
    (Zhang's constraints with a one-parameter K). Returns None when the view is too frontal to tell.
    """
    Hc = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H
    (a1, b1, c1), (a2, b2, c2) = Hc[:, 0], Hc[:, 1]
    est = []
    if abs(c1 * c2) > 1e-12:
        est.append(-(a1 * a2 + b1 * b2) / (c1 * c2))
    if abs(c1 * c1 - c2 * c2) > 1e-12:
        est.append(-(a1 * a1 + b1 * b1 - a2 * a2 - b2 * b2) / (c1 * c1 - c2 * c2))
    est = [e for e in est if e > 0]
    return float(np.sqrt(np.mean(est))) if est else None


class BoardView:
    """Result of locating the board in one frame."""

    def __init__(self, img_pts, board_pts_mm, source, size=None, intrinsics_file=C.INTRINSICS):
        self.size = size or (C.CAM_W, C.CAM_H)
        self.K, self.dist, self.calibrated = intrinsics(self.size, intrinsics_file)
        self.source = source  # "aruco" or "manual"
        und = self._undistort(img_pts)
        self.H, _ = cv2.findHomography(np.asarray(board_pts_mm, np.float32), und, 0)  # board mm -> undistorted px
        self.Hinv = np.linalg.inv(self.H)
        if not self.calibrated:
            f = focal_from_homography(self.H, self.K[0, 2], self.K[1, 2])
            if f is not None:
                self.K[0, 0] = self.K[1, 1] = f
        obj = np.c_[np.asarray(board_pts_mm, np.float32), np.zeros(len(board_pts_mm), np.float32)]
        ok, self.rvec, self.tvec = cv2.solvePnP(obj, np.asarray(img_pts, np.float32), self.K, self.dist,
                                                 flags=cv2.SOLVEPNP_IPPE if len(obj) >= 4 else cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.rvec = self.tvec = None

    def _undistort(self, px):
        px = np.asarray(px, np.float32).reshape(-1, 1, 2)
        if not self.calibrated:
            return px.reshape(-1, 2)
        return cv2.undistortPoints(px, self.K, self.dist, P=self.K).reshape(-1, 2)

    def px_to_mm(self, px):
        und = self._undistort(px).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(und, self.Hinv).reshape(-1, 2)

    def mm_to_px(self, mm):
        """Board-plane points (z=0) to raw image pixels, through the homography (exact for the plane)."""
        mm = np.asarray(mm, np.float64).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(mm, self.H).reshape(-1, 2)
        if not self.calibrated:
            return und
        # re-apply lens distortion: undistorted pixel -> normalized ray -> distorted pixel
        rays = cv2.convertPointsToHomogeneous((und - self.K[:2, 2]) / np.diag(self.K)[:2]).reshape(-1, 3)
        px, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.dist)
        return px.reshape(-1, 2)

    def project(self, pts_mm_3d):
        """3-D board-frame points (mm, z up) to raw image pixels, via the PnP pose (approximate without
        intrinsics: callers should anchor to mm_to_px of a known plane point)."""
        pts = np.asarray(pts_mm_3d, np.float64).reshape(-1, 3)
        if self.rvec is None:
            return cv2.perspectiveTransform(pts[:, None, :2], self.H).reshape(-1, 2)
        px, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    def warp(self, frame, px_per_mm=2.5, margin_mm=0.0):
        """Top-down view of the board plane (rank 8 at the top). Only for display/change detection."""
        B = 8 * C.SQUARE_MM
        s = px_per_mm
        # board mm -> output px: x right, y up flipped so rank 8 is at the top
        A = np.array([[s, 0, margin_mm * s], [0, -s, (B + margin_mm) * s], [0, 0, 1]])
        size = int((B + 2 * margin_mm) * s)
        und = frame if not self.calibrated else cv2.undistort(frame, self.K, self.dist)
        return cv2.warpPerspective(und, A @ self.Hinv, (size, size))


_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
_DETECTOR = cv2.aruco.ArucoDetector(_DICT, cv2.aruco.DetectorParameters())


def detect_markers(frame):
    """{marker id: center px} for the configured marker ids."""
    corners, ids, _ = _DETECTOR.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if int(i) in C.MARKER_CENTERS:
                out[int(i)] = c.reshape(4, 2).mean(0)
    return out


def corners_view(frame, path) -> BoardView:
    """BoardView from clicked playing-area corners a1, h1, h8, a8 (outer corners of those squares)."""
    d = json.loads(path.read_text())
    B = C.PLAY_MM
    h, w = frame.shape[:2]
    return BoardView(d["px"], [(0, 0), (B, 0), (B, B), (0, B)], "corners", (w, h), intrinsics_file=None)


def locate_board(frame, corners_file=None) -> BoardView:
    """Fixed camera: pass its corners file. Wrist camera: ArUco markers (all 4), else data/manual_corners.json."""
    if corners_file is not None:
        return corners_view(frame, corners_file)
    found = detect_markers(frame)
    if len(found) == len(C.MARKER_CENTERS):
        ids = sorted(found)
        return BoardView([found[i] for i in ids], [C.MARKER_CENTERS[i] for i in ids], "aruco")
    if C.MANUAL_CORNERS.exists():
        d = json.loads(C.MANUAL_CORNERS.read_text())
        B = C.PLAY_MM
        return BoardView(d["px"], [(0, 0), (B, 0), (B, B), (0, B)], "manual")
    raise RuntimeError(f"saw ArUco markers {sorted(found)} of {sorted(C.MARKER_CENTERS)} and no manual corners file")


def draw_overlay(frame, view: BoardView):
    """Grid + square names drawn on a copy of the frame."""
    import chess

    out = frame.copy()
    s = C.SQUARE_MM
    for i in range(9):
        for a, b in (((i * s, 0), (i * s, 8 * s)), ((0, i * s), (8 * s, i * s))):
            p = view.mm_to_px([a, b]).astype(int)
            cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 255, 0), 1)
    for sq in chess.SQUARES:
        c = view.mm_to_px([((chess.square_file(sq) + 0.5) * s, (chess.square_rank(sq) + 0.5) * s)])[0].astype(int)
        cv2.putText(out, chess.square_name(sq), tuple(c - [8, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return out
