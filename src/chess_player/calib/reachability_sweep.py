"""IK on all 64 squares (grasp height of every piece type + travel height), plus graveyard and spare slots.

  python -m chess_player.calib.reachability_sweep                 # uses data/board_to_robot.json
  python -m chess_player.calib.reachability_sweep --assume 110    # before calibration: near edge 110 mm out
  python -m chess_player.calib.reachability_sweep --search        # find the best near-edge distance / square size

A square is OK when every grasp height solves within IK_MAX_ERR_MM with tilt <= --grasp-tilt and the travel
height solves at all. Prints an 8x8 map (rank 8 at the top).
"""
import argparse

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.ik import IK

SEED = np.array([0, -30, 40, 80, C.WRIST_ROLL_DEG])


def grasp_z_mm(piece_letter):
    return C.GRASP_HEIGHT_MM  # pieces are grabbed at their bottom


def point_status(ik, T, x, y, grasp_tilt):
    """0 = fine, 1 = grasp tilt too large, 2 = travel or grasp unreachable."""
    worst = 0
    q, err, _ = ik.solve(T(x, y, C.Z_SAFE), SEED)
    if err > C.IK_MAX_ERR_MM:
        return 2, None
    for p in C.PIECES:
        _, err, tilt = ik.solve(T(x, y, grasp_z_mm(p)), q)
        if err > C.IK_MAX_ERR_MM:
            return 2, None
        if tilt > grasp_tilt:
            worst = 1
    return worst, q


def sweep(ik, T, grasp_tilt, show=True):
    grid = np.zeros((8, 8), int)
    for sq in chess.SQUARES:
        x, y = square_center_mm(sq)
        grid[7 - chess.square_rank(sq), chess.square_file(sq)], _ = point_status(ik, T, x, y, grasp_tilt)
    extra = {f"graveyard{i}": xy for i, xy in enumerate(C.GRAVEYARD)}
    extra.update({f"spare_{k}{i}": xy for k, v in C.SPARES.items() for i, xy in enumerate(v)})
    extra_status = {k: point_status(ik, T, *xy, grasp_tilt)[0] for k, xy in extra.items()}
    if show:
        sym = {0: " .", 1: " t", 2: " X"}
        print("   " + "".join(f" {f}" for f in "abcdefgh"))
        for r in range(8):
            print(f" {8 - r} " + "".join(sym[v] for v in grid[r]))
        print(f"  . ok   t = grasp tilt > {grasp_tilt:.0f} deg   X = unreachable")
        bad = {k: v for k, v in extra_status.items() if v}
        print(f"  off-board slots: {len(extra) - len(bad)}/{len(extra)} ok" + (f", problems: {bad}" if bad else ""))
    return grid, extra_status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assume", type=float, metavar="NEAR_MM", help="use an assumed placement instead of calibration")
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--table-z", type=float, default=0.0, help="board surface height in the URDF base frame (mm)")
    ap.add_argument("--grasp-tilt", type=float, default=15.0)
    ap.add_argument("--search", action="store_true")
    a = ap.parse_args()
    ik = IK()

    if a.search:
        print(f"square_mm near_mm  ok  tilted  unreachable   (grasp tilt <= {a.grasp_tilt:.0f} deg)")
        for s in (25.0, 28.0, 30.0, 32.0, 35.0):
            C.SQUARE_MM = s
            for near in range(60, 181, 20):
                g, _ = sweep(ik, BoardToRobot.assumed(near, a.lateral, a.table_z, C.ROBOT_COLOR), a.grasp_tilt, False)
                print(f"{s:8.0f} {near:7d} {np.sum(g == 0):4d} {np.sum(g == 1):6d} {np.sum(g == 2):8d}")
        return

    T = BoardToRobot.assumed(a.assume, a.lateral, a.table_z, C.ROBOT_COLOR) if a.assume is not None else BoardToRobot.load()
    print(f"square {C.SQUARE_MM} mm, z_safe {C.Z_SAFE:.0f} mm")
    sweep(ik, T, a.grasp_tilt)


if __name__ == "__main__":
    main()
