"""Turn recorded demonstrations into motion parameters for the hand-coded planner.

  python -m chess_player.teach.analyze_demo            # every episode folder in data/demos/
  python -m chess_player.teach.analyze_demo --last 1   # just the newest

Phases are found from the gripper reading (0 closed .. 100 open; empty-closed ~2-4 %, holding a piece ~10-20 %):
  open    the gripper rises well above empty-closed before the grasp
  grasp   it then drops and settles above empty-closed (something is in the jaws)
  release it rises again after holding
The source / target squares come from the task text ("rook e3 to e5"); offsets are measured from those square
centres, both in board axes and along the jaw axis (toward the moving finger, + = right of the fixed finger).
"""
import argparse
import csv
import re

import chess
import numpy as np

from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm

DEMOS = C.DATA / "demos"


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def phases(d):
    """Indices (open, grasp, release, close_after) or None.

    Holding a piece = the follower's jaws stay wider than the leader commands (the piece blocks them). The longest
    such stretch is the carry: its start is the grasp, its end the release. "Open" is where the jaws opened before it.
    """
    g, l = d["f_gripper"], d["l_gripper"]
    empty = np.nanpercentile(g, 5)
    holding = ((g - l) > 3) & (g > empty + 3)
    runs, start = [], None
    for i, v in enumerate(np.r_[holding, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    runs = [r for r in runs if r[1] - r[0] >= 15]
    if not runs:
        return None
    i_grasp, i_release = max(runs, key=lambda r: r[1] - r[0])
    before = np.where(g[:i_grasp] <= empty + 2)[0]
    i_open = int(before[-1]) + 1 if len(before) else 0
    after = np.where(g[i_release:] <= empty + 2)[0]
    i_close = i_release + int(after[0]) if len(after) else None
    return i_open, i_grasp, i_release, i_close


def jaw_axis(T, ik, d, i):
    """Unit board-xy vector toward the moving finger at frame i (from the recorded joints)."""
    from chess_player.motion.pick_place import PickPlace

    q = np.array([d[f"f_{j}"][i] for j in C.ARM_JOINTS])
    pp = PickPlace.__new__(PickPlace)
    pp.ik, pp.T = ik, T
    tip = ik.fk(q)[:3, 3]
    p_board = T.R.T @ (tip - T.t) * 1000
    _, u = pp._jaw_shifted(tuple(p_board[:2]), float(p_board[2]), q)
    return u


def summarise(path, T, ik):
    d = load(path)
    task = (path.parent / "info.json").read_text() if (path.parent / "info.json").exists() else ""
    sqs = re.findall(r"\b([a-h][1-8])\b", task.lower())
    t, g, h = d["t"], d["f_gripper"], d["height_above_board_mm"]
    bx, by = d["board_x_mm"], d["board_y_mm"]
    print(f"\n=== {path.parent.name}  ({t[-1]:.1f} s)")
    ph = phases(d)
    if ph is None:
        print("  no clear open -> grasp -> release in the gripper trace")
        return None
    i_open, i_grasp, i_rel, i_close = ph
    res = {}

    def offset(i, sq):
        c = square_center_mm(chess.parse_square(sq))
        off = np.array([bx[i] - c[0], by[i] - c[1]])
        u = jaw_axis(T, ik, d, i)
        w = np.array([-u[1], u[0]])
        return off, float(off @ u), float(off @ w)

    src = sqs[0] if sqs else d["square"][i_grasp]
    dst = sqs[1] if len(sqs) > 1 else (d["square"][i_rel] if i_rel is not None else None)
    print(f"  task squares: {src} -> {dst}")
    print(f"  open    at {t[i_open]:5.1f}s: gripper -> {np.nanmax(g[i_open:i_grasp + 1]):.1f}% at {h[i_open]:.0f} mm")
    off, along, across = offset(i_grasp, src)
    print(f"  grasp   at {t[i_grasp]:5.1f}s: tip {h[i_grasp]:.0f} mm above the board, tilt {d['tilt_deg'][i_grasp]:.0f} deg, "
          f"holding {np.nanmedian(g[i_grasp:i_grasp + 30]):.1f}% (leader asked {np.nanmedian(d['l_gripper'][i_grasp:i_grasp + 30]):.1f}%)")
    print(f"          tip offset from {src} centre: files {off[0]:+.0f} mm, ranks {off[1]:+.0f} mm  "
          f"= {along:+.0f} mm along the jaw axis ({'toward the fixed finger' if along < 0 else 'toward the moving finger'}), "
          f"{across:+.0f} mm across")
    res.update(open_pct=float(np.nanmax(g[i_open:i_grasp + 1])), grasp_h=float(h[i_grasp]), hold_pct=float(np.nanmedian(g[i_grasp:i_grasp + 30])),
               grasp_along=along, grasp_across=across)
    end = i_rel if i_rel is not None else len(t) - 1
    i_peak = i_grasp + int(np.nanargmax(h[i_grasp:end + 1]))
    res["lift_peak"] = float(h[i_peak] - h[i_grasp])
    print(f"  carry   peak {h[i_peak]:.0f} mm at {t[i_peak]:.1f}s (lift {res['lift_peak']:.0f} mm above the grasp height), "
          f"tilt {np.nanmax(d['tilt_deg'][i_grasp:end + 1]):.0f} deg max")
    if i_rel is not None and dst:
        off, along, across = offset(i_rel, dst)
        print(f"  release at {t[i_rel]:5.1f}s: tip {h[i_rel]:.0f} mm above the board, gripper -> "
              f"{np.nanmax(g[i_rel:i_rel + 45]):.1f}%; tip offset from {dst} centre: files {off[0]:+.0f}, ranks {off[1]:+.0f} mm "
              f"= {along:+.0f} along / {across:+.0f} across the jaw axis")
        res.update(release_h=float(h[i_rel]), release_along=along, release_across=across)
        after = slice(i_rel, i_close if i_close is not None else len(t))
        res["retreat_h"] = float(np.nanmax(h[after]) - h[i_rel])
        print(f"  retreat {res['retreat_h']:.0f} mm up before closing the gripper"
              + (f" (closed at {t[i_close]:.1f}s)" if i_close is not None else ""))
    print(f"  max load: shoulder {np.nanmax(d['lift_load_pct']):.0f}%, elbow {np.nanmax(d['elbow_load_pct']):.0f}%")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, help="only the newest N episodes")
    a = ap.parse_args()
    files = sorted(DEMOS.glob("*/poses.csv"))
    if a.last:
        files = files[-a.last:]
    if not files:
        raise SystemExit(f"no episodes in {DEMOS}")
    from chess_player.motion.ik import IK

    T, ik = BoardToRobot.load(), IK()
    results = [r for f in files if (r := summarise(f, T, ik))]
    if len(results) > 1:
        print("\n=== average over", len(results), "demos")
        for k in results[0]:
            v = [r[k] for r in results if k in r]
            print(f"  {k:15} {np.mean(v):7.1f}   (min {np.min(v):.1f}, max {np.max(v):.1f})")
    if results:
        m = {k: float(np.mean([r[k] for r in results if k in r])) for k in results[0]}
        print("\n  suggested planner settings (config.py):")
        print(f"    JAW_SHIFT_MM    = {-m['grasp_along']:.0f}   (now {C.JAW_SHIFT_MM:.0f})")
        print(f"    GRIPPER_OPEN    = {m['open_pct']:.0f}   (now {C.GRIPPER_OPEN:.0f})")
        print(f"    GRASP_HEIGHT_MM = {max(1.0, m['grasp_h']):.0f}   (now {C.GRASP_HEIGHT_MM:.0f})")
        print(f"    lift            ~ {m['lift_peak']:.0f} mm   (planner arc now {C.ARC_MIN_MM:.0f}-{C.ARC_MAX_MM:.0f} mm)")


if __name__ == "__main__":
    main()
