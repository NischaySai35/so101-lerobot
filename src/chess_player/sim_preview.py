"""Preview a robot move as a GIF before the real arm does it (nothing moves). Ported from chess-player's
sim/run_sim.py, but driven by this project's planner, IK and board calibration: the frames are exactly the joint
commands the real arm would get (FakeArm records them at CONTROL_HZ).

  python -m chess_player.sim_preview --move e2e4                 # from the start position
  python -m chess_player.sim_preview --fen "<FEN>" --move e4d5
  python -m chess_player.sim_preview --fen "<FEN>"               # Stockfish picks the move

Writes data/sim/<move>.gif plus start/carry/end PNGs, and prints the feasibility report.
"""
import argparse

import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from chess_player import config as C  # noqa: E402
from chess_player.geometry import BoardToRobot, square_center_mm  # noqa: E402
from chess_player.motion.arm import FakeArm, look_pose  # noqa: E402
from chess_player.motion.ik import IK  # noqa: E402
from chess_player.motion.pick_place import PickPlace, Slots  # noqa: E402

LINKS = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", C.EE_FRAME]
OUT = C.DATA / "sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default="start")
    ap.add_argument("--move", help="UCI; default: Stockfish's move")
    a = ap.parse_args()
    board = chess.Board() if a.fen == "start" else chess.Board(a.fen)
    if a.move:
        move = chess.Move.from_uci(a.move)
    else:
        from chess_player.game.engine import Engine

        e = Engine()
        move = e.best_move(board)
        e.close()
    T = BoardToRobot.load() if C.BOARD_TO_ROBOT.exists() else BoardToRobot.assumed(140)
    ik = IK()
    arm = FakeArm({**look_pose(), "gripper": C.GRIPPER_LOOK})
    pp = PickPlace(arm, ik, T, verify_grasp=False, log=print)
    slots = Slots(T)
    report = pp.check(board, move, slots)
    print(f"{report.get('san', move.uci())}: physical_ok={report.get('physical_ok')}")
    for s in report.get("steps", []):
        print("  ", s)
    if not report.get("physical_ok"):
        return
    arm.history.clear()
    transfers = pp.play(board, move, slots)
    hist = arm.history
    print(f"{len(hist)} joint commands = {len(hist) / C.CONTROL_HZ:.1f} s of motion")

    # robot-frame geometry for drawing
    def chain(q):
        ik.fk(np.array([q[j] for j in C.ARM_JOINTS]))
        return np.array([[0, 0, 0]] + [ik.robot.get_T_world_frame(l)[:3, 3] for l in LINKS])

    chains = [chain(q) for q in hist]
    tips = np.array([c[-1] for c in chains])
    grip = np.array([q["gripper"] for q in hist])
    holding = grip < C.GRIPPER_OPEN - 1.0  # jaws closed on a piece (between grasp and release)
    sq_xy = {sq: T(*square_center_mm(sq))[:2] for sq in chess.SQUARES}
    first = transfers[0]
    moving = board.piece_at(move.from_square) if not board.is_capture(move) else None

    fig = plt.figure(figsize=(11, 5))
    a3, a2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122)

    def draw(i):
        a3.cla(), a2.cla()
        c = chains[i]
        for sq in chess.SQUARES:
            x, y = sq_xy[sq]
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            a2.add_patch(plt.Rectangle((y - C.SQUARE_MM / 2000, x - C.SQUARE_MM / 2000), C.SQUARE_MM / 1000,
                                       C.SQUARE_MM / 1000, color="#7a4b1e" if dark else "#d9b27c", alpha=0.8))
        for sq, p in board.piece_map().items():
            x, y = sq_xy[sq]
            h = C.PIECES[p.symbol().lower()]["height"] / 1000
            col = "#eeeeee" if p.color else "#222222"
            a3.plot([y, y], [x, x], [T.t[2], T.t[2] + h], color=col, lw=4, solid_capstyle="round")
            a2.add_patch(plt.Circle((y, x), C.PIECES[p.symbol().lower()]["diameter"] / 2000, color=col, ec="gray"))
        if holding[i]:
            a3.plot([tips[i, 1]] * 2, [tips[i, 0]] * 2, [tips[i, 2] - 0.02, tips[i, 2]], color="red", lw=6)
            a2.add_patch(plt.Circle((tips[i, 1], tips[i, 0]), 0.01, color="red"))
        a3.plot(c[:, 1], c[:, 0], c[:, 2], "-o", color="#ff7f0e", lw=3, ms=3)
        a3.plot(tips[:, 1], tips[:, 0], tips[:, 2], color="g", lw=1, alpha=0.6)
        a2.plot(tips[:, 1], tips[:, 0], "g", lw=1)
        a2.plot(c[:, 1], c[:, 0], "-", color="#ff7f0e", lw=3)
        a2.plot(0, 0, "ko")
        lim = 0.5
        a3.set_xlim(-lim / 2, lim / 2), a3.set_ylim(0, lim), a3.set_zlim(0, 0.3)
        a3.set_xlabel("y"), a3.set_ylabel("x"), a3.set_zlabel("z"), a3.view_init(25, -60)
        a3.set_title(f"{board.san(move)}  t={i / C.CONTROL_HZ:.1f}s")
        a2.set_xlim(-0.3, 0.3), a2.set_ylim(-0.05, 0.5), a2.set_aspect("equal"), a2.set_title("top view (green = tip path)")

    OUT.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, len(hist), max(1, len(hist) // 120)))
    FuncAnimation(fig, lambda k: draw(idx[k]), frames=len(idx)).save(str(OUT / f"{move.uci()}.gif"), writer=PillowWriter(fps=15))
    held = np.where(holding)[0]
    for name, i in [("start", 0), ("carry", int(held[len(held) // 2]) if len(held) else 0), ("end", len(hist) - 1)]:
        draw(i)
        fig.savefig(OUT / f"{move.uci()}_{name}.png", dpi=110)
    print(f"wrote {OUT / (move.uci() + '.gif')}  (transfers: {[t.label for t in transfers]}; first: {first.label}; {moving})")


if __name__ == "__main__":
    main()
