"""Move one piece with the real arm: the motion learned from the demos, collision check always on. The arm MOVES
(after you confirm).

  python -m chess_player.move_piece e3 e5                         # uses the saved real-board position
  python -m chess_player.move_piece e3 e5 --pieces "Re3 Ke1 ke8"   # say where the pieces are (and save it)
  python -m chess_player.move_piece e3 e5 --fen "<FEN>"            # same, as a FEN
  python -m chess_player.move_piece e3 e5 --from-camera            # read the real board from the webcam first
  python -m chess_player.move_piece --show                         # print the saved position
  python -m chess_player.move_piece e3 e5 --dry-run                # plan + report only

Any from/to squares work (not a chess-rule check). The position (data/board_state.json, shared with the
pipeline window) must match the real board, because the collision check avoids the pieces it lists. After a
successful move the position is updated.
"""
import argparse

import chess

from chess_player import board_state
from chess_player import config as C
from chess_player.geometry import BoardToRobot, square_center_mm
from chess_player.motion.arm import Arm, FakeArm
from chess_player.motion.ik import IK, Unreachable
from chess_player.motion.pick_place import GraspFailed, PickPlace, Transfer, obstacles_for


class _NoSlots:
    occupied = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--pieces", help='where the pieces are, e.g. "Re3 Ke1 ke8 Pa2" (capital = white, small = black)')
    ap.add_argument("--fen", help="the position as a FEN ('start' = normal start position)")
    ap.add_argument("--show", action="store_true", help="print the saved position and exit")
    ap.add_argument("--from-camera", action="store_true", help="read the real board with the webcam (YOLO) first")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.from_camera:
        from chess_player.vision.ext_camera import ExtCamera
        from chess_player.vision.read_board import read_position, save_training_sample

        cam = ExtCamera()
        try:
            board, unsure, frame = read_position(cam)
        finally:
            cam.close()
        print(board_state.show(board))
        print("read from the webcam" + (f"; unsure squares: {' '.join(unsure)}" if unsure else ""))
        if input("Is this the real board? [y/n]: ").strip().lower() != "y":
            raise SystemExit('cancelled: give the position with --pieces "..." instead')
        save_training_sample(frame, board)
        board_state.save(board)
        print("saved as the real-board position (and as YOLO training data)")
    elif a.pieces:
        board = board_state.from_pieces(a.pieces)
    elif a.fen:
        board = chess.Board() if a.fen == "start" else chess.Board(a.fen)
    else:
        board = board_state.load()
    if a.pieces or a.fen:
        board_state.save(board)
        print("saved as the real-board position")
    if not a.from_camera:
        print(board_state.show(board))
    if a.show or not (a.src and a.dst):
        if not a.show:
            print("\ngive two squares, e.g.  python -m chess_player.move_piece e3 e5")
        return
    s, d = chess.parse_square(a.src), chess.parse_square(a.dst)
    piece = board.piece_at(s)
    if piece is None:
        raise SystemExit(f"\nno piece on {a.src} in this position. Tell me where the pieces really are, e.g.\n"
                         f'  python -m chess_player.move_piece {a.src} {a.dst} --pieces "R{a.src} Ke1 ke8"')
    if board.piece_at(d) is not None:
        raise SystemExit(f"\n{a.dst} is occupied ({board.piece_at(d)}) in this position: clear it, or fix the position "
                         f"with --pieces")
    t = Transfer(piece.symbol(), tuple(square_center_mm(s)), tuple(square_center_mm(d)), f"{piece.symbol()} {a.src}{a.dst}")
    (t, obst), = obstacles_for(board, _NoSlots(), [t])
    T = BoardToRobot.load()
    arm = FakeArm() if a.dry_run else Arm()
    pp = PickPlace(arm, IK(), T, verify_grasp=not a.dry_run)
    try:
        try:
            info = pp.plan_transfer(t, obst)["info"]
        except Unreachable as e:  # includes Blocked (would hit a piece)
            raise SystemExit(f"\ncan't do it safely: {e}")
        print(f"\nplan: {info['step']}: collision check ON ({len(obst)} other pieces), shift {info['jaw_shift_mm']} mm, "
              f"{info['route']}, up to {info['z_travel_mm']} mm, grasp tilt {info['max_grasp_tilt_deg']} deg, "
              f"~{info['est_seconds']} s, grip {C.PIECES[piece.symbol().lower()]['grip']} %")
        if a.dry_run:
            return
        input("Arm will move. Keep a hand near it. Enter to go (Ctrl+C to cancel)...")
        try:
            pp.execute(t, obstacles=obst)
            board.remove_piece_at(s)
            board.set_piece_at(d, piece)
            board_state.save(board)
            print("placed; real-board position updated")
        except GraspFailed as e:
            print(f"grasp failed: {e} (position not changed)")
        pp.go_look()
    finally:
        arm.close()


if __name__ == "__main__":
    main()
