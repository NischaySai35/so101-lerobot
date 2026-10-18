"""Read the real board from the webcam: which piece is on which square (YOLO, majority over several frames).

  python -m chess_player.vision.read_board          # print what it sees
  python -m chess_player.vision.read_board --save   # ...and, after you confirm, make it the real-board position
                                                     #    (data/board_state.json, used by move_piece / the pipeline)

Needs the webcam corners (vision/ext_camera corners) to match the current camera position. Uses our own YOLO model
(data/yolo/best.pt) when trained, else the public pretrained chess model. A confirmed reading is also saved as
training data (data/dataset/), so our own model learns this exact board and camera.
"""
import argparse
import collections
import json
import time

import chess

from chess_player import board_state
from chess_player import config as C


def read_position(cam, n=5, gap_s=0.4, turn=chess.WHITE):
    """(board, unsure_squares, last_frame) from n webcam frames: per square, the piece seen most often."""
    from chess_player.vision import detect_board as D
    from chess_player.vision.ext_camera import view

    votes = collections.defaultdict(collections.Counter)
    frame = None
    for i in range(n):
        if i:
            time.sleep(gap_s)
        frame = cam.grab(max_age_s=2.0)
        dets = D.detect_pieces(frame, view(frame))
        for sq in chess.SQUARES:
            votes[sq][dets[sq][0] if sq in dets else "."] += 1
    b = chess.Board(None)
    unsure = []
    for sq, c in votes.items():
        sym, k = c.most_common(1)[0]
        if k < n * 0.6:
            unsure.append(chess.square_name(sq))
        if sym != ".":
            b.set_piece_at(sq, chess.Piece.from_symbol(sym))
    b.turn = turn
    b.castling_rights = b.clean_castling_rights()
    return b, sorted(unsure), frame


def save_training_sample(frame, board):
    from chess_player.vision.capture_dataset import DS, write_sample

    stem = time.strftime("sync_%Y%m%d_%H%M%S")
    n, _ = write_sample(stem, frame, board, "ext")
    (DS / "meta" / f"{stem}.json").write_text(json.dumps({"fen": board.fen(), "position": stem, "camera": "ext"}))
    return stem, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="confirm and store as the real-board position")
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    from chess_player.vision import detect_board as D
    from chess_player.vision.ext_camera import ExtCamera

    kind = D.available()
    if kind is None:
        raise SystemExit("no YOLO model found")
    if not C.EXT_CORNERS.exists():
        raise SystemExit("webcam corners not set: python -m chess_player.vision.ext_camera corners")
    cam = ExtCamera()
    try:
        print(f"reading the board ({kind} YOLO model, {a.frames} webcam frames)...")
        board, unsure, frame = read_position(cam, a.frames)
    finally:
        cam.close()
    print(board_state.show(board))
    print(f"{len(board.piece_map())} pieces" + (f"; unsure squares: {' '.join(unsure)}" if unsure else ""))
    if not board.is_valid():
        print(f"note: not a legal chess position as read ({board.status().name}); fine for pick & place tests")
    if not a.save:
        return
    ans = input("Is this the real board? [y = save / n = cancel]: ").strip().lower()
    if ans != "y":
        print("not saved. Fix it with: python -m chess_player.move_piece --pieces \"...\"  (or Edit position in the window)")
        return
    board_state.save(board)
    stem, n = save_training_sample(frame, board)
    print(f"saved as the real-board position; also kept as YOLO training data ({stem}, {n} pieces)")


if __name__ == "__main__":
    main()
