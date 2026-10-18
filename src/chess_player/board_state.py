"""The position of the REAL board, shared by every tool (move_piece, the pipeline window).

data/board_state.json = {"fen": ..., "updated": ...}. The collision check needs to know where every piece is, so
keep this matching the physical board: move_piece and the pipeline update it after each move; set it with
`move_piece --pieces "..."`, `--fen`, or the window's Edit position.
"""
import json
import re
import time

import chess

from chess_player import config as C

PATH = C.DATA / "board_state.json"


def load(default_start=True):
    if PATH.exists():
        try:
            return chess.Board(json.loads(PATH.read_text())["fen"])
        except (ValueError, KeyError):
            pass
    return chess.Board() if default_start else None


def save(board: chess.Board):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fen": board.fen(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    tmp.replace(PATH)


def from_pieces(text, turn=chess.WHITE):
    """'Re3 Ke1 ke8 Pa2 pb7' -> Board (capital letter = white piece, small = black; 'e4' alone = white pawn)."""
    b = chess.Board(None)
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"([KQRBNPkqrbnp]?)([a-h][1-8])", tok)
        if not m:
            raise ValueError(f"can't read '{tok}': use e.g. Re3 (white rook e3) or ke8 (black king e8)")
        b.set_piece_at(chess.parse_square(m.group(2)), chess.Piece.from_symbol(m.group(1) or "P"))
    b.turn = turn
    return b


def show(board: chess.Board, human_side=None):
    """ASCII board from the human's seat (bottom row = nearest the human)."""
    if human_side is None:
        human_side = chess.WHITE
        if C.EXT_CORNERS.exists():
            from chess_player.vision.ext_camera import robot_color_from_corners

            human_side = chess.BLACK if robot_color_from_corners() == "white" else chess.WHITE
    ranks = range(8) if human_side == chess.BLACK else range(7, -1, -1)
    files = range(7, -1, -1) if human_side == chess.BLACK else range(8)
    lines = [f"  {r + 1}  " + " ".join(str(board.piece_at(chess.square(f, r)) or ".") for f in files) for r in ranks]
    lines.append("     " + " ".join("abcdefgh"[f] for f in files) + "    (bottom row = your side; CAPITALS = white)")
    return "\n".join(lines)
