"""Stockfish via python-chess. One engine process for the whole game."""
import chess
import chess.engine

from chess_player import config as C


class Engine:
    """Stockfish that restarts itself if it dies (e.g. after being handed an impossible position)."""

    def __init__(self, path=C.STOCKFISH, skill_level=None):
        self.path, self.skill = path, skill_level
        self._start()

    def _start(self):
        self.e = chess.engine.SimpleEngine.popen_uci(str(self.path))
        if self.skill is not None:  # 0..20, lower = weaker (nice for demos)
            self.e.configure({"Skill Level": int(self.skill)})

    def _call(self, fn, board, *args):
        if not board.is_valid():
            raise ValueError(f"not a legal chess position ({board.status().name}): fix it in Edit position")
        try:
            return fn(board, *args)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            self._start()
            return fn(board, *args)

    def best_move(self, board: chess.Board, seconds=C.ENGINE_TIME_S) -> chess.Move:
        return self._call(lambda b, t: self.e.play(b, chess.engine.Limit(time=t)).move, board, seconds)

    def evaluate(self, board: chess.Board, seconds=0.3) -> str:
        """Human-readable eval from White's side, e.g. '+0.35' or 'mate in 3'."""
        info = self._call(lambda b, t: self.e.analyse(b, chess.engine.Limit(time=t)), board, seconds)
        s = info["score"].white()
        return f"mate in {s.mate()}" if s.is_mate() else f"{s.score() / 100:+.2f}"

    def close(self):
        self.e.quit()
