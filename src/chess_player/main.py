"""Play a game: human vs robot. The arm MOVES (unless --dry-run).

  python -m chess_player.main --dry-run            # no hardware: type your moves, the arm is simulated
  python -m chess_player.main                      # camera sees your move; Enter after moving, or type it
  python -m chess_player.main --keyboard           # real arm, but you type your moves
  python -m chess_player.main --agent              # Claude orchestrates and commentates (needs ANTHROPIC_API_KEY)

At the human prompt: Enter = "I moved, look", or type a move (e4, Nf3, e2e4, O-O), 'undo', 'board', 'quit'.
"""
import argparse
import select
import sys


from chess_player import config as C
from chess_player.game import state
from chess_player.game.session import Session
from chess_player.motion.ik import Unreachable
from chess_player.motion.pick_place import GraspFailed


def read_line_or_watch(s: Session, prompt):
    """input(), except that with the board watcher running a move seen on the board also ends the wait.
    Returns (typed_text, inference_or_None)."""
    if s.watcher is None:
        return input(prompt), None
    print(prompt + "(watching the board...)", end="", flush=True)
    while True:
        if select.select([sys.stdin], [], [], 0.5)[0]:
            return sys.stdin.readline(), None
        inf = s.poll_human_move()
        if inf is not None:
            print()
            return "", inf


def human_turn(s: Session, keyboard: bool):
    while True:
        if keyboard or not s.has_vision:
            text, seen = input("\nYour move (type it): "), None
        else:
            text, seen = read_line_or_watch(s, "\nYour move (just play it, or type it; Enter = look now): ")
        if seen is not None:
            san = s.board.san(seen.move)
            s.push_human(seen.move)
            return san
        text = text.strip()
        if text == "quit":
            raise KeyboardInterrupt
        if text == "board":
            print(s.board)
            continue
        if text == "undo" and len(s.board.move_stack) >= 2:
            s.board.pop(), s.board.pop()
            print("took back the last two moves (put the pieces back by hand)\n", s.board)
            s.snapshot() if s.has_vision else None
            continue
        if text:
            try:
                move = state.parse_human_move(s.board, text)
            except ValueError as e:
                print(f"  not a legal move here: {e}")
                continue
            san = s.board.san(move)
            s.push_human(move)
            if s.has_vision:
                s.snapshot()  # refresh the 'before' image for the change detector
            return san
        if keyboard or not s.has_vision:
            continue
        inf = s.see_human_move()
        if inf.move is None:
            cands = ", ".join(f"{s.board.san(m)}({sc:.1f})" for m, sc in inf.ranked[:3])
            print(f"  not sure what you played ({inf.reason}). Candidates: {cands}. Type it, or fix the board and press Enter.")
            continue
        san = s.board.san(inf.move)
        s.push_human(inf.move)
        return san


def robot_turn(s: Session):
    move = s.engine_move()
    san = s.board.san(move)
    print(f"\nRobot plays {san} ({move.uci()})")
    for st in s.check(move).get("steps", []):
        print(f"  plan: {st['step']}: " + (f"travel {st['z_travel_mm']} mm, grasp tilt {st['max_grasp_tilt_deg']} deg, "
                                           f"~{st['est_seconds']} s" if st["ok"] else f"NOT POSSIBLE ({st['details']})"))
    try:
        done = s.execute(move)
        print("  done:", "; ".join(done))
    except Unreachable as e:
        print(f"  can't reach: {e}\n  Please play {san} for me by hand, then press Enter.")
        input()
        s.push_robot_by_hand(move)
    except GraspFailed as e:
        print(f"  grasp failed: {e}\n  Please finish {san} by hand (check nothing else moved), then press Enter.")
        input()
        s.push_robot_by_hand(move)
    if s.has_vision:
        ok, details = s.verify_position()
        if not ok:
            print(f"  board doesn't look right: {details}. Fix it by hand and press Enter.")
            input()
            s.snapshot()
    return san


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keyboard", action="store_true", help="type human moves instead of using the camera")
    ap.add_argument("--robot-color", choices=["white", "black"], help="default: read from the webcam corners")
    ap.add_argument("--skill", type=int, help="Stockfish skill level 0-20")
    ap.add_argument("--agent", action="store_true")
    a = ap.parse_args()
    if a.robot_color is None:
        if C.EXT_CORNERS.exists():
            from chess_player.vision.ext_camera import robot_color_from_corners

            a.robot_color = robot_color_from_corners()
        else:
            a.robot_color = C.ROBOT_COLOR

    s = Session(dry_run=a.dry_run, robot_color=a.robot_color, skill=a.skill, use_camera=not (a.keyboard or a.dry_run))
    print(f"vision: {s.vision if s.has_vision else 'off'}   robot plays {a.robot_color}   dry run: {a.dry_run}")
    try:
        if a.agent:
            from chess_player.agent.orchestrator import run

            return run(s)
        if s.has_vision:
            s.snapshot()
        while not s.board.is_game_over():
            if s.human_to_move():
                print(f"  you played {human_turn(s, a.keyboard)}")
            else:
                robot_turn(s)
            print(s.board, f"\n  eval {s.engine.evaluate(s.board)}")
        print("Game over:", s.board.result(), s.board.outcome().termination.name)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        s.close()


if __name__ == "__main__":
    main()
