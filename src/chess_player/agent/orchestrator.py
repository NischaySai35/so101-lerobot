"""Claude as orchestrator and commentator. Stockfish picks the moves; this layer runs the turn sequence,
recovers from failures by talking to the human, and commentates.

Each turn (one human move + one robot move) is a fresh, short conversation seeded with the game state, so
context stays small for a whole game. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
import anthropic

from chess_player.agent.tools import TOOLS, ToolBox, to_content

MODEL = "claude-opus-5-5"

SYSTEM = """You run a physical chess robot (SO-101 arm with a wrist camera) playing against a human.
Stockfish chooses the robot's moves; you never choose or change a chess move yourself.

Each request covers one turn cycle. Use the tools to:
1. If it's the human's turn: wait_for_human_move. If vision isn't sure, ask the human which move they made
   (offer the candidates) and record it with confirm_human_move.
2. If it's the robot's turn: get_engine_move, then execute_robot_move with exactly that move.
   - "unreachable" or "grasp_failed": ask the human to make or finish the move by hand, then
     robot_move_done_by_hand.
   - board_check doesn't match: tell the human what looks wrong and ask them to fix it.
3. Stop when the robot has moved (or the game is over).

Talk like a friendly, concise chess opponent: one or two short sentences of commentary per move, in plain
text (it is printed in a terminal). Mention the evaluation only when it changes a lot."""


def run_turn(client, tools: ToolBox, log=print):
    st = tools.get_game_state()
    messages = [{"role": "user", "content": f"Game state: {to_content(st)}\nRun the next turn cycle."}]
    for _ in range(30):  # hard cap on tool rounds per turn
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},  # routine orchestration; keeps turns snappy
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            log("[agent] request declined; finishing this turn without the agent")
            return False
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                log(f"[robot] {b.text.strip()}")
        if resp.stop_reason != "tool_use":
            return True
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = tools.run(b.name, dict(b.input))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": to_content(out),
                                "is_error": "error" in out})
        messages.append({"role": "user", "content": results})
    log("[agent] too many tool calls this turn; stopping")
    return False


def run(session, log=print):
    client = anthropic.Anthropic()
    tools = ToolBox(session)
    if session.cam:
        session.snapshot()
    while not session.board.is_game_over():
        try:
            ok = run_turn(client, tools, log)
        except anthropic.APIConnectionError:
            log("[agent] no connection to the API: continue without the agent (python -m chess_player.main)")
            return
        except anthropic.AuthenticationError:
            log("[agent] no valid ANTHROPIC_API_KEY: run without --agent")
            return
        if not ok:
            return
        print(session.board)
    log(f"[robot] Game over: {session.board.result()}")
