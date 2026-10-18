# Commands

Every terminal first (activates the `lerobot` env and the project):

```bash
cd ~/so101-lerobot && source scripts/env.sh
```

Only one program can use the follower arm at a time: stop the pipeline / teleop / any arm script before starting another.

## Play

| What | Command |
|---|---|
| Whole pipeline + window (arm home, cameras, YOLO, Stockfish; starts paused, press Play / Space) | `scripts/run_pipeline.sh` |
| Same with a simulated arm | `scripts/run_pipeline.sh --dry-run` |
| Weaker engine for demos (0-20) | `scripts/run_pipeline.sh --skill 5` |
| Terminal-only game (type your moves) | `python -m chess_player.main --keyboard` |

## Robo: voice, chat and the vision-language model (inside the pipeline)

Starts with `scripts/run_pipeline.sh` (first start loads Qwen3.5-2B, ~20 s). Say **"robot"** / **"robo"** first;
typed messages in the window's **Robo** box don't need it.

| Say / type | Does |
|---|---|
| "robot, play" / "pause" / "go home" / "new game" / "undo" | game controls |
| "robot, move e2 to e4" | manual pick & place (any squares) |
| "robot, I played knight f6" / "robot, knight f6" / "robot, I moved" | your move (typed/said, or "look now") |
| "robot, done" | after you made a move for it by hand |
| "robot, what can you see?" / "who's winning?" / anything else | answered by Qwen3.5 with both camera images |
| "robot, be quiet" / "robot, talk again" | mute / unmute its voice |

| Option | Command |
|---|---|
| Pipeline without Robo | `scripts/run_pipeline.sh --no-voice` |
| Robo talks and answers typed questions, mic off | `scripts/run_pipeline.sh --no-mic` |
| Start with Robo muted | `scripts/run_pipeline.sh --mute` |
| Without the webcam (e.g. while another program uses it) | `scripts/run_pipeline.sh --no-webcam` |
| Robo panel (left column) | voice on/off, volume, speed, stop talking, test voice, mic on/off + level meter, sensitivity, wake word on/off |

The phone webcam serves **one viewer at a time**: close the pipeline before `record_demo`, and close browser tabs
showing `192.168.101.104:8080` before starting the pipeline.

## Teleop and teaching (leader arm moves the follower)

| What | Command |
|---|---|
| Plain teleop (follower jumps to the leader's pose: line them up first) | `scripts/teleop.sh` |
| Teleop with the gripper camera in Rerun | `scripts/teleop_with_camera.sh` |
| **Record a demonstration**: teleop + both cameras + joints + 6-D tip pose + square + loads, saved as an ACT dataset | `python -m chess_player.teach.record_demo --task "rook e3 to e5"` |
| Record several in a row | `python -m chess_player.teach.record_demo --task "pawn e2 e4 past pieces" --episodes 5` |
| Summary of recorded demos (grasp / peak / release, offsets, heights, timing) | `python -m chess_player.teach.analyze_demo` |
| Train ACT on the demos (GPU) | `lerobot-train --dataset.repo_id=local/chess_demos --dataset.root=data/demos/chess_demos --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=data/act_runs/chess --steps=20000 --batch_size=8` |

In `record_demo`: Enter = start recording, Enter = stop, then keep (Enter) or discard (`n`). The follower follows
the leader the whole time, and eases onto it at the start instead of jumping. Data: `data/demos/chess_demos/`
(videos + parquet), per-frame pose logs in `data/demos/chess_demos/poses/episode_XXX.csv`.

## Board sync (which piece is on which square)

The real-board position lives in `data/board_state.json`, shared by `move_piece` and the pipeline (collision check
uses it). YOLO = our own model once trained (`data/yolo/best.pt`), else a public pretrained chess model.

| What | Command |
|---|---|
| Read the real board from the webcam (prints it from your seat) | `python -m chess_player.vision.read_board` |
| ...and save it as the real-board position (asks to confirm; also becomes YOLO training data) | `python -m chess_player.vision.read_board --save` |
| Read the board, then move a piece | `python -m chess_player.move_piece e3 e5 --from-camera` |
| Say it yourself (capital = white) | `python -m chess_player.move_piece --pieces "Re3 Ke1 ke8"` |
| Show the saved position | `python -m chess_player.move_piece --show` |
| King/queen check | YOLO K/Q below 85 % confidence are re-decided by the measured piece height (`KQ_HEIGHT_CHECK` in config). Hover a square on the board to see the camera's reading, confidence and height; orange outline = unsure |
| Dual-camera check | while the arm rests at the look pose, YOLO also runs on the robot eye (gripper camera); a piece's type only changes when both cameras agree (or the webcam alone is very sure). Hover a square: webcam + robot-eye readings, ✓ agree / ✗ disagree |
| Robot-eye calibration | automatic at startup (re-fitted only if the view no longer matches); redo with ⟳ calibrate on the Robot eye tile, or `python -m chess_player.vision.wrist_view calibrate` while the pipeline runs. Check: `data/wrist_corners_check.jpg` |
| In the window | Board card: 📷 Read board (fix mistakes in the editor, Apply = also training data), ✎ Edit position, ↺ Start |

## Single moves and checks

| What | Command |
|---|---|
| Pick & place one piece (learned motion, collision check on; waits for Enter) | `python -m chess_player.move_piece e2 e4` |
| Same in a given position | `python -m chess_player.move_piece e2 e4 --fen "<FEN>"` |
| Plan only, no motion | `python -m chess_player.move_piece e2 e4 --dry-run` |
| GIF preview of a planned move (`data/sim/`) | `python -m chess_player.sim_preview --move g1f3` |
| Reach map of all 64 squares | `python -m chess_player.calib.reachability_sweep --grasp-tilt 55` |

## Calibration (redo the one whose thing moved)

| What | Command |
|---|---|
| Webcam corners (click a1, h1, h8, a8 playing-area corners) | `python -m chess_player.vision.ext_camera corners` |
| Live webcam with grid / YOLO | `python -m chess_player.vision.ext_camera watch` |
| Board → robot (touch the 4 corners with the closed jaw tip; torque off) | `python -m chess_player.calib.board_to_robot` |
| Re-fit from the saved touches | `python -m chess_player.calib.board_to_robot --refit` |
| Hover test: tip 15 mm over the corner squares (arm moves) | `python -m chess_player.calib.hover_test --height 15` |
| Grip width (jaws on the BASE of a pawn, then a king) | `python -m chess_player.calib.teach_grip` |
| LeRobot follower / leader calibration | `scripts/calibrate_follower.sh` / `scripts/calibrate_leader.sh` |

## YOLO

| What | Command |
|---|---|
| Guided dataset capture (shows each position, Enter = 5 webcam shots, auto-labels) | `python -m chess_player.vision.capture_dataset --presets 1-21` |
| Continue from a position | `python -m chess_player.vision.capture_dataset --presets 8-21` |
| Train (writes `data/yolo/best.pt`) | `python -m chess_player.vision.train_yolo` |
| Detect on the current webcam frame | `python -m chess_player.vision.detect_board` |

## Arm tools

| What | Command |
|---|---|
| Joint positions (read-only) | `python src/tools/read_positions.py` |
| Save the current pose as the home/look pose | `python src/tools/read_positions.py --save data/look_pose.json` |
| Go to the home pose and hold | `python src/tools/go_home.py` |
| Torque off (the arm drops: hold it) | `python src/tools/release.py` |
| Leader's gripper rotation doesn't match the follower's in teleop (wrist roll): line them up by hand, Enter (only the leader's calibration changes) | `python src/tools/align_wrist_roll.py` |
| Follower's jaw opens wider/narrower than the leader's in teleop: squeeze the leader closed, Enter, open, Enter (only the leader's calibration changes) | `python src/tools/align_gripper.py` |
| Make the FOLLOWER's wrist roll + gripper match the leader in teleop (leader stays as is; done 2026-10-18) | `python src/tools/match_follower_to_leader.py --dry-run` (plan) / without `--dry-run` (apply) |
| Ping servos 1-10 | `python src/tools/scan_motors.py` |
| Share the gripper camera on the network (`http://<laptop-ip>:8081/`) | `python src/tools/wrist_stream.py` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named chess_player` | `cd ~/so101-lerobot && source scripts/env.sh` (or `pip install --no-deps -e ~/so101-lerobot`) |
| `Missing motor IDs` / `Incorrect status packet` | another program has the port, or a USB dropout: close it, replug, check `journalctl -k --since -5min \| grep -i usb` |
| Webcam grid doesn't sit on the board | webcam or board moved: redo the webcam corners |
| Arm misses squares | board moved relative to the robot: redo board → robot, then the hover test |
