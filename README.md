# SO-101 chess robot

An SO-101 follower arm that plays chess against a person on a real board. The robot plays **white**, you play
**black**. Two cameras read the board:

- a fixed USB webcam on your side;
- the camera on the robot's gripper (the "robot eye").

YOLO finds the pieces and Stockfish picks the moves. The arm makes its own moves with a pick-and-place motion
learned from teleop demos. **Robo**, a voice assistant, follows spoken or typed commands and answers questions. Its automatic game
commentary is off by default (`VOICE_COMMENTARY` in config).

All the commands are in **[commands.md](commands.md)**. This page explains how it fits together.

## How it works, in plain English

Think of the robot as four helpers working together:

1. **Eyes: two cameras + YOLO.** A webcam on your side of the table and a small camera on the robot's gripper
   (the "robot eye") film the board. **YOLO** ("You Only Look Once") is an AI that finds objects in a picture and
   draws a box around each one: here, every chess piece, with its type and colour. A calibration tells the program
   where each of the 64 squares is in each camera's picture, so every box becomes "white knight on g1". The two
   cameras check each other: pieces hidden from the webcam behind others are seen by the robot eye, and a piece's
   type only changes when both agree.
2. **Brain: Stockfish.** Stockfish is a free chess engine, one of the strongest in the world (much stronger than
   any human). Given the position, it returns the best move in about a second. Its strength can be turned down
   for friendlier games (`--skill 0-20`). **python-chess** keeps the official game: whose turn it is, which moves
   are legal, check, mate.
3. **Hand: the SO-101 arm.** A small 6-motor robot arm. To move a piece, the program works out where the gripper
   has to be in 3-D (board position → robot position, from a calibration where the gripper touched the board's
   corners), then which angle each motor needs for that (**inverse kinematics**, using a 3-D model of the arm).
   How to grab, how far to open the gripper, how high to lift and how to let go were **learned from teleop
   demonstrations**: a person moved the robot through real moves with a second "leader" arm while everything was
   recorded, and the recordings were turned into numbers. Before every move a **collision check** makes sure the
   fingers won't knock over a neighbouring piece; if they would, Robo asks you to make that move.
4. **Voice: Robo.** Optional. It understands spoken commands ("robot, play", "robot, knight f6") through
   **Whisper** (speech → text), speaks with **Piper** (text → speech), and answers questions about the game and
   what the cameras see with **Qwen3.5-2B**, a small vision-language model running on the laptop.

**One move, start to finish:**

- **The robot's turn.**
  1. Stockfish picks the move, e.g. Nf3.
  2. The planner checks that the arm can reach it and won't hit anything.
  3. The arm opens the gripper over g1, steps sideways (one finger is fixed, so the open finger has to go around
     the piece), goes down, closes and checks that it is really holding the piece.
  4. It lifts straight up, carries the piece high, lowers it on f3 and opens just enough to let go.
  5. It goes back to its resting pose, and the cameras check the from and to squares look right.
- **Your turn.** You move a piece by hand. When nothing has moved for about a second (your hand is gone), the
  program compares what the cameras see with every legal move and picks the one that matches. If what it sees
  isn't legal (say you're in check), it tells you why and lists your legal moves.
- **The window** shows everything live: both cameras with their boxes, the game board (hover a piece to see its
  moves and what each camera reads there), whose turn it is, Stockfish's evaluation, a log, and a red **STOP**
  button that freezes the arm instantly.

## Technologies

| What | Used for |
|---|---|
| **SO-101 arm** + **Feetech STS3215** servo motors | the robot (a follower arm, plus a leader arm for teaching) |
| **LeRobot** (Hugging Face, Seeed fork) | motor drivers, arm calibration, teleoperation |
| **placo** + the SO-101 **URDF** (3-D model) | inverse kinematics: gripper position ↔ motor angles |
| **Stockfish 19** | chess engine: picks the robot's moves, evaluates the position |
| **python-chess** | game rules, legal moves, move notation |
| **YOLOv8m** (Ultralytics) | piece detection; started from the public pretrained model *KanisornPutta/chess-model-yolov8m*, fine-tuned on 105 photos of this board (21 positions, labelled automatically) → `data/yolo/best.pt` |
| **OpenCV** | cameras, board grid (homography), image processing |
| **Whisper-small** (OpenAI, via transformers) | speech recognition for Robo |
| **Piper** (`en_US-ryan-medium`) | Robo's voice |
| **Qwen3.5-2B** (vision-language model) | Robo's answers to questions, with both camera pictures |
| **PyTorch + CUDA** | runs YOLO, Whisper and Qwen on the laptop's GPU |
| **Python HTTP server + HTML/JS page** (Chrome app window) | the pipeline window |

## Status (2026-10-18)

| Part | State |
|---|---|
| Arm setup, calibration, teleop (leader → follower) | ✅ done |
| Board ↔ robot calibration | ✅ 3.9 mm fit error |
| Pick & place (grasp at the piece base, collision check, drop detection + retry) | ✅ works |
| Tight-spot grasp (pieces right next to it), learned from 3 crowded teleop demos | ✅ added 2026-10-18, check on the real arm |
| Pick correction (6 mm toward the arm, 4 mm toward the fixed finger, at the pick only) | ✅ added 2026-10-18 |
| Arm speed | ✅ 0.8× by default (`ARM_SPEED` in config) |
| Webcam board reading (YOLO) | ✅ works |
| Robot-eye second opinion (both cameras agree before a piece's type changes) | ✅ added 2026-10-18 |
| King/queen height check | ✅ added 2026-10-18 |
| Pipeline window (game, board, cameras, Robo, STOP, torque) | ✅ works |
| Robo voice (Whisper + Piper + Qwen3.5-2B) | ✅ works, short replies; mic off by default |
| Own YOLO model fine-tuned on this board (105 photos, 21 positions) | ✅ 2026-10-18: ~98 % of pieces found and named correctly on unseen positions |
| Start position snap (king e1 / queen d1 whatever the cameras guess) | ✅ added 2026-10-18 |
| Post-move check (only the squares the robot's move changed) | ✅ added 2026-10-18 |

## Quick start

```bash
cd ~/so101-lerobot && source scripts/env.sh   # every terminal
scripts/run_pipeline.sh                        # arm home, cameras, YOLO, Stockfish, window
```

The window starts **paused**:

1. Check that the board on the right matches the real board. Hover a square to see what each camera reads there.
   If something is wrong, use ⟳ Recheck or ✎ Edit position.
2. Pick who moves first: **first: White / Black** under the board.
3. Press **▶ Play** (or Space).

Only one program can use the arm and each camera at a time. Close the pipeline before teleop, `move_piece`,
`record_demo` or `wrist_stream.py`, and the other way round.

## How a game works

- **Robot's turn:** Stockfish picks a move. The arm picks the piece up, lifts it straight up, carries it high,
  lowers it and lets go. Captured pieces go off the board first; when castling, the order depends on the side.
  Next to other pieces it uses a tighter grasp learned from teleop demos (opens less, shifts less, lets go
  with a small opening). If the path would still hit a piece, or a grasp fails, Robo asks you to make that move
  by hand. Then press
  **Done, I moved it**.
- **Your turn:** move a piece on the board. The webcam sees the change and works out your move. You can also type
  it (`e5`, `Nf6`, `e7e5`) and press **Send my move**, or say "robot, knight f6".
- **If a piece drops while being carried:** the arm goes up, looks with the webcam, finds the piece and tries
  again (twice at most).
- **■ STOP** (or Esc) freezes the arm right away. **Torque off** lets it go limp, so hold it first.

## How the board is read

The two cameras check each other:

1. **Webcam** (every 0.3 s): YOLO boxes → base point of each piece → square, through the webcam's corner
   calibration. A reading counts once 3 frames in a row agree on which squares are occupied
   (no hand over the board); each square's piece type is a majority vote over the last 6 frames.
2. **Robot eye**: only while the arm rests at its look pose. YOLO runs on the gripper image too, and its pieces are
   matched to the webcam's by colour and position. It also adds pieces the webcam can't see (e.g. black pawns
   hidden behind the back rank), if it sees them steadily and well inside a square.
3. **Combining them:**
   - Which squares are occupied, and by which colour, comes from the webcam (it sees every square).
   - The **type** of a known piece only changes when both cameras agree, or when the webcam alone is very sure and
     the robot eye can't see that square.
   - A known king is never lost.
   - A "pawn" that is far too tall, or a pawn on the first or last rank, is treated as a misreading.
4. **King vs queen:** the pieces are 64 mm vs 58 mm tall. When YOLO isn't sure (below 85 %), the height measured
   from the webcam image decides.

**Live sync** (on by default) keeps the game board matching the real board while no game is running. A change is
adopted only after the cameras have shown it for 1.2 s, so a flickering square doesn't flip the board. When the
cameras see the starting setup, the board becomes the exact standard start (king e1, queen d1, White to move).
Red squares on the board mean the cameras see something different there. Orange outlines mean an unsure reading.

**The robot eye calibrates itself.** At startup, the webcam's reading of the pieces is matched to the gripper
image. The grid is then lined up with the light and dark squares and snapped to the square corners. It is only
re-fitted when the saved fit stops matching. To redo it, use ⟳ calibrate on the Robot eye tile. Check the result
in `data/wrist_corners_check.jpg`.

**Our own YOLO model.** The public chess model was fine-tuned on this board, these pieces and this webcam angle
(`data/dataset/`: 105 photos of 21 positions; the labels are computed from the known position, no hand drawing).
40 epochs; it finds and names ~98 % of the pieces on positions it never saw. To improve it, add positions and
retrain (more variety helps; more epochs don't):

1. `python -m chess_player.vision.capture_dataset --presets 1-21 --shots 5` shows each position to set up (as seen
   from your seat), Enter takes 5 labelled shots.
2. `python -m chess_player.vision.train_yolo` writes `data/yolo/best.pt`, which the pipeline uses automatically.

Every corrected reading you **Apply** in the position editor is also saved as training data.

## Calibration files (`data/`)

| File | What | Redo when |
|---|---|---|
| `ext_corners.json` | webcam: board corners in its image | the webcam or the board moved |
| `wrist_corners.json` | robot eye: board corners in the gripper image at the look pose (automatic) | done automatically; ⟳ calibrate |
| `board_to_robot.json` | board ↔ robot transform (corner touches with the jaw tip) | the board moved relative to the robot |
| `grip.json` | gripper closing for small / big pieces | new pieces |
| `look_pose.json`, `poses/home_pose.json` | where the arm rests and looks | — |
| `board_state.json` | the real board's current position (shared by the window and `move_piece`) | updated automatically |
| `calibration_backup/` | copies of the LeRobot arm calibrations | copy back if `~/.cache/huggingface` is wiped |

Motion and vision settings are all in `src/chess_player/config.py`. The ones you're most likely to change:
`ARM_SPEED`, `PICK_EXTRA_DOWN_MM` / `PICK_EXTRA_LEFT_MM`, `JAW_SHIFT_MM`, `GRASP_HEIGHT_MM`, `SYNC_TYPE_CONF`,
`KQ_HEIGHT_CHECK`, `VOICE_QUIPS`, `VOICE_SNR`.

## Layout

| Path | What |
|---|---|
| `scripts/` | `env.sh` (conda env + `PYTHONPATH`), `run_pipeline.sh`, `teleop*.sh`, `calibrate_*.sh` |
| `src/chess_player/ui/` | pipeline server (`server.py`, port 8770) + window (`index.html`) |
| `src/chess_player/game/` | session (arm + cameras + engine), move inference / board sync (`state.py`), Stockfish |
| `src/chess_player/motion/` | arm control (`arm.py`), IK (placo, URDF), pick & place + collision check (`pick_place.py`) |
| `src/chess_player/vision/` | webcam (`ext_camera.py`), YOLO (`detect_board.py`), robot eye (`wrist_view.py`), dataset capture + training |
| `src/chess_player/voice/` | Robo: speech in/out (`audio.py`), commands (`assistant.py`), language model (`brain.py`) |
| `src/chess_player/calib/`, `teach/` | calibration tools; teleop demo recording and analysis |
| `src/tools/` | arm utilities: `go_home`, `read_positions`, `release`, `scan_motors`, `wrist_stream` (share the gripper camera) |
| `data/` | calibrations, demos, datasets, models (Piper voice, pretrained YOLO) |
| `bin/stockfish`, `external/chess-player` | engine; teammate's simulator repo |
| `setup/` | `install_lerobot.sh`, udev rule `99-so100.rules` |

## Fresh clone

```bash
git clone --recurse-submodules <this repo> ~/so101-lerobot && cd ~/so101-lerobot
setup/install_lerobot.sh              # conda env `lerobot` + LeRobot (Ubuntu 24.04), once
setup/download_models.sh              # Stockfish, Piper voice, pretrained chess YOLO (not stored in git)
source scripts/env.sh && pip install --no-deps -e .
python -m chess_player.vision.train_yolo   # rebuild our YOLO model (data/yolo/best.pt) from data/dataset
```

Not in git (see `.gitignore`): model weights (`*.pt`, `*.onnx`), Stockfish, demo videos, training runs and
other generated files, and `data/board_state.json` (this machine's live board). The calibration files in `data/`
(board ↔ robot, webcam / robot-eye corners, grip, look pose) are for this exact rig: redo them on another setup.
The arm calibrations LeRobot reads live in `~/.cache/huggingface/...`; copies are in `data/calibration_backup/`.
`external/chess-player` is a git submodule (Nithya-shree182/chess-player).

## Hardware

| Thing | Details |
|---|---|
| Arms | SO-101 leader + follower, Feetech STS3215 servos, IDs 1–6: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| Driver boards | CH343 USB-serial (`1a86:55d3`), 1 Mbaud. Follower `5B14031355` → `/dev/so100_follower`; leader `5B14110588` → `/dev/so100_leader` |
| Gripper camera (robot eye) | Sonix "USB2.0_CAM1", 640×480 @ 30 fps, `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0` |
| Webcam | Ausha USB webcam on the human's side, MJPG 1280×720, `/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0`; white balance matched to the gripper camera |
| Board | 33.3 mm squares, 2.34 cm border; the fixed jaw finger is on the left |

## Software

- Conda env `lerobot` (Python 3.12) in `~/miniforge3`; LeRobot 0.6.2 from `~/lerobot` (Seeed fork), installed with `pip install -e ".[feetech]"`
- This repo is pip-installed in editable mode (`pip install --no-deps -e ~/so101-lerobot`), so `python -m chess_player...` works from anywhere
- torch 2.11 + CUDA, ultralytics (YOLO), placo (IK), python-chess, Stockfish, transformers (Whisper-small, Qwen3.5-2B), Piper TTS (`en_US-ryan-medium`)
- Keep **numpy 2.2.6** and **opencv-python-headless 4.13** pinned: newer ultralytics installs break them
- LeRobot calibration: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json` and `.../teleoperators/so_leader/leader_arm.json`

One-time udev rule (fixed port names for both arms):

```bash
sudo cp setup/99-so100.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Known issues / lessons

- **USB dropouts** ("disabled by hub (EMI?)"): scripts fail with `Input/output error` or `Missing motor IDs`. Use a
  short good cable and a solid servo power supply, and keep USB away from the servo wires. Check with
  `journalctl -k --since -5min | grep -i usb`.
- **Teleop start:** the follower jumps to the leader's pose. Line them up first. Stopping teleop turns torque off,
  so support the arm.
- **After changing code, restart the pipeline** (Ctrl+C, `scripts/run_pipeline.sh`). The window's page reloads
  on its own, but the server doesn't.
- **YOLO misreads** (king ↔ queen, pawn ↔ bishop) come from the pretrained model. The two-camera check and the
  height check catch most of them. Fine-tuning is the real fix.
- **The collision check can refuse a move** in a crowded spot (the open fingers would touch a neighbour). Robo
  then asks you to make that move by hand.
- **Robo's language model is small (2B).** Its answers about what it "sees" are unreliable. Game commands use
  fixed rules, never the model.
