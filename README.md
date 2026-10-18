# SO-101 chess robot

An SO-101 follower arm that plays chess against a person on a real board. The robot plays **white**, you play
**black**. Two cameras read the board:

- a fixed USB webcam on your side;
- the camera on the robot's gripper (the "robot eye").

YOLO finds the pieces and Stockfish picks the moves. The arm makes its own moves with a pick-and-place motion
learned from teleop demos. **Robo**, a voice assistant, follows spoken or typed commands and answers questions. Its automatic game
commentary is off by default (`VOICE_COMMENTARY` in config).

All the commands are in **[commands.md](commands.md)**. This page explains how it fits together.

## Status (2026-10-18)

| Part | State |
|---|---|
| Arm setup, calibration, teleop (leader → follower) | ✅ done |
| Board ↔ robot calibration | ✅ 3.9 mm fit error |
| Pick & place (grasp at the piece base, collision check, drop detection + retry) | ✅ works |
| Tight-spot grasp (pieces right next to it), learned from 3 crowded teleop demos | ✅ added 2026-10-18, check on the real arm |
| Pick correction (4 mm lower, 4 mm toward the fixed finger at the pick) | ✅ added 2026-10-18, check on the real arm |
| Arm speed | ✅ 0.8× by default (`ARM_SPEED` in config) |
| Webcam board reading (YOLO, pretrained model) | ✅ works, but mixes up some pieces (see below) |
| Robot-eye second opinion (both cameras agree before a piece's type changes) | ✅ added 2026-10-18 |
| King/queen height check | ✅ added 2026-10-18 |
| Pipeline window (game, board, cameras, Robo, STOP, torque) | ✅ works |
| Robo voice (Whisper + Piper + Qwen3.5-2B) | ✅ works, short replies; mic off by default |
| **Own YOLO model fine-tuned on this board** | ⏳ **not started**: no training data yet (`data/dataset/` is empty) |

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
   matched to the webcam's by colour and position.
3. **Combining them:**
   - Which squares are occupied, and by which colour, comes from the webcam (it sees every square).
   - The **type** of a known piece only changes when both cameras agree, or when the webcam alone is very sure and
     the robot eye can't see that square.
   - A known king is never lost.
   - A "pawn" that is far too tall, or a pawn on the first or last rank, is treated as a misreading.
4. **King vs queen:** the pieces are 64 mm vs 58 mm tall. When YOLO isn't sure (below 85 %), the height measured
   from the webcam image decides.

**Live sync** (on by default) keeps the game board matching the real board while no game is running. Red squares
on the board mean the cameras see something different there. Orange outlines mean an unsure reading.

**The robot eye calibrates itself.** At startup, the webcam's reading of the pieces is matched to the gripper
image. The grid is then lined up with the light and dark squares and snapped to the square corners. It is only
re-fitted when the saved fit stops matching. To redo it, use ⟳ calibrate on the Robot eye tile. Check the result
in `data/wrist_corners_check.jpg`.

**For better accuracy, fine-tune YOLO.** The model is a public pretrained chess model, not trained on these pieces,
and that is the main source of misreads.

1. Run `python -m chess_player.vision.capture_dataset --presets 1-21`. It shows each position to set up, and Enter
   takes 5 labelled shots.
2. Run `python -m chess_player.vision.train_yolo`. It writes `data/yolo/best.pt`, which the pipeline uses
   automatically.

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
