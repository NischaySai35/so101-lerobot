"""All tunable numbers in one place. Values marked MEASURE are placeholders until measured on the real setup.

Frames
- board frame (mm): origin at the outer corner of a1, x along files a->h, y along ranks 1->8, z up from the
  board surface. Square centers are ((file + 0.5) * SQUARE_MM, (rank + 0.5) * SQUARE_MM).
- robot frame (m): the URDF base_link frame. calib/board_to_robot.py measures the transform between the two.
"""
from pathlib import Path

PKG = Path(__file__).resolve().parent  # src/chess_player
ROOT = PKG.parents[1]  # ~/so101-lerobot
DATA = ROOT / "data"  # calibration results, poses, datasets, models (created on demand)

# ---------------------------------------------------------------- hardware
FOLLOWER_PORT = "/dev/so100_follower"
FOLLOWER_ID = "follower_arm"
CAMERA = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
# Fixed third-person camera facing the robot from the human side: the main vision source (sees the whole board).
# A USB webcam (/dev/... path, opened as MJPG 1280x720) or an MJPEG URL. Since 2026-10-17: the Ausha USB webcam.
EXT_CAMERA_URL = "/dev/v4l/by-id/usb-Ausha_Webcam_Ausha_Webcam-video-index0"
# old phone stream: "http://192.168.101.104:8080/stream"
EXT_CAMERA_SIZE = (1280, 720)
# A USB camera can only be opened by one program: while the pipeline runs it re-serves the raw frames here
EXT_SHARED_URL = "http://localhost:8770/stream/ext_raw"
WRIST_STREAM_URL = "http://localhost:8081/stream"  # wrist_stream.py (shares the gripper cam on the network)
EXT_WHITE_BALANCE = True  # make the webcam's colours match the robot camera before anything uses the frames
EXT_WB_TINT = (1.052, 1.002, 0.946)  # B, G, R of white in the robot camera (measured 2026-10-17); (1, 1, 1) = neutral
EXT_WB_WHITE_LEVEL = 235.0  # brightness the light squares / white pieces are brought to
WATCH_PERIOD_S = 0.3  # board watcher runs YOLO this often (~75 ms per frame on the RTX 5050)
WATCH_STABLE_N = 3  # readings that must agree before a change counts: ~1 s from the real board to the window
LIVE_SYNC_DEFAULT = True  # the window's board follows the real board (when no game is being played)
STOCKFISH = ROOT / "bin" / "stockfish"
URDF = PKG / "urdf" / "so101_new_calib.urdf"
EE_FRAME = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

# ---------------------------------------------------------------- calibration files
BOARD_TO_ROBOT = DATA / "board_to_robot.json"  # from calib/board_to_robot.py
INTRINSICS = DATA / "camera_intrinsics.npz"  # from calib/camera_intrinsics.py (optional)
MANUAL_CORNERS = DATA / "manual_corners.json"  # wrist cam: fallback when ArUco markers aren't visible
EXT_CORNERS = DATA / "ext_corners.json"  # third-person cam: clicked a1/h1/h8/a8 playing-area corners
LOOK_POSE = DATA / "look_pose.json"  # from tools/read_positions.py --save; falls back to data/poses/home_pose.json
HOME_POSE = DATA / "poses" / "home_pose.json"
YOLO_WEIGHTS = DATA / "yolo" / "best.pt"  # our own model (vision/train_yolo.py); preferred when present
# Until then: a public chess-piece YOLOv8m (KanisornPutta/chess-model-yolov8m on Hugging Face, scanned with picklescan),
# best of 7 tested on this board (2026-10-17). Confirmed camera readings become training data for our own model.
YOLO_PRETRAINED = DATA / "models" / "pretrained_chess" / "KanisornPutta_chess-model-yolov8m" / "chess-model-yolov8m.pt"
YOLO_IMGSZ = 960
# King vs queen: YOLO mixes them up; their heights differ (64 vs 58 mm). When YOLO's confidence for a K/Q is below
# KQ_TRUST_CONF, the piece's height measured from its box (corrected by the other pieces in the frame) decides.
KQ_HEIGHT_CHECK = True
KQ_TRUST_CONF = 0.85
# Live sync: a piece that stays on its square keeps its known type unless the cameras (stable over several frames)
# call it something else with at least this confidence. Both cameras agreeing raises the confidence (0.6 and 0.7
# -> 0.88), disagreeing drops it, so in practice: both cameras agree, or the webcam alone is very sure.
SYNC_TYPE_CONF = 0.8
RECHECK_TYPE_CONF = 0.5  # ⟳ Recheck: a fresh look, a bit more willing to change types
# Gripper camera as the second judge (vision/wrist_view.py), at the look pose
WRIST_CORNERS = DATA / "wrist_corners.json"  # auto-calibrated board corners in the gripper image
WRIST_GRIPPER_MASK = (0.19, 0.78, 0.60)  # fingers in the gripper image: x from, x to, y below (fractions)
WRIST_MIN_SCORE = 40.0  # light-dark contrast of the fitted grid; below = calibration doesn't fit (redo it)
WRIST_WATCH_PERIOD_S = 0.6
EYE_ADD_MIN_CONF = 0.55  # the robot eye adds a piece the webcam can't see (hidden behind others) at this confidence
LIVE_SYNC_HOLD_S = 1.2  # live sync applies a change only after the cameras have shown it this long (no flicker)

# ---------------------------------------------------------------- board (measured 2026-10-17)
# Cross-section from the playing area outwards: 8 squares | thin light line | wooden border.
SQUARE_MM = 33.3  # one square
BOARD_LINE_MM = 3.3  # light line around the playing area
BOARD_BORDER_MM = 23.4  # wooden border outside the line
PLAY_MM = 8 * SQUARE_MM  # 266.4 playing area (board frame origin = its a1 corner)
BOARD_OUTER_MM = PLAY_MM + 2 * (BOARD_LINE_MM + BOARD_BORDER_MM)  # 339.8 whole board
EDGE_MM = BOARD_LINE_MM + BOARD_BORDER_MM  # 26.7 from the playing area to the board's outer edge
BOARD_THICKNESS_MM = 0.0  # MEASURE: playing surface height above the table (graveyard pieces stand on the table)
# ArUco markers (DICT_4X4_50), ids 0..3 printed MARKER_MM wide, taped outside the playing area.
# MARKER_CENTERS gives each marker's center in board mm. Default: diagonally out from each corner.
ARUCO_DICT = "DICT_4X4_50"
MARKER_MM = 30.0  # MEASURE: printed black square edge
_off = 25.0  # MEASURE: distance from the board corner to the marker center along x and y
_B = 8 * SQUARE_MM
MARKER_CENTERS = {
    0: (-_off, -_off),  # beside a1
    1: (_B + _off, -_off),  # beside h1
    2: (_B + _off, _B + _off),  # beside h8
    3: (-_off, _B + _off),  # beside a8
}

# ---------------------------------------------------------------- pieces
# height mm, base diameter mm, gripper % when holding it (0 closed .. 100 open).
# Heights/diameters of k, n, r were estimated from the third-person camera (2026-10-17); p, b, q follow
# standard proportions. Refine with a ruler.
# Grip: gripper % reading when the jaws are closed on a piece. Two groups (user, 2026-10-17): king and queen share
# one width, all other pieces another. Taught with calib/teach_grip.py -> data/grip.json; defaults until then.
GRIP_FILE = DATA / "grip.json"
try:
    import json as _json

    _g = _json.loads(GRIP_FILE.read_text())
except (OSError, ValueError):
    _g = {}
GRIP_SMALL = float(_g.get("small", 10.0))  # pawn, knight, bishop, rook
GRIP_BIG = float(_g.get("big", 12.0))  # king, queen
_per = {k: float(v) for k, v in _g.get("per_piece", {}).items()}  # learned per piece (overrides small/big)
PIECES = {
    "p": dict(height=36.0, diameter=19.0, grip=_per.get("p", GRIP_SMALL)),
    "n": dict(height=48.0, diameter=21.0, grip=_per.get("n", GRIP_SMALL)),
    "b": dict(height=52.0, diameter=21.0, grip=_per.get("b", GRIP_SMALL)),
    "r": dict(height=42.0, diameter=21.0, grip=_per.get("r", GRIP_SMALL)),
    "q": dict(height=58.0, diameter=23.0, grip=_per.get("q", GRIP_BIG)),
    "k": dict(height=64.0, diameter=23.0, grip=_per.get("k", GRIP_BIG)),
}
# From the teleop demonstrations (data/demos, analyze_demo; medians of the 5 clean demos, 2026-10-17):
GRASP_HEIGHT_MM = 4.0  # jaw tip this far above the surface when grasping: pieces grabbed at the bottom (demos 0-11)
JAW_SHIFT_MM = 14.0  # the LEFT finger is fixed: the tip goes this far toward it before closing, so that finger lands
                     # beside the piece (demos: 10-17 mm, median 14)
HELD_OFFSET_MM = 4.0  # once gripped, the moving finger has pushed the piece against the fixed one: its centre is
                      # this far from the tip (demos: release tip 1-5 mm from the target centre, median 4)
# Pick-up correction measured on the real arm (2026-10-18), for the PICK only (placing unchanged): at the grasp the
# tip goes this much further toward the robot's base (along the board), further toward the fixed finger (the
# "go left" step), and lower (vertical; 0 = the demonstrated grasp height).
PICK_EXTRA_TOWARD_ARM_MM = 6.0
PICK_EXTRA_LEFT_MM = 4.0
PICK_EXTRA_DOWN_MM = 0.0
GRIPPER_OPEN = 28.4  # (22 before the 2026-10-18 gripper re-range, x1.2914) % open before descending (demos: 20-32 %, median 21; wider hits neighbours)
DROP_DETECT_PCT = 3.9  # while carrying: jaws closing this much past the holding reading = the piece slipped out
RECOVERY_ATTEMPTS = 2  # after a failed grab / drop: look with the webcam and retry this many times
RELEASE_OPEN_PCT = 10.3  # to let go, open only this much past the holding reading (keeps the moving finger off neighbours)
RELEASE_GAP_MM = 4.0  # how far the moving finger moves away from the piece when releasing (collision model)
GRIPPER_LOOK = 0.0  # % at the look/home pose: closed (user decision 2026-10-17)
# Grasp: command (grip - GRIP_SQUEEZE) and read where the jaws stop. `grip` above is the reading at contact.
GRIP_SQUEEZE = 5.2  # % past contact (LeRobot's gripper torque limit keeps the squeeze gentle)
GRIP_EMPTY_TOL = 1.9  # jaws stopped within this of the command = nothing in the hand
GRIP_TOO_WIDE = 12.0  # reading more than this above `grip` = misgrip (caught a neighbour, piece on its side)
GRASP_MAX_TILT = 55.0  # deg from vertical allowed at grasp/place height; far ranks need ~45-50 deg (teleop showed tilted grasps work)

# ---------------------------------------------------------------- z levels (mm above the board surface)
Z_SAFE = max(p["height"] for p in PIECES.values()) + 66.0  # 130 mm: reachable over every square (checked)  # max travel height (used when nothing better is known)
Z_PLACE_LIFT = 3.0  # release this far above the grasp height (demos: the tip was 9-20 mm up; 3 is gentler)
LIFT_MM = 10.0  # straight up after grasping, before the arc starts
# Carry shape copied from the demos: straight up, travel high with a gentle hump, straight down over the target.
ARC_PER_MM = 0.35  # lift = this x carry distance, clamped (demos: 95-120 mm above the grasp, median 116):
ARC_MIN_MM = 90.0  # even short moves lift clearly
ARC_MAX_MM = 110.0
ARC_RISE_FRACTION = 0.85  # vertical part of the lift; the rest is the hump in the middle of the travel
DETOUR_COST = 0.4  # mm of peak height one mm of extra path length is worth (bend around vs go over)
SIDE_STEP_CLEAR_MM = 12.0  # the sideways "go left" step happens this far above the piece's top
# Tight-spot grasp, learned from the crowded teleop demos (2026-10-18: neighbours on the fixed-finger side, the
# moving-finger side, both sides; analyze_demo). Used only when the normal grasp would hit a neighbour. You opened
# less, shifted less, did the sideways step above the neighbours and opened only ~2 % to let go.
TIGHT_JAW_SHIFT_MM = (8.0, 6.0)  # demos: 6-10 mm toward the fixed finger, median 8
TIGHT_GRIPPER_OPEN = 24.5  # % (demos 22.7-26.2; normal GRIPPER_OPEN 28.4)
TIGHT_RELEASE_OPEN_PCT = 4.0  # % past the holding reading to let go in a tight spot. Tried 1.5 % (demos released at
                              # +1-1.7 %, but with room around the target): on the robot the piece lifted out with the
                              # fingers twice (2026-10-18). If even 4 % would touch a neighbour, the move is refused
                              # (Robo asks for help) instead of carrying the piece away.
RELEASE_FREE_LOAD_PCT = 8.0  # gripper load above this, or jaws short of the command = still pressing on the piece
RELEASE_STEP_PCT = 2.0  # ...then open this much more (up to the grasp opening) until they don't
JAW_MM_PER_PCT = 1.78  # jaw opening per gripper % (demos: pawn 19 mm held at 15.3 %, queen 23 mm at 17.2 %, empty 4.5 %)
TRAVEL_SPEED_MM_S = 120.0
TRAVEL_CLEAR_MM = 15.0  # carried piece's bottom clears the tallest piece near the path by this (travel height per move)
CORRIDOR_MARGIN_MM = 4.0  # extra width of the carried piece's path when looking for pieces to clear
# Gripper footprint for the collision check (MEASURE roughly with a ruler; generous by default). Along the jaw
# axis u (toward the moving finger): the fixed finger reaches FINGER_LEFT_MM to the left of the tip; the moving
# finger's outer face sits at the piece width + FINGER_WIDTH_MM to the right when open around a piece.
FINGER_LEFT_MM = 7.0
FINGER_WIDTH_MM = 7.0
FINGER_HALF_DEPTH_MM = 6.0  # half the fingers' size across the jaw axis (~12 mm fingers)
FINGER_LENGTH_MM = 70.0  # fingers below the gripper body: tip is ~98 mm below the wrist roll axis, motor body ends ~25 mm below it
BODY_RADIUS_MM = 25.0
TOOL_RADIUS_MM = 18.0  # (legacy, unused)

# ---------------------------------------------------------------- off-board slots (board mm)
# Captured pieces go on the table just outside the board, one slot each, nearest the robot first (the robot sits
# behind rank 8 when it plays black). Spare queens for promotion sit in SPARES. Slots are (x, y) board mm;
# their z is -BOARD_THICKNESS_MM. Check them with calib/reachability_sweep.py.
_gx = EDGE_MM + 20.0  # slot centers 20 mm beyond the board's outer edge
# Slots run along both side edges; Slots() uses the ones nearest the robot first.
GRAVEYARD = [(_B + _gx, 20.0 + 32.0 * i) for i in range(8)] + [(-_gx, 20.0 + 32.0 * i) for i in range(8)]
SPARES = {"Q": [(_B + _gx + 32.0, 60.0)], "q": [(-_gx - 32.0, _B - 60.0)]}

# ---------------------------------------------------------------- motion
CONTROL_HZ = 30
MAX_JOINT_SPEED = 60.0  # deg/s cap; durations are stretched to respect it
MIN_MOVE_S = 0.6
DESCEND_S = 1.2  # vertical moves near pieces: slow
GRIP_S = 0.8
DWELL_S = 0.3
# Overall arm speed (1.0 = the speeds above). Scales every move: joint speed cap, carry speed, descents, gripper.
ARM_SPEED = 0.8
MAX_JOINT_SPEED *= ARM_SPEED
TRAVEL_SPEED_MM_S *= ARM_SPEED
MIN_MOVE_S /= ARM_SPEED
DESCEND_S /= ARM_SPEED
GRIP_S /= ARM_SPEED
# 2026-10-18 the follower was re-calibrated to match the leader in teleop (src/tools/match_follower_to_leader.py):
# wrist roll reads +0.879 deg for the same physical angle, gripper % x1.2914 for the same jaw position. The values
# here are converted so the arm moves physically exactly as before.
WRIST_ROLL_ZERO_SHIFT_DEG = 0.879  # LeRobot wrist_roll degrees - the URDF/IK model's (offset applied in motion/ik.py)
WRIST_ROLL_DEG = -86.9  # (-87.8 before the shift) # roll used for the corner touches (calibration) and grasps: jaws side by side, so a tilted
                        # gripper keeps both fingers level (at roll 0 the lower finger hit the far squares)
USE_SAG_MODEL = False  # gravity sag model (calib/sag_calibration.py) made far squares worse; use Run B's method:
                       # measure at the hover point (settle) and carry that correction down
SAG_ITERS = 3  # settle(): correction rounds after arriving at a hover / grasp point
SAG_WAIT_S = 0.25
SAG_DEADBAND_DEG = 0.3
SAG_MAX_DEG = 10.0  # never correct a joint by more than this
IK_MAX_ERR_MM = 4.0  # reject IK solutions worse than this
JOINT_MARGIN_DEG = 3.0  # stay this far inside the calibrated range

# ---------------------------------------------------------------- voice + vision-language model ("Robo")
VLM_ENABLED = True
VOICE_COMMENTARY = True  # Robo announcing game events (moves, "your move", retries...); the status bar shows them anyway
VOICE_QUIPS = False  # language-model comments after every move (chatty); answers to questions stay on
VOICE_ANSWER_SENTENCES = 1  # spoken answers are cut to this many sentences
VLM_MODEL = "Qwen/Qwen3.5-2B"  # fits the 8 GB GPU next to YOLO; "Qwen/Qwen3.5-4B" needs VLM_4BIT = True
VLM_4BIT = False
VLM_IMAGE_WIDTH = 448  # camera frames are shrunk to this width before the VLM sees them (speed)
WHISPER_MODEL = "openai/whisper-small"
VOICE_LANGUAGE = "en"
PIPER_VOICE = DATA / "models" / "piper" / "en_US-ryan-medium.onnx"
VOICE_VOLUME = 1.0  # starting values; the window's Robo panel changes them live
VOICE_SPEED = 1.0
VOICE_MIC_DEVICE = None  # arecord -D device; None = system default (PipeWire)
VOICE_WAKE_WORDS = ["robot", "robo", "roboto", "robbo", "robert"]  # Whisper sometimes hears "Robert"
VOICE_MIC_ON_AT_START = False  # mic starts switched off; turn it on with the Microphone switch in the window
VOICE_REQUIRE_WAKE = True  # spoken commands/questions must contain a wake word (typed ones never need it)
VOICE_SNR = 5.0  # speech = this many times the room's noise floor (5 = ignores more background talk)
VOICE_MIN_RMS = 600.0
VOICE_END_SILENCE_S = 0.8
VOICE_MIN_UTTERANCE_S = 0.5
VOICE_MAX_UTTERANCE_S = 12.0

# ---------------------------------------------------------------- game
ENGINE_TIME_S = 1.0
ROBOT_COLOR = "black"  # fallback only: once the webcam corners are clicked, the robot's colour is read from them
                       # (vision/ext_camera.robot_color_from_corners): rank 1 on the robot's side = robot white
