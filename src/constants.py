DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 30
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 960
# Zero means run without an automatic frame-count stop condition.
DEFAULT_MAX_FRAMES = 0
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5
DEFAULT_MODEL_COMPLEXITY = 1
GESTURE_DEFAULT_HAND_ID = "hand-1"
GESTURE_MODEL_RELATIVE_PATH = "models/hand_landmarker.task"
GESTURE_DETECT_MAX_SIDE = 640
GESTURE_SMOOTHING_PRESET = "medium"

# Emit one summary log for every N processed gesture packets.
GESTURE_FRAME_SUMMARY_INTERVAL = 30
BRIDGE_HEARTBEAT_INTERVAL_FRAMES = 30
BRIDGE_MIN_TRACKING_CONFIDENCE = 0.6

MAX_ERROR_HISTORY = 10
RENDER_POSE_LOG_DEBOUNCE_MS = 500
DEBUG_FPS_SAMPLE_WINDOW = 10

# Keep tracking in a temporary-loss state for this many consecutive missing frames.
TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES = 6
TEMPORAL_PINCH_ENTER_THRESHOLD = 0.10
TEMPORAL_PINCH_HOLD_THRESHOLD = 0.08
TEMPORAL_PINCH_RELEASE_THRESHOLD = 0.18
# Require this many consecutive frames before promoting a pinch candidate to confirmed pinched.
TEMPORAL_PINCH_CONFIRM_FRAMES = 4
# Require this many consecutive frames before promoting a release candidate to confirmed open.
TEMPORAL_RELEASE_CONFIRM_FRAMES = 4
# Z-axis EMA gain. Higher values follow depth changes more directly with less lag.
TEMPORAL_SMOOTHING_ALPHA = 0.88
# X/Y EMA gain. Higher values reduce lateral/vertical smoothing and increase responsiveness.
TEMPORAL_XY_SMOOTHING_ALPHA = 0.82
# Ignore motions smaller than this camera_norm delta to suppress sub-pixel landmark jitter.
TEMPORAL_POSITION_DEADZONE = 0.001
# Weight applied to short-term extrapolation when tracking is temporarily lost.
TEMPORAL_PREDICTION_BLEND = 0.25
# How far to project the last measured velocity forward during temporary tracking loss.
TEMPORAL_PREDICTION_LEAD = 0.35
# Per-frame decay applied to predicted velocity while the hand is not currently detected.
TEMPORAL_LOST_TRACKING_MOTION_DAMPING = 0.55

# Hand-scale heuristic depth estimation tuning.
DEPTH_ESTIMATION_NEAR_HAND_SCALE = 0.55
DEPTH_ESTIMATION_FAR_HAND_SCALE = 0.18
DEPTH_ESTIMATION_LOCAL_Z_WEIGHT = 0.35