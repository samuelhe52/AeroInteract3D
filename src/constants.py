BRIDGE_HEARTBEAT_INTERVAL_FRAMES = 30
BRIDGE_MIN_TRACKING_CONFIDENCE = 0.6
# Increase X/Y hand-to-object sensitivity so the cube covers a larger visible area.
BRIDGE_POSITION_GAIN_XY = 1.35
# Keep depth gain neutral for now to avoid abrupt front/back jumps.
BRIDGE_POSITION_GAIN_Z = 1.0

MAX_ERROR_HISTORY = 10
RENDER_POSE_LOG_DEBOUNCE_MS = 500