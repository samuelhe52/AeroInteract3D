# App startup defaults are duplicated here intentionally to avoid importing
# src.gesture package at module import time (prevents circular imports).
DEFAULT_CAMERA_INDEX = 0
DEFAULT_TARGET_FPS = 30
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 960

BRIDGE_HEARTBEAT_INTERVAL_FRAMES = 30
BRIDGE_MIN_TRACKING_CONFIDENCE = 0.6
# Increase X/Y hand-to-object sensitivity so the cube covers a larger visible area.
BRIDGE_POSITION_GAIN_XY = 1.35
# Keep depth gain neutral for now to avoid abrupt front/back jumps.
BRIDGE_POSITION_GAIN_Z = 1.0

MAX_ERROR_HISTORY = 10
RENDER_POSE_LOG_DEBOUNCE_MS = 500

__all__ = [
	"BRIDGE_HEARTBEAT_INTERVAL_FRAMES",
	"BRIDGE_MIN_TRACKING_CONFIDENCE",
	"BRIDGE_POSITION_GAIN_XY",
	"BRIDGE_POSITION_GAIN_Z",
	"DEFAULT_CAMERA_INDEX",
	"DEFAULT_FRAME_HEIGHT",
	"DEFAULT_FRAME_WIDTH",
	"DEFAULT_TARGET_FPS",
	"MAX_ERROR_HISTORY",
	"RENDER_POSE_LOG_DEBOUNCE_MS",
]