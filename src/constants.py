BRIDGE_HEARTBEAT_INTERVAL_FRAMES = 30
BRIDGE_MIN_TRACKING_CONFIDENCE = 0.6
# Increase X/Y hand-to-object sensitivity so the cube covers a larger visible area.
BRIDGE_POSITION_GAIN_XY = 1.35
# Keep depth gain neutral for now to avoid abrupt front/back jumps.
BRIDGE_POSITION_GAIN_Z = 1.0

MAX_ERROR_HISTORY = 10
RENDER_POSE_LOG_DEBOUNCE_MS = 500

# 交互模式枚举
BRIDGE_MODE_NORMAL = "normal"
BRIDGE_MODE_ROTATING = "rotating"

# 交互状态枚举
BRIDGE_STATE_IDLE = "idle"
BRIDGE_STATE_PENDING_GRAB = "pending_grab"
BRIDGE_STATE_GRABBING = "grabbing"

# 物体交互状态（对应Rendering材质）
INTERACTION_IDLE = "idle"
INTERACTION_PENDING_GRAB = "pending_grab"
INTERACTION_GRABBED = "grabbed"

# 距离阈值（世界坐标系下，适配物体setScale(0.2)的大小）
HOVER_DISTANCE_THRESHOLD = 0.15  # 小于此距离进入hover
GRAB_RELEASE_DISTANCE_THRESHOLD = 0.2  # 大于此距离强制释放

# 交互物体ID
PRIMARY_OBJECT_ID = "primary_cube"
