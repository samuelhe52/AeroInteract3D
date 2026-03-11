# 共享契约规范（Bridge 优先）

本文件是模块集成契约的唯一事实来源。

所有模块都必须实现该契约。任何不兼容变更都必须升级 `contract_version`，并由各模块负责人共同评审。

Python 数据类（dataclass）的唯一定义位于 `src/contracts.py`。

- 各模块必须从 `src/contracts.py` 导入共享数据类。
- 各模块不得在本地重复定义 `GesturePacket` 或 `SceneCommand`。

## 1）契约版本

- `contract_version`：字符串，语义化版本格式（`MAJOR.MINOR.PATCH`）。
- 当前基线：`0.1.0`。
- 兼容性规则：
  - `PATCH`：仅文档澄清，不改 schema。
  - `MINOR`：向后兼容的字段新增（仅新增可选字段）。
  - `MAJOR`：向后不兼容的 schema 或语义变更。

## 2）坐标系与单位

所有包含坐标的数据都必须带有明确的坐标系元数据。

- 世界单位：MVP 阶段采用 `[-1.0, 1.0]` 归一化浮点。
- 轴约定（默认）：
  - `+x`：向右
  - `+y`：向上
  - `+z`：朝向用户/摄像头
- `coordinate_space` 允许值：
  - `camera_norm`：摄像头归一化坐标。
  - `world_norm`：Bridge 映射后的世界归一化坐标。

## 3）`GesturePacket`（Gesture -> Bridge）

必须以有序流的形式输出。

### 必填字段（`GesturePacket`）

- `contract_version: str`
- `frame_id: int`（单调递增，可从 0 或 1 开始）
- `timestamp_ms: int`（生产者时钟的单调时间戳）
- `hand_id: str`（同一只手持续跟踪期间应保持稳定）
- `tracking_state: str`，取值 `{ "tracked", "temporarily_lost", "not_detected" }`
- `confidence: float`，范围 `[0.0, 1.0]`
- `pinch_state: str`，取值 `{ "open", "pinch_candidate", "pinched", "release_candidate" }`
- `index_tip: {"x": float, "y": float, "z": float}`
- `thumb_tip: {"x": float, "y": float, "z": float}`
- `palm_center: {"x": float, "y": float, "z": float}`
- `coordinate_space: str`（Gesture 模块通常输出 `camera_norm`）

### 可选字段（推荐）

- `pinch_distance: float`
- `velocity: {"x": float, "y": float, "z": float}`
- `smoothing_hint: {"method": str, "window": int}`
- `debug: dict`

## 4）`SceneCommand`（Bridge -> Rendering）

必须以有序流的形式消费，并尽可能按幂等方式应用。

当前项目决策：渲染消费端采用 **Python + Panda3D** 实现。该决策不改变 `SceneCommand` 的 schema 要求。

### 必填字段（`SceneCommand`）

- `contract_version: str`
- `command_id: str`（唯一）
- `frame_id: int`（可用时与上游对齐）
- `timestamp_ms: int`
- `command_type: str`，取值
  - `{ "init_scene", "set_object_pose", "set_object_state", "heartbeat", "reset_interaction" }`
- `object_id: str`（对象级命令时必需）
- `payload: dict`（结构由 `command_type` 决定）

### 按 `command_type` 的 `payload` 要求

- `init_scene`
  - 必须包含目标对象标识与默认变换。
- `set_object_pose`
  - 必须包含 `position: {x,y,z}`。
  - 可选包含 `rotation: {x,y,z,w}` 与 `scale: {x,y,z}`。
  - 必须包含 `coordinate_space`（`world_norm`）。
- `set_object_state`
  - 必须包含 `interaction_state`，取值 `{ "idle", "hover", "grabbed" }`。
- `heartbeat`
  - 必须包含最小存活信息。
- `reset_interaction`
  - 必须包含原因，取值 `{ "tracking_lost", "manual", "reinitialize" }`。

## 5）顺序与传递语义

- 生产者必须输出非递减的 `frame_id` 和单调 `timestamp_ms`。
- 消费者必须容忍重复消息，并忽略已应用过的 `command_id`。
- 当 `frame_id` 回退时，消费者应忽略过期消息。
- 只有 Bridge 可以做手势语义到渲染语义的转换。

## 6）生命周期与健康信号

各模块应暴露生命周期状态：

- `INITIALIZING`
- `RUNNING`
- `DEGRADED`
- `STOPPED`

模块必须以结构化错误信息显式失败，禁止静默失败。

## 7）MVP 非目标

- 暂不定义多手交互契约。
- 暂不标准化持久化/回放格式。
- MVP 阶段不要求网络传输（仅进程内集成）。
