# Gesture 模块规范

目标：检测手部关键点，并向 Bridge 输出有序且符合契约的 `GesturePacket` 消息流。

## 交付契约

Gesture 模块必须严格按照 `src/contract_stub.md` 定义输出 `GesturePacket`。
Gesture 模块必须从 `src/contracts.py` 导入 `GesturePacket`。
Gesture 模块不得在本地重复定义 `GesturePacket` 数据类。

## 功能要求

- MVP 必须支持单手跟踪。
- 必须检测捏合意图，并输出 `pinch_state`：`open`、`pinch_candidate`、`pinched`、`release_candidate`。
- 必须输出 `tracking_state`：`tracked`、`temporarily_lost`、`not_detected`。
- 必须保证 `frame_id` 与 `timestamp_ms` 单调。
- 必须以 `camera_norm` 坐标空间输出坐标。

## 稳定性与信号质量

- 必须采用滞回（hysteresis）或等效机制抑制捏合抖动。
- 应对噪声坐标进行平滑，同时保持交互响应性。
- 应输出有意义的 `confidence`（0.0-1.0）。
- 应在连续跟踪期间保持 `hand_id` 稳定。

## 错误与生命周期

- 必须暴露生命周期状态：`INITIALIZING`、`RUNNING`、`DEGRADED`、`STOPPED`。
- 摄像头不可用时，必须进入明确的降级行为。
- 临时关键点检测失败时，必须避免崩溃。

## 集成约束

- 不得导入 Bridge 内部实现。
- 不得导入 Rendering 内部实现。
- 除非升级 `contract_version`，否则输出 schema 必须保持向后兼容。

## 非目标（Gesture）

- 多手仲裁。
- 超出单立方体抓取/拖拽所需范围的手势词汇。
