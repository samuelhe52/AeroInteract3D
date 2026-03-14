# Gesture 模块规范

目标：检测手部关键点，并向 Bridge 输出有序且符合契约的 `GesturePacket` 消息流。

实现流程说明见 [WORKFLOW.md](WORKFLOW.md)。专用实时预览 debug 入口位于 `src/gesture/debug/live_preview.py`。

## 交付契约

Gesture 模块必须严格按照 `src/contract.md` 与 `src/contracts.py` 定义输出 `GesturePacket`。
Gesture 模块必须从 `src/contracts.py` 导入 `GesturePacket`。
Gesture 模块不得在本地重复定义 `GesturePacket` 数据类。

## 实现归属

- Gesture 模块维护者必须实现继承自 `src/ports.py` 中 `GestureInputPort` 的具体服务类。
- 请以 `src/gesture/service.py` 中的 `GestureServiceImpl` 作为当前正式实现。

## 当前结构

- `src/gesture/service.py`：运行时 Gesture 服务
- `src/gesture/temporal.py`：共享时序归约器，负责状态稳定与坐标平滑
- `src/gesture/runtime.py`：检测器与采集运行时辅助
- `src/gesture/debug/live_preview.py`：实时预览配置入口
- `src/gesture/debug/live_preview_runtime.py`：预览窗口、叠加层与实际 FPS 显示

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
- 实时服务与 live preview 应共用同一套手势时序逻辑。

## 错误与生命周期

- 必须暴露生命周期状态：`INITIALIZING`、`RUNNING`、`DEGRADED`、`STOPPED`。
- 摄像头不可用时，必须进入明确的降级行为。
- 临时关键点检测失败时，必须避免崩溃。

## 集成约束

- 不得导入 Bridge 内部实现。
- 不得导入 Rendering 内部实现。
- 除非升级 `contract_version`，否则输出 schema 必须保持向后兼容。

## 当前运行默认值

- 目标 FPS 请求：`60`
- 请求采集分辨率：`1280x640`
- 预览窗口显示的是实际主循环测得的 FPS

## 非目标（Gesture）

- 多手仲裁。
- 超出单立方体抓取/拖拽所需范围的手势词汇。
