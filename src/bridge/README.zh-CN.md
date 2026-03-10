# Bridge 模块规范

目标：在保持交互行为确定性的前提下，将上游 `GesturePacket` 流转换并同步为下游 `SceneCommand` 流。

## 范围

- Bridge 是手势语义与渲染语义之间唯一的转换层。
- Bridge 不得依赖具体渲染后端细节。
- Bridge 不得要求 Gesture 模块内部模型细节。

## 进程内接口（MVP）

- `GestureInputPort`
  - `start() -> None`
  - `poll() -> GesturePacket | None`
  - `health() -> dict`
  - `stop() -> None`
- `RenderOutputPort`
  - `start() -> None`
  - `push(command: SceneCommand) -> None`
  - `health() -> dict`
  - `stop() -> None`
- `BridgeService`
  - `start() -> None`
  - `process(packet: GesturePacket) -> list[SceneCommand]`
  - `health() -> dict`
  - `stop() -> None`

Bridge 必须通过上述抽象端口（或等价接口）集成，不得直接导入队友模块内部实现。

Bridge 必须从 `src/contracts.py` 导入 `GesturePacket` 与 `SceneCommand`。
Bridge 不得在本地重复定义这两个契约数据类。

## 实现归属

- Bridge 模块维护者必须实现继承自 `src/ports.py` 中 `BridgeService` 的具体服务类。
- 请以 `src/bridge/service_stub.py` 中的 `BridgeServiceStub` 作为起始骨架，逐步替换 no-op 逻辑。
- `main.py` 当前已接入该 stub，便于在完整实现前继续集成联调。
- 当前协作说明：Bridge 仍然负责在发出 `world_norm` 位姿命令前调用相机到世界坐标的转换钩子，但该转换逻辑的具体实现由渲染模块维护者负责。

## 核心职责

- 校验输入 `GesturePacket` 是否符合共享契约。
- 维护交互状态机。
- 将坐标从 `camera_norm` 映射到 `world_norm`。
- 输出有序 `SceneCommand` 消息。
- 安全处理异常包（重复、过期帧、跟踪丢失）。

## 交互状态机

必需状态：

- `idle`
- `pinch_candidate`
- `grabbing`
- `release_candidate`

状态迁移只能由 `GesturePacket` 中的 `pinch_state`、`tracking_state` 和置信度门限驱动。

必需行为：

- `idle -> pinch_candidate`：出现捏合意图。
- `pinch_candidate -> grabbing`：满足稳定条件后进入抓取。
- `grabbing -> release_candidate`：出现释放意图。
- `release_candidate -> idle`：释放确认后回到空闲。
- 任意状态 -> `idle`：持续 `not_detected` 或显式重置信号时。

## 命令发射规则

- 启动或重新初始化时，必须发射 `init_scene`。
- 抓取/释放边界上，必须发射 `set_object_state`。
- 仅在对象位移有效时，才发射 `set_object_pose`。
- 当上游突发速率高于下游消费速率时，应合并位姿更新。
- 抓取中若跟踪丢失，必须发射 `reset_interaction`。

## 错误与生命周期要求

- Bridge 生命周期状态：`INITIALIZING`、`RUNNING`、`DEGRADED`、`STOPPED`。
- 必须返回结构化错误（错误码、信息、可恢复性提示）。
- 在可行时，`DEGRADED` 状态下必须尽力继续运行。
- 启停行为必须支持幂等。

## 非目标（Bridge）

- 手势模型调优。
- 渲染后端性能优化。
- 多手仲裁。
