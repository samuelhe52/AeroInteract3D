# Rendering 模块规范

目标：消费 Bridge 输出的 `SceneCommand` 流，并以确定性方式渲染场景与对象交互反馈。

当前里程碑渲染技术栈已固定为：**Python + Panda3D**。

无论具体实现细节如何，契约行为必须固定。

## 技术基线（Python + Panda3D）

- 必须使用 Python 实现渲染逻辑。
- 必须使用 Panda3D 作为渲染引擎/运行时。
- 应优先使用 Panda3D 的场景图、相机与任务循环能力，而非直接调用底层图形 API。
- 应将窗口/上下文管理封装在适配层中，保证命令消费逻辑可测试。
- 应将渲染循环时序与命令消费解耦（每帧消费最近一次有效命令状态）。
- GPU/资源初始化失败时，应以结构化启动错误显式返回。

## 交付契约

Rendering 模块必须严格按照 `src/contract_stub.md` 定义消费 `SceneCommand`。
Rendering 模块必须从 `src/contracts.py` 导入 `SceneCommand`。
Rendering 模块不得在本地重复定义 `SceneCommand` 数据类。

## 实现归属

- Rendering 模块维护者必须实现继承自 `src/ports.py` 中 `RenderOutputPort` 的具体服务类。
- 请以 `src/rendering/service_stub.py` 中的 `RenderOutputServiceStub` 作为起始骨架，逐步替换 no-op 逻辑。
- `main.py` 当前已接入该 stub，便于在完整实现前继续集成联调。
- 当前协作说明：渲染模块维护者负责实现供 Bridge 调用的相机到世界坐标转换逻辑，然后由 Bridge 在发出 `world_norm` 位姿更新前调用该逻辑。

## 功能要求

- 必须先处理 `init_scene`，再处理对象更新命令。
- 必须将 `set_object_pose` 应用于 `object_id` 指定的目标对象。
- 必须将 `set_object_state` 映射为交互视觉状态（`idle`、`hover`、`grabbed`）。
- 必须接收并安全忽略重复的 `command_id`。
- 必须忽略或安全处理过期/乱序帧命令。
- 必须在 `reset_interaction` 时将对象交互状态恢复到安全默认值。

## 契约与数据处理

- 必须将 `world_norm` 作为 Bridge 输入位姿的标准空间。
- 不得重解释手势侧语义（捏合逻辑归 Bridge 负责）。
- 应容忍未知可选字段，以保持前向兼容。
- 对格式错误命令应输出轻量运行时告警，且避免崩溃。

## 错误与生命周期

- 必须暴露生命周期状态：`INITIALIZING`、`RUNNING`、`DEGRADED`、`STOPPED`。
- 命令短时中断时，必须继续渲染最近一次有效场景状态。
- 场景初始化失败时，必须显式返回结构化错误。

## 集成约束

- 不得导入 Gesture 内部实现。
- 不得依赖 Bridge 私有实现细节。
- 必须通过命令消费边界（队列、回调或等价接口）接入。

## 非目标（Rendering）

- 手势识别逻辑。
- 契约翻译逻辑。
- 超出 MVP 交互需求的多对象物理仿真。
