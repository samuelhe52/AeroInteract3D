# GesturePacket 双手化重构提案

本文档描述当前 `GesturePacket` 从单手结构演进为双手结构的建议方案。它是设计提案，不是当前已生效的共享契约。

当前事实仍以 `src/contracts.py`、`src/contract.zh-CN.md` 和 `src/contract.md` 为准；这些文件目前定义的仍是单手 `GesturePacket`，`contract_version` 基线仍为 `2.0.0`。

## 背景

当前 `GesturePacket` 定义如下：

- 帧级字段：
  - `contract_version`
  - `frame_id`
  - `timestamp_ms`
  - `coordinate_space`
  - `camera_frame`
- 单手字段：
  - `hand_id`
  - `tracking_state`
  - `confidence`
  - `pinch_state`
  - `index_tip`
  - `thumb_tip`
  - `wrist`
  - `pinch_distance`
  - `velocity`
  - `smoothing_hint`
  - `debug`

这套结构在单手场景下没有问题，但它把“整帧信息”和“单手信息”混在了同一个对象里。当前实现里，`src/gesture/temporal.py` 负责生成这些字段，`src/bridge/service.py`、`src/gesture/debug/live_preview_runtime.py`、`src/rendering/debug/data_panel.py` 和多组测试都直接读取这些顶层单手字段。

一旦需要同时支持两只手，现有结构会立刻遇到几个问题：

- `GesturePacket` 顶层字段默认只对应一只手，无法自然表达第二只手。
- Bridge、debug 视图和测试代码都把这些字段当作唯一输入，扩展时会产生大量条件分支。
- `debug` 和 `smoothing_hint` 本质上也是单手时序结果，继续留在帧级别会导致语义变得混乱。

## 重构目标

目标不是简单地“再加一组 second hand 字段”，而是把单手状态抽象成独立对象，再让 `GesturePacket` 退回到“整帧容器”角色。

建议重构后满足以下原则：

- `GesturePacket` 只保存整帧共享信息和双手槽位。
- 每只手的追踪状态由独立数据类描述，字段边界清晰。
- `primary_hand` 与 `secondary_hand` 使用统一结构，避免双份逻辑。
- 单手消费者在迁移期间可以只读取 `primary_hand`，降低改造成本。

## 建议结构

建议新增单手状态对象，例如 `HandState`：

```python
@dataclass(slots=True)
class HandState:
    hand_id: str
    handedness: Literal["left", "right", "unknown"]
    tracking_state: TrackingState
    confidence: float
    pinch_state: PinchState
    index_tip: Vec3
    thumb_tip: Vec3
    wrist: Vec3
    pinch_distance: float | None = None
    velocity: Vec3 | None = None
    smoothing_hint: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None
```

然后将 `GesturePacket` 重构为整帧容器：

```python
@dataclass(slots=True)
class GesturePacket:
    contract_version: str
    frame_id: int
    timestamp_ms: int
    coordinate_space: CoordinateSpace
    primary_hand: HandState | None
    secondary_hand: HandState | None = None
    camera_frame: CameraFrame | None = None
    frame_debug: dict[str, Any] | None = None
```

## 基于当前实现的字段归属调整

结合现有代码，字段建议按下面的边界拆分：

### 保留在 `GesturePacket` 的字段

- `contract_version`
  - 契约版本描述的是整条消息，不是某一只手。
- `frame_id`
  - 表示整帧时序，不应复制到手级对象之外再做二次同步。
- `timestamp_ms`
  - 与 `frame_id` 同理，属于帧级元数据。
- `coordinate_space`
  - 当前实现中两只手都处于同一坐标空间，继续保留在帧级最简洁。
- `camera_frame`
  - 是整帧相机数据，不属于某一只手。
- `frame_debug`
  - 这是为双手仲裁、跨手关系或全局调试预留的帧级扩展位。当前如果没有需要，可以先不落地。

### 下沉到 `HandState` 的字段

- `hand_id`
- `tracking_state`
- `confidence`
- `pinch_state`
- `index_tip`
- `thumb_tip`
- `wrist`
- `pinch_distance`
- `velocity`
- `smoothing_hint`
- `debug`

这些字段在当前实现里都由 `src/gesture/temporal.py` 围绕单手时序归约生成，天然属于单手状态。

### 建议新增到 `HandState` 的字段

- `handedness`
  - 当前实现里 handedness 只存在于 `debug` 中。
  - 双手支持后，`left/right/unknown` 应提升为显式字段，用于槽位分配、调试显示和后续 Bridge 策略判断。

## `primary_hand` / `secondary_hand` 的语义

这里不建议把 `primary_hand` 直接等同于“左手”或“右手”，而应定义成当前帧的主槽位和副槽位：

- `primary_hand`
  - 当前交互主通道对应的手。
  - 在仅支持单手消费的阶段，Bridge 和 debug 视图默认只读取它。
- `secondary_hand`
  - 同帧内的第二只手。
  - 当只有一只手时，置为 `None`。

这样做有两个好处：

- 兼容当前大量“单通道”消费者，迁移时可以先只接 `primary_hand`。
- 后续如果需要按交互优先级、置信度或连续性做主副手切换，协议层不需要再次改名。

如果未来业务明确要求固定语义，也可以在 `HandState.handedness` 上表达左右手，在 `primary_hand` / `secondary_hand` 上表达处理优先级，两者不要混用。

## 为什么这是一次 MAJOR schema 变更

当前共享契约版本规则定义如下：

- `PATCH`：文档澄清
- `MINOR`：只允许新增可选字段
- `MAJOR`：不兼容 schema 或语义变化

这次重构会把现有顶层字段迁移到子对象中，属于明确的非兼容变更，因此不能继续沿用 `2.0.0`。如果按最终目标完全切换到双手结构，建议将共享契约升级到 `3.0.0`。

此外，`src/contract.zh-CN.md` 和 `src/contract.md` 目前都把“多手交互契约”列为 MVP 非目标。真正落地时，这两份文档也必须同步更新。

## 迁移影响面

按当前代码，至少会影响以下位置：

- `src/contracts.py`
  - 新增 `HandState`，重构 `GesturePacket`。
- `src/utils/contracts.py`
  - 校验逻辑需要从“验证一组顶层单手字段”改为“验证 0 到 2 个 `HandState`”。
- `src/gesture/temporal.py`
  - 当前 `TemporalGestureReducer` 直接返回单手 `GesturePacket`，需要调整为构造 `HandState` 并装入 `GesturePacket`。
- `src/bridge/service.py`
  - 当前大量逻辑直接访问 `packet.index_tip`、`packet.thumb_tip`、`packet.wrist`、`packet.pinch_state`、`packet.tracking_state`。
  - 在过渡阶段，可先统一改为读取 `packet.primary_hand`。
- `src/gesture/debug/live_preview_runtime.py`
  - 当前预览面板默认展示唯一一只手，后续至少要定义“显示主手”还是“并排显示双手”。
- `src/rendering/debug/data_panel.py`
  - 当前调试面板也是单手视图，需要明确主副手展示策略。
- `tests/test_bridge_service.py`
- `tests/test_gesture_service.py`
- `tests/test_gesture_temporal.py`
- `tests/test_rendering_service.py`
  - 构造数据与断言方式都需要同步改造。

## 建议落地顺序

为了降低风险，建议分两步做，而不是一次性硬切：

### 第一步：内部结构先解耦，外部接口暂时兼容

- 在 `src/contracts.py` 中先引入 `HandState`。
- `src/gesture/temporal.py` 先改为内部构造 `HandState`。
- `GesturePacket` 暂时同时保留：
  - 新字段：`primary_hand`、`secondary_hand`
  - 旧字段：`hand_id`、`tracking_state`、`pinch_state`、`index_tip`、`thumb_tip`、`wrist` 等
- 旧字段直接镜像 `primary_hand`，作为兼容层。

这一阶段的目标是先把“数据组织方式”从单手顶层改成双手容器，但尽量不打断 Bridge、debug 和测试。

### 第二步：正式切换共享契约

- 更新 `src/contract.zh-CN.md`、`src/contract.md` 和 `src/utils/contracts.py`。
- 删除顶层单手兼容字段，只保留 `primary_hand` / `secondary_hand`。
- Bridge、debug、测试全部改为从 `HandState` 读取。
- 将 `EXPECTED_CONTRACT_VERSION` 从 `2.0.0` 升级到新的 MAJOR 版本。

这一阶段完成后，`GesturePacket` 才真正成为双手帧容器。

## 结论

基于当前实现，`GesturePacket` 的双手化重构方向应当是：

1. 把所有单手状态字段收拢到 `HandState`。
2. 让 `GesturePacket` 只保留帧级元数据和双手槽位。
3. 将当前藏在 `debug` 中的 `handedness` 提升为显式字段。
4. 通过“兼容层过渡 + MAJOR 契约升级”完成迁移，而不是直接在现有顶层字段上继续堆补丁。

这样调整之后，数据边界会更清晰，Bridge 和调试链路也更容易在双手场景下保持统一和可维护。
