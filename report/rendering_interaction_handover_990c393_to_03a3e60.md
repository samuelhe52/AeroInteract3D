# Rendering / 交互模块交接说明

本文档总结 `990c393` 到当前 `HEAD(03a3e60)` 之间，围绕 rendering / 交互链路发生的主要改动。目标读者是原 rendering/交互模块负责人，因此重点不是逐行 diff，而是帮助快速恢复以下上下文：

- 这段时间模块职责发生了什么变化
- 当前交互链路实际如何工作
- 哪些行为是新约定，哪些只是实现细节
- 接手后优先应该看哪些文件和测试

## 1. 一句话结论

这段时间最大的变化不是“把某个效果调好了”，而是把交互语义进一步前移到了 Bridge：

- Bridge 现在负责 scene 初始化、多物体 hover/grab/rotation 状态机、相对位移、桌面约束、以及虚拟手所需关键点的生成。
- Rendering 现在更像一个严格消费 `SceneCommand` 的执行器：负责解析 `init_scene / set_object_pose / set_object_state / set_hand_pose`，把 world_norm 坐标映射到 Panda3D scene，并渲染对象与双指虚拟手。
- 虚拟手不再直接依赖 gesture/raw landmarks，而是完全由 Bridge 输出的 `set_hand_pose` 驱动。

如果只记一件事：当前架构里，交互“语义”基本都在 `src/bridge/service.py`，Rendering 主要负责“表现”和“合同执行”。

## 2. 提交范围概览

从 `990c393` 到 `03a3e60` 主要提交如下：

1. `40e546f` Implement bridge-driven grab cues and hand overlay
2. `82b4a82` Refine relative object interaction and rotation controls
3. `2b6f13d` Remove camera view configuration knob
4. `df37c7e` Improve demo scene layout and multi-object interaction
5. `4d1f702` Fix rotation target locking and mirrored heading
6. `6b8835e` Add stylized two-finger hand visual
7. `82979af` Improve two-finger hand structure
8. `03a3e60` Add table blocking to grabbed objects

建议理解顺序不是按时间，而是按职责：

1. `src/bridge/service.py`
2. `src/rendering/service.py`
3. `src/rendering/interaction/virtual_hand.py`
4. `src/contract.zh-CN.md`
5. `tests/test_bridge_service.py`
6. `tests/test_rendering_service.py`

## 3. 当前架构状态

### 3.1 数据流

当前主链路可以理解为：

`GesturePacket -> BridgeServiceImpl.process() -> SceneCommand stream -> RenderingServiceImpl.push() -> Panda3D scene`

其中关键变化是：

- Bridge 每帧都会先发 `set_hand_pose`，tracked 且 confidence 足够时 `visible=true`，否则 `visible=false`。
- 首个有效包会发 `init_scene`，scene 由 Bridge 描述，而不是 Rendering 内置一个默认 cube。
- 交互对象状态通过 `set_object_state` 表达，pose 通过 `set_object_pose` 表达，二者现在支持多物体。

### 3.2 职责边界

当前职责分布已经比较清楚：

- Gesture：继续负责手势观测、rotation debug 数据等原始输入。
- Bridge：负责 camera_norm -> world_norm、hover/grab/rotation 状态机、对象选择、对象约束、hand pose 生成。
- Rendering：负责 SceneCommand 校验与消费、scene object 创建、材质更新、pose 应用、虚拟手绘制。

和之前相比，Rendering 明显去掉了“自己从 observation/raw landmarks 推导手”的逻辑。

## 4. Bridge 侧的主要变化

关键文件：`src/bridge/service.py`

### 4.1 从单物体状态机变成多物体状态机

之前交互逻辑基本围绕单个 `primary_cube`。现在 Bridge 引入了：

- `ObjectInteractionState`
- `_object_states`
- `_hovered_object_id`
- `_grabbed_object_id`
- `_rotation_object_id`

这意味着 Bridge 内部已经维护对象级交互状态，而不是全局只有一个 object。

`_make_init_scene()` 现在会根据 `TABLE_SCENE_OBJECTS` 初始化一组对象，并把 interactable 对象同步进 `_object_states`。

### 4.2 scene 初始化从“默认 cube”变成“Bridge 描述 scene”

`TABLE_SCENE_OBJECTS` 是这段时间架构变化的一个核心锚点。

它现在定义了：

- `table_plane`
- `primary_cube`
- `tile_left`
- `pillar_left`
- `cube_right`
- `tile_right`

每个对象包含：

- `object_id`
- `init_pos`
- `init_hpr`
- `shape`
- `scale`
- `color`
- `interactable`
- 对 interactable 对象额外包含 `interaction_radius`

这说明当前 demo scene 的“布局真相”已经从 Rendering 转移到 Bridge。

### 4.3 hover 语义从 `hover` 改成 `pending_grab`

之前 hover 更像视觉高亮状态。现在合同和实现统一改成了：

- `idle`
- `pending_grab`
- `grabbed`
- `rotating`

原因是现在 hover 不只是“悬停”，而是“可抓取候选对象”。这个命名和状态机语义更贴近交互行为。

对应逻辑主要在：

- `_select_hovered_object()`
- `_sync_hover_state()`
- `_set_object_interaction_state()`
- `_render_state()`

### 4.4 grab 改成相对位移，而不是绝对吸附

这是这轮交互手感改动里最重要的一点。

抓取开始时：

- Bridge 不再把物体中心直接吸到 pinch midpoint
- 而是记录 `grab_offset_world = object_pos - hand_anchor_world`

拖动过程中：

- 通过 `_drag_world_position()` 计算 `hand_anchor_world + grab_offset_world`

结果是：

- 物体保持抓取瞬间的相对偏移
- 不会出现一 pinch 就突然跳到手中心的现象

对应逻辑：

- `_interaction_anchor()`
- `_drag_world_position()`
- `_make_object_pose()`
- `_pose_payload()`

### 4.5 rotation 模式被正规化

rotation 现在不是一个临时分支，而是完整状态机的一部分。

Bridge 侧新增/强化的约定：

- rotation 触发来自 `packet.debug.rotation.mode_active`
- rotation 时优先锁定一个 target object，而不是每帧重新找目标
- rotation 模式下发出的 `set_object_pose` 可以只有 `hpr`，没有 `position`
- rotation 起始参考值不再直接等于当前手部角度，而是：
  - `rotation_reference_hpr = 当前物体 world_hpr`
  - `rotation_reference_input = 当前 rotation 输入`

这样后续角度更新是“相对增量”，不会在进入 rotation 的第一帧瞬间跳变。

关键方法：

- `_rotation_mode_active()`
- `_rotation_target_object()`
- `_handle_rotation_mode()`
- `_rotation_input_hpr()`
- `_rotation_hpr_payload()`

后续两个修正点也在这里：

- `4d1f702` 修了 rotation target locking
- `4d1f702` 也修了 mirrored input 下 heading delta 需要反向的问题

### 4.6 桌面约束与强制释放

`03a3e60` 新增了比较实用的一层物理约束：grabbed object 不能穿过桌面。

现在流程是：

- `TABLE_SURFACE_Y` 由 `table_plane` 位置和厚度推导
- `_constrain_object_to_table()` 保证 object center 的最低 y 是 `table_surface + half_height`
- 如果用户手继续往桌下拖，且 hand anchor 到 constrained object 的距离超过 `GRAB_RELEASE_DISTANCE_THRESHOLD`，则 Bridge 直接释放抓取

这个逻辑解决了两个问题：

- 视觉上物体不会穿过桌面
- 用户继续下压时不会一直处在“卡住但仍算 grabbed”的模糊状态

## 5. Rendering 侧的主要变化

关键文件：`src/rendering/service.py`

### 5.1 Rendering 更像通用 SceneCommand consumer 了

这段时间 Rendering 最大的变化是：不再假设 scene 里只有默认 cube，也不再自己推导 hand。

新增了：

- `SceneObjectDescriptor`
- `_parse_scene_object_descriptor()`
- `_create_scene_object()`
- `_handle_set_hand_pose()`
- `hand_pose_updates` metrics

现在 `init_scene` 支持的对象描述比以前完整很多，包含：

- 形状 `shape`
- 尺寸 `scale`
- 颜色 `color`
- 是否可交互 `interactable`

### 5.2 scene object 创建支持 shape / scale / color / interactable

`_handle_init_scene()` 现在不再只做：

- attach box
- setPos
- setHpr
- setScale(0.2)

而是会先解析对象描述，再创建对象节点，并记录：

- `shape` tag
- `interactable` tag

另外修了 Panda3D `box` 模型的中心问题：

- 通过 `_box_model_center_offset() = (-0.5, -0.5, -0.5)`
- 把 visual model 居中挂在 transform node 下

这个改动对 rotation 很关键，否则转动中心会偏。

### 5.3 `set_object_pose` 现在允许“只改 position”或“只改 hpr”

这是为了匹配 Bridge 的 rotation 逻辑。

之前 `set_object_pose` 实际上默认把 position/hpr 都当成会出现的字段处理。现在改成：

- `position` 可选
- `hpr` 可选
- 至少一个存在
- 如果只有 `hpr`，则保留当前 position
- 如果只有 `position`，则保留当前 hpr

这使得 rotation-only 更新不会意外重置位置。

### 5.4 材质状态扩展到四态

材质从过去的三态近似，扩展为：

- `idle`
- `pending_grab`
- `grabbed`
- `rotating`

其中：

- `pending_grab` 是偏暖色的预抓取高亮
- `rotating` 是青色高亮

这和 Bridge 当前状态机保持一致。

### 5.5 手部渲染彻底改为命令驱动

旧逻辑里，Rendering 在 `step()` 中会直接读取 observation/raw landmarks，再自己估计深度、转换 landmark、做 debounce 后更新虚拟手。

这一整段已经被移除。

现在逻辑是：

- `Bridge.process()` 输出 `set_hand_pose`
- `Rendering.push()` 分发到 `_handle_set_hand_pose()`
- `_handle_set_hand_pose()` 把 world_norm 点转成 scene 点
- 然后调用 `VirtualHand.update_points()`

这对调试和职责划分都更干净：

- 手的形态由 Bridge 明确输出
- Rendering 只负责画
- gesture debug observation 不再是 hand overlay 的真实数据源

## 6. 虚拟手的变化

关键文件：`src/rendering/interaction/virtual_hand.py`

这是视觉上变化最大的部分。

### 6.1 从 21 点 debug hand 改成双指风格化 hand

以前的 `VirtualHand` 主要是：

- 21 个 landmark marker
- HAND_CONNECTIONS 骨架线
- index/thumb collider 可视化

本质更像调试骨架。

现在的 `VirtualHand` 改成了 bridge-driven stylized two-finger hand，核心节点是：

- `wrist`
- `anchor`
- `thumb_base`
- `index_base`
- `thumb_tip`
- `index_tip`
- `pinch_center`

再配合若干 box segment：

- `palm_bridge`
- `thumb_root`
- `index_root`
- `thumb_finger`
- `index_finger`
- `pinch_bar`

### 6.2 手的几何输入不再来自原始 landmarks

Bridge 现在只输出双指需要的关键点，而不是 21 landmarks。

特别是新增了两个可选点：

- `thumb_base`
- `index_base`

它们由 Bridge 通过 wrist / anchor / fingertip 推导，用来让手掌和指根结构更自然。

### 6.3 pinch 视觉反馈更明确

现在 pinch bar / pinch center 会根据 thumb-tip 和 index-tip 距离切换三种视觉状态：

- open
- candidate
- locked

对应厚度、颜色、alpha、center scale 都会变化。

这比原先只显示 landmarks 的反馈更直接，也更适合演示交互状态。

## 7. 合同与配置层变化

### 7.1 contract 扩展了 `set_hand_pose`

`src/contract.zh-CN.md` 现在明确了：

- `SceneCommand.command_type` 包含 `set_hand_pose`
- `set_object_state.interaction_state` 允许 `pending_grab` 和 `rotating`
- `set_hand_pose.visible=true` 时必须有：
  - `wrist`
  - `thumb_tip`
  - `index_tip`
  - `anchor`
- 可选有：
  - `thumb_base`
  - `index_base`

这意味着 hand overlay 已经成为 contract 正式部分，不再是 rendering 内部实现细节。

### 7.2 入口配置变化

`main.py` 和 `.run.example.yaml` 这段时间与交互相关的配置主要有：

- `render_position_sensitivity`
- `bridge_rotation_sensitivity`
- `motion_preset`
- `aggressive_release_guard`
- `virtual_hand`

另外，`2b6f13d` 去掉了 camera view configuration knob，当前相机位姿回归 `RenderingCoreManager.camera_pose_for_world_norm()` 的固定定义。

对接手来说，这意味着：

- 现在不需要先理解一套外部 camera view knob
- 先按固定视角理解 scene 即可

## 8. 测试层反映出的当前行为

优先建议看：

- `tests/test_bridge_service.py`
- `tests/test_rendering_service.py`

这些测试基本把当前设计意图说清了。

Bridge 侧被明确测试的行为包括：

- 首包发 `init_scene + set_hand_pose`
- 必须先 hover，才能进入 grab
- grab 使用相对 offset，而不是吸附
- pinch midpoint 作为 interaction anchor
- x 轴在 mirrored 输入下反向
- rotation 模式下只发 `hpr`
- rotation 不再从原始手角度重置物体姿态
- rotation sensitivity 会缩放角度增量
- 松手后如果手还在附近，会回到 hover/pending_grab
- 多物体选择和 rotation target lock 的预期行为
- 桌面阻挡与必要时释放 grabbed object

Rendering 侧被明确测试的行为包括：

- start/stop/restart 时 runtime state 正确重置
- `set_object_pose` 的 rotation-only 更新保留 position
- position sensitivity 会影响坐标应用
- `camera_pose_for_world_norm()` 是固定入口相机定义
- data panel 能格式化 rotation debug 信息

## 9. 当前我认为最重要的设计点

如果你要接回这个模块，我建议先用下面这些判断当前代码，而不是用旧心智模型：

### 9.1 Bridge 现在已经不是“纯转发层”

它已经承担：

- scene layout 真相
- 对象级交互状态
- 交互语义转换
- hand pose 生成

如果后面要继续扩交互，优先应该在 Bridge 想清楚 contract 和状态机，再让 Rendering 被动消费。

### 9.2 Rendering 当前的价值在“执行一致性”

它现在更像：

- contract validator
- scene command applier
- Panda3D adapter
- visual presentation layer

这比让 Rendering 直接读 gesture/raw observation 更容易测，也更容易在未来换 gesture 输入。

### 9.3 scene 初始化已经有“场景描述层”的雏形

`TABLE_SCENE_OBJECTS` 目前还是硬编码在 Bridge 里，但它实际上已经接近一个 scene description/config 的雏形了。

如果后续 demo scene 继续复杂化，一个自然方向是：

- 把 scene descriptor 从 bridge/service.py 中拆出来
- 让 Bridge 读取结构化 scene config

不过当前规模下，先保留在 Bridge 内部也合理。

## 10. 接手建议

如果你要快速接手，我建议按下面顺序读代码：

1. `src/contract.zh-CN.md`
2. `src/bridge/service.py`
3. `src/rendering/service.py`
4. `src/rendering/interaction/virtual_hand.py`
5. `tests/test_bridge_service.py`
6. `tests/test_rendering_service.py`

读的时候重点盯以下问题：

1. `TABLE_SCENE_OBJECTS` 是否应该继续由 Bridge 持有
2. `rotation.debug.mode_active` 是否需要从 debug 字段升级为正式输入字段
3. 桌面约束是否要继续保持在 Bridge，而不是 Rendering
4. `shape` tag 目前只是保留在 Rendering object tag，中期是否要长成真正不同几何体
5. `virtual_hand` 的 config 字段里仍保留了不少旧配置项，是否需要清理成和新实现一致的参数集合

## 11. 已知遗留 / 我会优先关注的点

- `virtual_hand` 的运行配置仍带有旧版 landmark hand 的参数名，例如 `scale/depth_scale/perspective_scale/bone_color/bone_width`，但新 `VirtualHand` 实现实际读取的是 wrist/anchor/thumb/index/pinch 等颜色项。配置层现在有一定历史残留。
- `aggressive_release_guard` 仍在 `main.py` 配置里，但本轮交互逻辑核心释放行为已经更依赖 Bridge 的桌面约束与距离判定，建议后续再确认它和当前状态机的关系是否仍清晰。
- `shape` 目前只影响 tag 和 scene 描述，不影响真正不同 mesh；渲染上仍然复用了 `box` 模型，只通过 scale/color 区分。

## 12. 验证情况

我已基于提交记录、关键实现文件和现有测试代码完成这份交接梳理。

本地未能执行测试，因为当前环境缺少 `pytest` 模块：

- `python3 -m pytest tests/test_bridge_service.py tests/test_rendering_service.py tests/test_main.py -q`
- 结果：`No module named pytest`

因此本文档中的“当前行为”判断，来自代码与测试用例的一致性分析，而不是本地重新跑通测试后的结论。
