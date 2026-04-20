---
papersize: a4
fontsize: 12pt
linestretch: 1.3
geometry: margin=2cm
lang: zh-CN
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
mainfont: "Source Han Serif SC"
CJKmainfont: "Source Han Serif SC"
---

\pagenumbering{gobble}
\begin{titlepage}
\centering
\vspace*{\fill}
{\LARGE \textbf{AeroInteract3D：}\\\textbf{基于普通摄像头的 3D 手势交互系统}\par}
\vspace{0.25cm}
{\large 第十组\par}
\vspace{0.10cm}
{\large 何子谦、徐逸博、陈金龙、夏凡程\par}
\vspace{0.15cm}
{\large 日期：\today\par}
\vspace*{\fill}
\end{titlepage}

\clearpage
\tableofcontents

\clearpage
\pagenumbering{arabic}
\counterwithin{figure}{section}
\counterwithin{table}{section}

# 摘要

AeroInteract3D 是一个基于单目摄像头的实时 3D 手势交互原型系统，无需深度相机等专用设备，即可完成"手部视觉感知 → 交互语义解释 → 三维场景响应"的完整链路，实现桌面场景下的抓取、移动、旋转与可视化反馈。

系统以 Python 开发，用 MediaPipe Hand Landmarker 做手部关键点检测，用 Panda3D 做渲染，通过 `GesturePacket` 与 `SceneCommand` 两个共享契约将 gesture、bridge、rendering 三个模块解耦。系统支持主手与副手双槽位输入，已实现主手优先的双手实时感知链路。

核心成果包括：时序稳定与 pinch 概率状态机（解决检测抖动和短时遮挡问题）、交互语义收敛与坐标映射（Bridge 层）、幂等命令消费（Rendering 层）。整个系统已形成可运行、可测试、可扩展的课程设计原型。

# 项目背景与意义

## 背景

传统三维交互依赖鼠标、键盘、手柄或专用空间交互设备，在无接触交互、低成本原型和自然界面研究场景下存在局限。

随着轻量化视觉感知方案的发展，基于普通摄像头的手势交互成为可行路径。本项目选择"摄像头手势输入驱动三维对象交互"为核心问题，涉及计算机视觉、实时系统、交互设计和软件工程集成，具有较强的综合训练价值。

核心难点在于：单目视觉手势系统面临检测抖动、深度不稳定、短时遮挡、重捕获跳变和模块耦合等问题。即使底层关键点检测可用，缺乏系统级设计也难以形成可操作的三维交互体验。

## 意义

- **技术层面**：将手部关键点检测、时序滤波、状态机控制和三维渲染联结为完整数据链路，验证视觉输入驱动实时场景交互的可行性。
- **工程层面**：通过共享契约和端口抽象划分模块边界，支持并行开发、独立测试和逐步集成。
- **教学层面**：完整展示需求分析、系统设计、模块实现、测试验证和报告整理的全过程。

# 系统概述

## 项目目标

- 基于普通摄像头实现双手感知输入（主手负责默认交互，副手参与协同操作），不依赖深度相机或专用手套。
- 输出统一手势契约数据，保证下游模块稳定消费。
- 实现抓取、移动、旋转等基础交互能力，并提供明确视觉反馈。
- 模块化设计，使系统具备可测试、可维护、可扩展的工程特征。

## 系统总体架构

系统采用"检测驱动、契约传输、命令消费"的三段式流水线：

```
摄像头帧 → 手部检测与时序归约 → GesturePacket → 交互状态机与坐标映射 → SceneCommand → Panda3D 场景更新
```

三个核心模块：

| 模块      | 源码位置         | 职责                                     |
| --------- | ---------------- | ---------------------------------------- |
| gesture   | `src/gesture/`   | 手势采集、检测、平滑、状态稳定、契约输出 |
| bridge    | `src/bridge/`    | 手势语义翻译为场景命令，维护对象交互状态 |
| rendering | `src/rendering/` | 场景初始化、命令消费、对象渲染、调试视图 |

模块间通过 `src/contracts.py` 中的 `GesturePacket` 与 `SceneCommand` 交换数据，不直接共享内部实现。

## 数据流与控制流

**单帧数据流**（响应速度）：

1. `CaptureRuntime` 读取摄像头帧并做镜像翻转
2. `HandLandmarkerRuntime` 完成关键点检测、手尺度估计和深度提示计算
3. 若检测缺失，执行模板匹配与速度外推（fallback）
4. `TemporalReducer` 对关键点、pinch 状态和跟踪状态进行时序稳定
5. 输出稳定的 `GesturePacket`
6. `BridgeServiceImpl.process()` 生成 `SceneCommand`
7. `RenderingServiceImpl.push()` 接收命令，渲染循环在后续 `step()` 中应用

**跨帧状态流**（交互连续性）：

- Gesture 侧：上一帧关键点、速度、pinch 状态、重捕获混合进度
- Bridge 侧：当前悬停对象、抓取对象、旋转对象及其参考位姿
- Rendering 侧：对象缓存、显示状态、命令去重信息、UI 视图状态

## 主循环

`main.py` 中的 `App` 类负责系统生命周期和主运行循环：

```python
def run(self) -> None:
    frame_interval = 1.0 / max(self.config.target_fps, 1)
    self._running = True

    while self._running:
        loop_start = time.perf_counter()
        self.render_output.step()

        packet = self.gesture_input.poll()
        if packet is not None:
            self.render_output.update_gesture_data(packet)
            commands = self.bridge.process(packet)
            for command in commands:
                self.render_output.push(command)

        elapsed = time.perf_counter() - loop_start
        if (sleep_for := frame_interval - elapsed) > 0:
            time.sleep(sleep_for)
```

配置通过 `AppConfig` 管理，支持 YAML 配置文件覆盖默认参数（摄像头编号、目标帧率、分辨率、镜像选项、Bridge 旋转灵敏度等）。

## 共享契约

契约定义在 `src/contracts.py`，当前版本基线为 `2.0.0`：

- **`GesturePacket`**：描述单帧手势观测及其时序语义。核心字段包括 `frame_id`、`timestamp_ms`、`tracking_state`（`tracked / temporarily_lost / not_detected`）、`pinch_state`（`open / pinch_candidate / pinched / release_candidate`）、`index_tip`、`thumb_tip`、`wrist`、`coordinate_space`。

- **`SceneCommand`**：描述场景更新指令。核心字段包括 `command_id`、`command_type`、`object_id` 和 `payload`。支持的命令类型有 `init_scene`、`set_hand_pose`、`set_object_state`、`set_object_pose`、`reset_interaction`、`heartbeat`。

关键设计决策：所有坐标必须携带坐标空间元数据；视觉语义只能由上游解释一次，渲染端不再重复推断手势含义。

# 逐模块介绍

## Gesture 模块

### 模块职责

`src/gesture/service.py` 中的 `GestureServiceImpl` 是系统的输入端，负责：

- 从摄像头采集视频帧
- 调用 MediaPipe Hand Landmarker 检测手部关键点
- 在短时检测失败时执行有限预测和回退（fallback）
- 通过时序归约器生成稳定的 `GesturePacket`
- 为调试视图提供相机帧和中间观测信息

### 核心算法链路

```
图像预处理 → 模糊度估计 → 手部检测 → 关键点归一化 → 局部回退/运动预测 → 概率式 pinch 判定 → 时序平滑 → 稳定输出
```

本项目不包含从零开始的深度模型训练，感知基础建立在 MediaPipe Hand Landmarker 之上。项目工作重点在于其上的工程集成与时序交互算法设计——即将不稳定的视觉观测转化为稳定的交互输入。

### 输入预处理与模糊估计

```python
frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
blur_level = self._estimate_blur_level(frame_gray)
detect_frame = resize_for_detection(frame_bgr, max_side=GESTURE_DETECT_MAX_SIDE)
rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
result = self._landmarker.detect_for_video(image, timestamp_ms)
```

系统从彩色图像中提取灰度图，用 Laplacian 方差衡量清晰度 `blur_level`（方差越低图像越模糊），再将输入缩放到固定上限以控制检测负载。`blur_level` 不是附带指标，而是后续平滑强度、质量评分和释放保护的统一输入。

### 关键点归一化与尺度修正

```python
landmarks = [Vec3(x=float(lm.x), y=float(lm.y), z=float(lm.z)) for lm in hand_landmarks]
hand_scale = estimate_hand_scale(landmarks)
depth_hint = estimate_hand_depth(landmarks, hand_scale)
index_tip = landmark_to_camera_vec3(landmarks[INDEX_TIP_LANDMARK_INDEX], depth_hint=depth_hint)
thumb_tip = landmark_to_camera_vec3(landmarks[THUMB_TIP_LANDMARK_INDEX], depth_hint=depth_hint)
pinch_distance = normalized_pinch_distance(
    landmarks[INDEX_TIP_LANDMARK_INDEX],
    landmarks[THUMB_TIP_LANDMARK_INDEX],
    hand_scale=hand_scale,
)
```

MediaPipe Hand Landmarker 输出 21 个手部关键点，系统重点使用 `wrist`、`thumb_tip` 和 `index_tip` 作为交互主锚点。手尺度归一化是工程关键点：手离摄像头远近显著改变像素级指尖距离，不做尺度修正则 pinch 判定会严重依赖手与摄像头的相对距离，导致抓取状态不稳定。

### 检测失败时的 fallback

```python
if not result.hand_landmarks:
    fallback = self._detect_fallback(frame_bgr, frame_gray, timestamp_ms, blur_level)
    return [] if fallback is None else [fallback]
```

当检测器当帧失效时，系统不会完全丢弃上一帧结果。Fallback 策略采用"速度外推 + 局部模板匹配"：根据上一帧 wrist 在图像平面中的位移估计当前关键点的新位置，再尝试用局部模板匹配寻找更可信的候选区域。如果模板匹配成功，就用新 wrist 位置带动整组关键点平移；如果失败，则继续短时速度预测。其价值不在于提高静态识别精度，而在于保证交互不因单帧检测缺失而立刻断裂。

### 概率式 pinch 判定

```python
pinched_likelihood = self._gaussian(raw_pinch_distance, mean=0.06, sigma=0.04)
open_likelihood = self._gaussian(raw_pinch_distance, mean=0.18, sigma=0.08)
likelihood_sum = max(pinched_likelihood + open_likelihood, 1e-6)
geometry_score = pinched_likelihood / likelihood_sum

prior_score = 0.78 if self._is_pinch_stable() else 0.24
pinch_score = (
    weight_geometry * geometry_score
    + weight_prior * prior_score
    + weight_appearance * self._clamp01(appearance_match_score)
)
```

系统没有把 pinch 语义简化为"距离小于阈值"，而是把当前几何距离映射成两类高斯似然（`pinched` / `open`），再结合先验状态和外观匹配分数形成 `pinch_score`。这相当于构造了一种轻量的概率式语义判定方法。进入 pinch 需要连续确认帧，释放也需要满足速度、质量等额外条件。相比硬阈值，边界过渡更平滑，能够更自然地与时序状态机结合。

### pinch 状态机

```python
if pinch_score > ENTER_THRESHOLD:
    self._pinch_confirm_count += 1
    if self._pinch_confirm_count >= PINCH_CONFIRM_FRAMES:
        self._last_pinch_state = "pinched"
        return self._last_pinch_state

release_allowed = self._allow_release(
    pinch_score=pinch_score,
    quality_score=quality_score,
    geometry_open_margin=geometry_open_margin,
    source=source,
    wrist_speed=wrist_speed,
)
```

pinch 判定采用四态状态机（`open → pinch_candidate → pinched → release_candidate → open`），而非单阈值开关。进入抓取需要连续确认，退出抓取也需要独立确认。在快速移动或低质量帧下会主动提高释放门槛。这种设计在交互稳定性与响应速度之间做了平衡——如果只采用静态阈值，系统会在对象移动过程中频繁出现"刚抓住又松开"的现象。

### 时序稳定策略（分层）

| 层次 | 策略         | 作用                                                 |
| ---- | ------------ | ---------------------------------------------------- |
| 1    | 坐标平滑     | 对 x/y/z 三个方向使用不同响应策略的指数型平滑        |
| 2    | 质量感知修正 | 根据 `blur_level` 调节平滑强度，模糊时自动提高保守性 |
| 3    | 短时失跟预测 | 利用最近速度做短窗口预测，按缺失帧数逐步衰减         |
| 4    | 重捕获回接   | 检测恢复时做若干帧混合回接，限制单帧回跳幅度         |
| 5    | 语义状态滞回 | pinch 四态状态机，进入和退出有独立门限与确认条件     |

```python
if low_quality or blur_level > HIGH_BLUR_LEVEL:
    alpha_y = max(base_alpha_y * LOW_QUALITY_MOTION_ALPHA_Y_MULTIPLIER, LOW_QUALITY_MOTION_ALPHA_Y_FLOOR)
elif motion_y > 0.04:
    alpha_y = max(base_alpha_y, HIGH_MOTION_ALPHA_Y_FLOOR)
else:
    alpha_y = base_alpha_y

factor = self.tuning.prediction_blend * (self.tuning.lost_tracking_motion_damping ** self._missing_frames)
lead = self.tuning.prediction_lead * max(self._missing_frames, 1)
```

系统根据 `blur_level` 和瞬时运动量动态调整平滑系数，而不是把所有帧按同一种滤波强度处理。当检测中断时，通过速度、预测前瞻量和阻尼系数对位置进行短时外推。这不是普通均值滤波，而是"质量感知平滑 + 常速度短时预测"的组合式时序算法。

### 测试与验证

- 稳定手部观测时，输出是否满足契约字段要求
- 遮挡或模糊情况下，fallback 与预测是否保持输出连续性
- pinch 状态迁移是否符合确认帧和释放保护规则

### 典型工程问题：模糊帧与误检抑制

低光照、快速移动或镜头晃动时手部关键点出现明显漂移，导致 pinch 语义和对象位置随之抖动。采用三种手段共同应对：模糊度感知平滑（根据图像质量调节保守程度）、局部模板匹配（在检测失败时恢复短时观测）、短时速度预测（维持轨迹连续性）。三者分别负责判定输入质量、恢复短时观测和保持交互连贯性。

---

## Bridge 模块

### 模块职责

`src/bridge/service.py` 中的 `BridgeServiceImpl` 是语义转换层，负责：

- 校验 `GesturePacket` 合法性与时序有效性
- 维护对象交互状态机（`idle`、`pending_grab`、`grabbed`、`rotating`）
- 将 `camera_norm` 输入映射到 `world_norm` 场景坐标
- 输出有序 `SceneCommand` 序列
- 异常情况下发出安全重置命令

核心价值是**语义收敛**——Gesture 只负责给出稳定的输入意图，Bridge 决定对象是否进入悬停、抓取或旋转。交互策略集中在 Bridge 层而非分散在多个模块中。

### 交互规则

Bridge 的交互规则具有三个明确特点：

- **抓取前必须先进入悬停区域**：系统不会在"手刚 pinched 但尚未接近对象"时直接抓取，降低误操作概率
- **使用指尖中点作为手部锚点**：抓取位置使用指尖中点而非单点关键点，让对象跟随更符合 pinch 操作直觉
- **旋转与平移模式分离**：只有当上游明确给出旋转模式激活信息时，Bridge 才输出 `hpr` 更新，否则优先维持位置控制链路

### 坐标映射

```python
unclipped_world_x = -x if self._input_mirrored else x  # 镜像输入横向翻转
unclipped_world_y = y
unclipped_world_z = -z  # 相机深度转场景深度方向

final_x = max(-1.0, min(1.0, unclipped_world_x))
final_y = max(-1.0, min(1.0, unclipped_world_y))
final_z = max(-1.0, min(1.0, unclipped_world_z))
```

坐标映射包含三层含义：方向修正（镜像输入下做横向翻转、相机深度方向转场景深度方向）、尺度约束和场景边界控制（输出限制在 `[-1, 1]` world_norm 范围内）。此外，在拖拽对象时还会结合桌面高度和平面约束，避免对象穿透桌面。

### 命令生成

| 命令类型            | 用途                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `init_scene`        | 初始化场景对象集合                                                |
| `set_hand_pose`     | 同步虚拟手或调试手部姿态                                          |
| `set_object_state`  | 切换对象交互状态（`idle`、`pending_grab`、`grabbed`、`rotating`） |
| `set_object_pose`   | 更新对象的位置、姿态或缩放                                        |
| `reset_interaction` | 跟踪丢失等场景下恢复安全状态                                      |

Bridge 还内置了桌面场景对象配置（桌面、主立方体、平板、立柱等），系统已具备多对象场景组织能力。Bridge 同时承担输入包过滤与异常兜底职责：对于重复包、过期帧、契约不合法数据或持续跟踪丢失情况，模块不会继续发出不安全的姿态命令，而是根据情况忽略、拒绝或下发 `reset_interaction`。

### 旋转与平移分流

```python
rotation_hpr = self._rotation_hpr_payload(packet, object_state)
if rotation_hpr is not None:
    payload["hpr"] = rotation_hpr
    return payload

world_position, _ = self._drag_world_position(packet, object_state, hand_anchor_world)
object_state.world_position = world_position
payload["position"] = vec3_payload(world_position)
```

Bridge 对"旋转"和"平移"两类交互做显式分流：若当前处于旋转模式则优先构造姿态命令，否则继续走位置拖拽链路。同一帧内避免同时向对象写入相互冲突的位置与姿态语义。

### 手部可视化命令

```python
if packet.tracking_state == "tracked" and packet.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE:
    index_tip_world = self._camera_to_world_position(packet.index_tip)
    thumb_tip_world = self._camera_to_world_position(packet.thumb_tip)
    wrist_world = self._camera_to_world_position(packet.wrist)
    anchor_world = self._camera_to_world_position(self._interaction_anchor(packet))
    payload["visible"] = True
    payload["points"] = {
        "wrist": vec3_payload(wrist_world),
        "thumb_tip": vec3_payload(thumb_tip_world),
        "index_tip": vec3_payload(index_tip_world),
        "anchor": vec3_payload(anchor_world),
    }
```

Bridge 不仅负责对象控制，还负责把 Gesture 的关键交互点翻译成渲染可见的世界坐标。渲染模块不需要理解任何视觉检测细节，只需消费统一的世界坐标点集即可完成虚拟手显示。

### 测试与验证

- 首帧有效输入是否先发 `init_scene` 再进入交互命令链
- 光标进入对象邻域后，是否先产生悬停状态，再允许抓取
- 仅在旋转模式激活时，是否输出 `hpr` 更新而非位置更新
- 过期帧、重复包和持续失跟时，是否进入安全重置路径

### 典型工程问题

**旋转和平移冲突**：抓取、旋转和平移可能被同一帧输入同时触发，导致对象既被拖动又被旋转的冲突行为。Bridge 层通过模式检测将两类交互分开处理，旋转模式只输出姿态命令，平移模式只输出位置命令。

**过期命令与状态残留**：手已离开视野但对象仍停留在抓取状态，或上一帧命令延迟到达后覆盖当前状态。渲染端通过命令去重、过期帧忽略和 `reset_interaction` 恢复机制，把这类问题控制在安全范围内。

---

## Rendering 模块

### 模块职责

`src/rendering/service.py` 中的 `RenderingServiceImpl` 负责消费 `SceneCommand` 并在 Panda3D 中更新场景：

- 初始化 Panda3D 窗口、相机和基础光照
- 创建场景对象并维护其视觉状态
- 按顺序消费命令并忽略重复或过期命令
- 渲染对象位姿变化、交互高亮和手部可视化
- 提供调试面板、摄像头预览和设置视图

渲染模块将"命令消费逻辑"和"图形资源管理"进行了分离：`RenderingCoreManager` 负责底层 Panda3D 上下文和场景资源，`RenderingServiceImpl` 负责契约校验、状态维护和命令应用。这种分工使测试可以在较轻量的假窗口环境下进行。

### 命令分发

```python
command_type = command.command_type
if command_type == "init_scene":
    self._handle_init_scene(command)
elif command_type == "set_object_pose":
    self._handle_set_object_pose(command)
elif command_type == "set_object_state":
    self._handle_set_object_state(command)
elif command_type == "set_hand_pose":
    self._handle_set_hand_pose(command)
elif command_type == "reset_interaction":
    self._handle_reset_interaction(command)
elif command_type == "heartbeat":
    self._metrics.heartbeats_received += 1
    self._metrics.commands_applied += 1
```

渲染模块显式区分不同命令类型，分别做 payload 结构检查。不同命令有不同处理入口，场景初始化、位姿更新、手部显示和交互恢复分别实现。这种"命令驱动架构"有利于测试，也方便演示时单独说明某个功能链路。

### 位姿命令校验

```python
has_position = "position" in payload
has_hpr = "hpr" in payload

if has_position:
    pos_data = payload["position"]
    if isinstance(pos_data, dict) and all(k in pos_data for k in ["x", "y", "z"]):
        pos = [pos_data["x"], pos_data["y"], pos_data["z"]]
    else:
        self._record_error(...)
        return
```

渲染模块不是盲目应用上游命令，而是先对位姿数据进行结构检查。只有当 `position` 或 `hpr` 满足契约要求时，渲染层才继续执行对象更新。这样可以显著降低跨模块联调时的错误传播——错误被限制在模块边界内而不是扩散到整个系统。

### 幂等消费与异常处理

- 对命令类型和 payload 键集合进行显式检查
- 对重复命令和过期命令执行安全忽略
- 对隐藏对象或格式错误命令输出运行时告警，而不是直接崩溃
- `reset_interaction` 时恢复对象状态，避免交互残留

渲染模块需要面对的主要风险不是"图形算法错误"，而是命令乱序、重复、字段缺失和运行时资源加载失败。这种设计使渲染模块可以作为稳定的命令消费端运行，而不是把所有正确性都押在上游模块上。

### 场景初始化与对象管理

`init_scene` 命令一次性下发对象描述（标识、初始位置/姿态、缩放、颜色、形状类型、是否可交互等），渲染模块收到后建立对象缓存，后续仅根据增量命令更新状态。对象定义与对象更新被清晰分开，减少了命令冗余，也方便后续扩展新模型。

当前渲染实现还支持通过模型模板工厂统一注册内置模型和自定义模型，并对材质、锚点偏移、双面渲染和懒加载做集中管理。对于课程设计原型来说，这种资源工厂模式足以支撑中小规模场景扩展。

### 测试与验证

- `init_scene` 是否建立对象缓存并设置初始状态
- `set_object_pose` 是否正确解析 `position`、`hpr` 和 `scale`
- `set_object_state` 是否将交互状态映射为可视化状态
- `set_hand_pose` 是否只在条件满足时显示虚拟手
- `reset_interaction` 是否能够把场景恢复到安全默认值

这一层测试的价值是把"场景可显示"进一步提升为"场景可重复、可恢复"。对课程设计答辩来说，稳定性往往比单纯的视觉效果更重要，因为它直接决定演示是否容易出错。

---

## 主循环联调

Gesture、Bridge 和 Rendering 都可用轻量接口对象替换，验证三者在统一节拍下完成数据传递。可确认 `poll → process → push → step` 的调用顺序没有被破坏，也能检查生命周期管理是否一致。

联调阶段暴露的两个主要问题：

1. **坐标空间理解必须完全一致**：靠契约定义解决——所有坐标必须携带坐标空间元数据
2. **命令消费必须严格按帧序和状态序执行**：靠幂等分发和状态机解决——重复命令被安全忽略，状态迁移严格按序

---

## 自定义模型接入

自定义模型如果没有统一命名和注册方式，会造成配置碎片。当前实现通过模型模板工厂把基础几何体和自定义模型统一到同一套注册入口，对材质、锚点偏移、双面渲染和懒加载做集中管理，降低了新增资源的成本，也减少了模型接入时的人工错误。

# 性能评价

本节待补充统一测试环境下的定量数据。计划测试内容：

- 主循环帧率与渲染链路延迟
- 手势检测端到端延迟
- 命令吞吐与丢帧情况
- 典型交互（抓取、移动、旋转）成功率与稳定性

# 系统展示

## 建议展示流程

"初始化 → 识别 → 悬停 → 抓取 → 移动 → 旋转 → 丢失恢复"

## 展示交互流程

- 启动后桌面场景初始化
- 手部进入视野后的预览与虚拟手显示
- 指尖靠近对象触发悬停高亮
- 稳定 pinch 后抓取对象并跟随移动
- 进入旋转模式后对对象姿态进行调整
- 跟踪丢失后系统自动恢复到安全状态

## 素材占位

| 素材                   | 路径                                              |
| ---------------------- | ------------------------------------------------- |
| 系统主界面截图         | `report/assets/system-ui-placeholder.png`         |
| 手势跟踪与调试面板截图 | `report/assets/debug-panel-placeholder.png`       |
| 对象抓取过程截图       | `report/assets/grab-sequence-placeholder.png`     |
| 对象旋转过程截图       | `report/assets/rotation-sequence-placeholder.png` |

# 总结

本项目完成了从手势检测、时序稳定、交互语义解释到 Panda3D 场景响应的完整系统链路。

核心工程成果：

1. **共享契约与端口抽象**：模块边界清晰，便于协作开发与独立测试
2. **面向场景的时序稳定与状态机机制**：提升了原型的可操作性
3. **渲染端场景、UI、调试和模型扩展能力**：为后续完善提供基础

团队在以下方面有显著收获：

- **算法层面**：认识到视觉交互系统的难点不只是调用现成检测器，更在于把不稳定的观测变成稳定的交互语义。模糊感知平滑、pinch 概率评分、短时预测和重捕获回接等实践，让团队对实时 CV 系统中的鲁棒性问题有了更直接的理解。
- **软件工程层面**：经历了契约设计、模块解耦、接口联调、异常处理和测试补齐等过程。系统级实现训练了成员对架构边界、状态管理和可维护性的判断能力。
- **协作层面**：体会到"统一数据契约"和"明确模块职责"的重要性。Gesture、Bridge 和 Rendering 之所以能够逐步收敛并联调成功，本质上依赖于前期对职责边界的约束，而非临时性的代码互相适配。

后续工作建议：

- 补充统一环境下的性能测试数据
- 完善截图与展示材料
- 继续收敛多手交互、深度估计和参数标定问题

# 项目成员与分工

- **何子谦**（2025080905004）：架构设计。负责系统架构设计、技术选型、功能兜底、报告撰写。
- **徐逸博**（2025080905025）：核心算法。负责手势检测链路实现、稳定坐标相关算法实现。
- **陈金龙**（2025080905001）：交互与 3D。负责 Panda3D 场景渲染实现、三维物体控制逻辑开发。
- **夏凡程**（2025080905020）：文档与报告。负责开题报告、周报、PPT 制作。

# 附录

## 项目结构

```
AeroInteract3D/
├── main.py                     # 主循环入口
├── src/
│   ├── gesture/                # 手势模块
│   │   ├── service.py          # GestureServiceImpl（主类）
│   │   ├── runtime.py          # CaptureRuntime、HandLandmarkerRuntime
│   │   ├── temproal.py         # TemporalReducer（时序归约与 pinch 状态机）
│   │   ├── constants.py
│   │   └── debug/              # 实时预览工具
│   ├── bridge/                 # 语义桥接模块
│   │   └── service.py          # BridgeServiceImpl
│   ├── rendering/              # 渲染模块
│   │   ├── service.py          # RenderingServiceImpl
│   │   ├── rendering_core.py   # Panda3D 底层封装
│   │   ├── ui/                 # 设置、校准等 UI 视图
│   │   ├── interaction/        # 虚拟手渲染
│   │   └── debug/              # 摄像头预览、数据面板
│   └── contracts.py            # GesturePacket、SceneCommand 契约
├── models/
│   └── hand_landmarker.task    # MediaPipe 模型文件
├── tests/                      # 单元测试与集成测试
├── assets/custom_models/       # 自定义 3D 模型资源
└── report/                     # 技术报告与文档
```

## 运行与测试命令

```bash
make setup   # 安装依赖
make run     # 运行主程序
make test    # 运行测试
make report  # 构建 PDF 报告
```
