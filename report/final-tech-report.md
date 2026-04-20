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

## 模块概述

- **Gesture 模块**（`src/gesture/`）：系统输入端。从摄像头采集视频帧，调用 MediaPipe Hand Landmarker 检测手部关键点，稳定坐标以及交互状态，生成标准化 `GesturePacket`，供下游消费。
- **Bridge 模块**（`src/bridge/`）：语义转换层，系统的"胶水"。消费 `GesturePacket`，维护可交互对象对象状态表，进行坐标映射，计算用于 3D 渲染的坐标，并生成 `SceneCommand` 序列，供渲染模块消费。
- **Rendering 模块**（`src/rendering/`）：系统输出端。消费 `SceneCommand`，在 Panda3D 中进行场景渲染、UI 视图和虚拟手显示。

| 模块      | 源码位置         | 主要类                 |
| --------- | ---------------- | ---------------------- |
| gesture   | `src/gesture/`   | `GestureServiceImpl`   |
| bridge    | `src/bridge/`    | `BridgeServiceImpl`    |
| rendering | `src/rendering/` | `RenderingServiceImpl` |

模块间通过 `src/contracts.py` 中的 `GesturePacket` 与 `SceneCommand` 交换数据，实现模块解耦和职责分离。

# 逐模块介绍

## 系统架构详解

系统采用"检测驱动、契约传输、命令消费"的三段式流水线，三个核心模块通过共享契约 `GesturePacket` 与 `SceneCommand` 解耦，如图 \ref{fig:system-arch} 所示。

```{=latex}
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  >=Stealth,
  every node/.style={font=\small},
  mainblock/.style={
    draw, rounded corners=3pt,
    minimum width=6.2cm, minimum height=0.75cm,
    text centered, inner sep=5pt
  },
  contblock/.style={
    mainblock, fill=orange!20, draw=orange!70!black,
    font=\small\ttfamily
  },
  arr/.style={->, thick},
]

\node[mainblock, fill=gray!25] (cam) at (0, 0) {摄像头输入};

\node[mainblock, fill=teal!20] (cap) at (0, -1.6)
  {CaptureRuntime（帧采集 · 预处理）};
\node[mainblock, fill=teal!20] (lm)  at (0, -2.6)
  {HandLandmarkerRuntime（关键点检测 · 推算深度）};
\node[mainblock, fill=teal!20] (tr)  at (0, -3.6)
  {TemporalReducer（时序平滑 · pinch 手势状态机）};

\begin{scope}[on background layer]
  \node[draw=teal!60, fill=teal!8, rounded corners=6pt, line width=1.2pt,
        inner sep=10pt, fit=(cap)(lm)(tr)] (gmod) {};
\end{scope}
\node[font=\footnotesize\bfseries, text=teal!70!black,
      fill=white, inner sep=1pt, anchor=north west]
  at (gmod.north west) {Gesture 模块};

\node[contblock] (gp) at (0, -5.2)
  {GesturePacket};

\node[mainblock, fill=blue!15] (br) at (0, -6.8)
  {BridgeServiceImpl（可交互对象状态管理 · 坐标映射）};

\begin{scope}[on background layer]
  \node[draw=blue!50, fill=blue!6, rounded corners=6pt, line width=1.2pt,
        inner sep=10pt, fit=(br)] (bmod) {};
\end{scope}
\node[font=\footnotesize\bfseries, text=blue!70!black,
      fill=white, inner sep=1pt, anchor=north west]
  at (bmod.north west) {Bridge 模块};

\node[contblock] (sc) at (0, -8.4)
  {SceneCommand};

\node[mainblock, fill=purple!15] (rs)  at (0, -10.0)
  {RenderingServiceImpl（命令处理 · UI 逻辑 · 模型加载）};
\node[mainblock, fill=purple!15] (p3d) at (0, -11.2)
  {Panda3D 场景（场景/物体渲染 · 虚拟手 · UI 视图）};

\begin{scope}[on background layer]
  \node[draw=purple!50, fill=purple!5, rounded corners=6pt, line width=1.2pt,
        inner sep=10pt, fit=(rs)(p3d)] (rmod) {};
\end{scope}
\node[font=\footnotesize\bfseries, text=purple!70!black,
      fill=white, inner sep=1pt, anchor=north west]
  at (rmod.north west) {Rendering 模块};

\draw[arr] (cam.south)  -- (cap.north);
\draw[arr] (cap.south)  -- (lm.north);
\draw[arr] (lm.south)   -- (tr.north);
\draw[arr] (gmod.south) -- (gp.north);
\draw[arr] (gp.south)   -- (bmod.north);
\draw[arr] (bmod.south) -- (sc.north);
\draw[arr] (sc.south)   -- (rmod.north);
\draw[arr] (rs.south)   -- (p3d.north);

\end{tikzpicture}
\caption{系统总体架构与数据流}
\label{fig:system-arch}
\end{figure}
```

## 共享契约

模块间通过 `src/contracts.py` 中定义的 `GesturePacket` 和 `SceneCommand` 进行数据交换，形成清晰的输入输出边界：

**`GesturePacket`**：描述单帧手势检测结果。

| 字段 | 说明 |
| ---- | ---- |
| `frame_id` / `timestamp_ms` | 帧序号与时间戳 |
| `tracking_state` | `tracked` / `temporarily_lost` / `not_detected` |
| `pinch_state` | `open` / `pinch_candidate` / `pinched` / `release_candidate` |
| `index_tip` / `thumb_tip` / `wrist` | 食指尖、拇指尖、腕部三维坐标 |
| `coordinate_space` | 坐标空间标记，下游必须据此做单位转换 |

**`SceneCommand`**：描述渲染模块需要执行的场景更新命令。

| 字段 | 说明 |
| ---- | ---- |
| `command_id` | 唯一命令标识 |
| `command_type` | `init_scene` / `set_hand_pose` / `set_object_state` / `set_object_pose` / `reset_interaction` / `heartbeat` |
| `object_id` | 命令目标对象 ID |
| `payload` | 命令附带数据，结构随 `command_type` 不同而变化 |

通过共享契约，模块之间实现了松耦合：Gesture 只需保证输出满足 `GesturePacket` 的字段要求，Bridge 只需保证输出满足 `SceneCommand` 的字段要求，渲染模块则只需根据命令类型和 payload 结构进行处理，而不需要关心上游的具体实现细节。

## Bridge 模块

### 模块职责

Bridge 模块是系统的语义转换层，负责把 Gesture 模块输出的 `GesturePacket` 转化为渲染模块可消费的 `SceneCommand`。其核心职责包括：

- 校验 `GesturePacket` 合法性并过滤异常输入
- 维护一张对象交互表，对每个对象记录交互状态（`idle`、`pending_grab`、`grabbed`、`rotating`）
- 将 `camera_norm` 输入映射到 `world_norm` 场景坐标
- 输出有序 `SceneCommand` 序列

Gesture 只负责给出稳定的手势状态，Bridge 决定对象是否应该被悬停、抓取、旋转或缩放。

### 交互规则

Bridge 的交互规则设计如下：

- 手必须先进入对象附近区域（`pending_grab`），pinch 后才会进入抓取状态，避免误触。
- 单独设置旋转模式，只有在旋转模式激活时才输出设置物体姿态的命令，避免物体移动和旋转命令冲突。
- 画面中有两只手时，可以实现缩放物体效果：当两只手同时悬停在同一对象上并进入`pinched` 状态时，双手的移动会被转换为对物体的缩放命令。

### 坐标映射与场景约束

Bridge 模块负责把 Gesture 模块输出的 `camera_norm` 坐标映射到渲染模块使用的 `world_norm` 坐标。映射过程中会进行：

- 方向修正：根据输入视频是否镜像，对画面做横向翻转；相机深度方向转场景深度方向。
- 尺度约束和场景边界控制：输出限制在 `[-1, 1]` world_norm 范围内。

以下代码片段展示了坐标映射的核心逻辑：

```python
unclipped_world_x = -x if self._input_mirrored else x  # 镜像输入横向翻转
unclipped_world_y = y
unclipped_world_z = -z  # 相机深度转场景深度方向

final_x = max(-1.0, min(1.0, unclipped_world_x))
final_y = max(-1.0, min(1.0, unclipped_world_y))
final_z = max(-1.0, min(1.0, unclipped_world_z))
```

在拖拽对象时还会结合桌面高度和平面约束，避免对象穿透桌面：使用 `_constrain_object_to_table` 把对象中心的 y 坐标限定在桌面以上，赋予物体真实的物理性质。

```python
def _constrain_object_to_table(self, world_position, object_state):
    minimum_center_y = TABLE_SURFACE_Y + object_state.half_height
    if world_position.y >= minimum_center_y:
        return world_position, False
    return Vec3(world_position.x, minimum_center_y, world_position.z), True
```

### 生成 `SceneCommand`

| 命令类型            | 用途                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `init_scene`        | 初始化场景                                                |
| `set_hand_pose`     | 同步虚拟手姿态                                          |
| `set_object_state`  | 切换对象交互状态（`idle`、`pending_grab`、`grabbed`、`rotating`） |
| `set_object_pose`   | 更新对象的位置、姿态或缩放状态                                        |
| `reset_interaction` | 输入无效或 Gesture 模块异常情况下重置物体状态                                      |

Bridge 还内置了桌面场景对象管理能力，支持导入 3D 模型，可以对多对象场景进行组织和管理。Bridge 同时承担输入包过滤与异常兜底职责：对于重复包、过期帧、契约不合法数据或持续跟踪丢失情况，模块不会继续发出不安全的姿态命令，而是根据情况忽略、拒绝或下发 `reset_interaction`。

### 手部可视化命令

Bridge 不仅负责对象控制，还负责输出手部可视化命令。当 `GesturePacket` 中的 `tracking_state` 是 `tracked` 且 `confidence` 超过设定阈值时，Bridge 会把食指尖、拇指尖、腕部和交互锚点的坐标转换为世界坐标，并通过 `set_hand_pose` 命令发送给渲染模块，供其在场景中渲染虚拟手，使用户能够看到手部位置和 pinch 状态的实时反馈。

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

### 典型工程问题

**旋转和平移冲突**：抓取、旋转和平移可能被同一帧输入同时触发，导致对象既被拖动又被旋转的冲突行为。Bridge 层通过设置不同模式，将两类交互分开处理，旋转模式只输出姿态命令，平移模式只输出位置命令。

**过期命令与状态残留**：手已离开视野但对象仍停留在抓取状态，或上一帧命令延迟到达后覆盖当前状态。渲染端通过命令去重、过期帧忽略和 `reset_interaction` 恢复机制，把这类问题控制在安全范围内。

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

系统从彩色图像中提取灰度图，用 Laplacian 方差衡量清晰度 `blur_level`（方差越低图像越模糊），再将输入缩放到固定上限以控制检测负载。`blur_level` 不是附带指标，而是后续平滑强度、质量评分和释放保护的统一输入。

```python
frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
blur_level = self._estimate_blur_level(frame_gray)
detect_frame = resize_for_detection(frame_bgr, max_side=GESTURE_DETECT_MAX_SIDE)
rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
result = self._landmarker.detect_for_video(image, timestamp_ms)
```

### 关键点归一化与尺度修正

MediaPipe Hand Landmarker 输出 21 个手部关键点，系统重点使用 `wrist`、`thumb_tip` 和 `index_tip` 作为交互主锚点。手尺度归一化是工程关键点：手离摄像头远近显著改变像素级指尖距离，不做尺度修正则 pinch 判定会严重依赖手与摄像头的相对距离，导致抓取状态不稳定。

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

### 检测失败时的 fallback

当检测器当帧失效时，系统不会完全丢弃上一帧结果。Fallback 策略采用"速度外推 + 局部模板匹配"：根据上一帧 wrist 在图像平面中的位移估计当前关键点的新位置，再尝试用局部模板匹配寻找更可信的候选区域。如果模板匹配成功，就用新 wrist 位置带动整组关键点平移；如果失败，则继续短时速度预测。其价值不在于提高静态识别精度，而在于保证交互不因单帧检测缺失而立刻断裂。

```python
if not result.hand_landmarks:
    fallback = self._detect_fallback(frame_bgr, frame_gray, timestamp_ms, blur_level)
    return [] if fallback is None else [fallback]
```

### 概率式 pinch 判定

系统没有把 pinch 语义简化为"距离小于阈值"，而是把当前几何距离映射成两类高斯似然（`pinched` / `open`），再结合先验状态和外观匹配分数形成加权 `pinch_score`。进入 pinch 需要连续确认帧，释放也需要满足速度、质量等额外条件。相比硬阈值，边界过渡更平滑，能够更自然地与时序状态机结合。

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

### pinch 状态机

pinch 判定采用四态状态机（`open → pinch_candidate → pinched → release_candidate → open`），而非单阈值开关。进入抓取需要连续确认，退出抓取也需要独立确认。在快速移动或低质量帧下会主动提高释放门槛。这种设计在交互稳定性与响应速度之间做了平衡——如果只采用静态阈值，系统会在对象移动过程中频繁出现"刚抓住又松开"的现象。

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

### 时序稳定策略（分层）

| 层次 | 策略         | 作用                                                 |
| ---- | ------------ | ---------------------------------------------------- |
| 1    | 坐标平滑     | 对 x/y/z 三个方向使用不同响应策略的指数型平滑        |
| 2    | 质量感知修正 | 根据 `blur_level` 调节平滑强度，模糊时自动提高保守性 |
| 3    | 短时失跟预测 | 利用最近速度做短窗口预测，按缺失帧数逐步衰减         |
| 4    | 重捕获回接   | 检测恢复时做若干帧混合回接，限制单帧回跳幅度         |
| 5    | 语义状态滞回 | pinch 四态状态机，进入和退出有独立门限与确认条件     |

系统根据 `blur_level` 和瞬时运动量动态调整平滑系数，而不是把所有帧按同一种滤波强度处理。当检测中断时，通过速度、预测前瞻量和阻尼系数对位置进行短时外推。这不是普通均值滤波，而是"质量感知平滑 + 常速度短时预测"的组合式时序算法。

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

### 测试与验证

- 稳定手部观测时，输出是否满足契约字段要求
- 遮挡或模糊情况下，fallback 与预测是否保持输出连续性
- pinch 状态迁移是否符合确认帧和释放保护规则

### 典型工程问题：模糊帧与误检抑制

低光照、快速移动或镜头晃动时手部关键点出现明显漂移，导致 pinch 语义和对象位置随之抖动。采用三种手段共同应对：模糊度感知平滑（根据图像质量调节保守程度）、局部模板匹配（在检测失败时恢复短时观测）、短时速度预测（维持轨迹连续性）。三者分别负责判定输入质量、恢复短时观测和保持交互连贯性。

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

渲染模块显式区分不同命令类型，分别做 payload 结构检查。不同命令有不同处理入口，场景初始化、位姿更新、手部显示和交互恢复分别实现。这种"命令驱动架构"有利于测试，也方便演示时单独说明某个功能链路。

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

### 位姿命令校验

渲染模块不是盲目应用上游命令，而是先对位姿数据进行结构检查。只有当 `position` 或 `hpr` 满足契约要求时，渲染层才继续执行对象更新。这样可以显著降低跨模块联调时的错误传播——错误被限制在模块边界内而不是扩散到整个系统。

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

### 幂等消费与异常处理

- 对命令类型和 payload 键集合进行显式检查
- 对重复命令和过期命令执行安全忽略
- 对隐藏对象或格式错误命令输出运行时告警，而不是直接崩溃
- `reset_interaction` 时恢复对象状态，避免交互残留

渲染模块需要面对的主要风险不是"图形算法错误"，而是命令乱序、重复、字段缺失和运行时资源加载失败。这种设计使渲染模块可以作为稳定的命令消费端运行，而不是把所有正确性都押在上游模块上。

### 场景初始化与对象管理

`init_scene` 命令一次性下发对象描述（标识、初始位置/姿态、缩放、颜色、形状类型、是否可交互等），渲染模块收到后建立对象缓存，后续仅根据增量命令更新状态。对象定义与对象更新被清晰分开，减少了命令冗余，也方便后续扩展新模型。

当前渲染实现还支持通过模型模板工厂统一注册内置模型和自定义模型，并对材质、锚点偏移、双面渲染和懒加载做集中管理。对于课程设计原型来说，这种资源工厂模式足以支撑中小规模场景扩展。

### UI 视图系统

Rendering 模块实现了一套完整的多视图 UI 系统，所有视图均基于指尖锚点进行手势驱动交互，无需键盘鼠标：

| 视图 | 源文件 | 内容 |
| ---- | ------ | ---- |
| Home 视图 | `home_view.py` | 启动主页，含 **table**、**setting** 两个导航按钮 |
| Table 视图 | `table_overlay_view.py` | 桌面场景叠加层，含 **resume table**、**table options**、**return home** 按钮及亮度/音量两个滑块 |
| Setting 视图 | `setting_view.py` | 全局设置页，含光标缩放、光标透明度、亮度、音量四个滑块 |
| Calibration 视图 | `calibration_view.py` | 光标标定页，支持对 cursor scale x/y 和 cursor offset x/y 四项参数的键盘精调，并实时预览标定效果 |

所有可交互控件（按钮、滑块）均实现了三态视觉样式（idle / hover / pressed 或 idle / hover / active），hover 判定和 press 判定分别基于指尖中点在屏幕像素坐标系中的位置，并引入 slop 容差（press 区域比 hover 区域略宽），避免边界误触。UI 视图切换通过 `RenderView` 枚举管理（`home / table / setting / calibration`），Table 视图内还有独立的浮层枚举（`none / menu / option`）控制叠加菜单的显示。

### 支持的交互能力

系统当前支持的完整手势交互功能如下：

| 功能 | 触发方式 | 说明 |
| ---- | -------- | ---- |
| UI 导航 | 指尖悬停 + pinch | 指尖锚点进入按钮区域后高亮，稳定 pinch 触发按钮激活 |
| 滑块调节 | 指尖悬停 + pinch 拖拽 | 悬停至滑块轨道后 pinch，拖拽改变参数值，释放提交 |
| 对象悬停 | 指尖接近对象 | 进入邻域后对象进入 pending\_grab 状态，显示高亮 |
| 对象抓取 | 悬停中 pinch | 稳定 pinch 后对象跟随指尖中点移动 |
| 对象平移 | 抓取中移动 | 拖拽期间对象实时跟随，并受桌面平面约束 |
| 对象旋转 | 抓取中切换旋转模式 | 副手激活旋转模式后，主手拖拽量转化为姿态增量（HPR） |
| 虚拟手显示 | 检测到手部 | 在场景中以几何体渲染腕部、拇指、食指及捏合中心点，pinch 状态变化时颜色和尺寸实时更新 |
| 跟踪丢失恢复 | 手部离开视野 | 自动发出 reset\_interaction，对象回到安全状态 |

### 测试与验证

- `init_scene` 是否建立对象缓存并设置初始状态
- `set_object_pose` 是否正确解析 `position`、`hpr` 和 `scale`
- `set_object_state` 是否将交互状态映射为可视化状态
- `set_hand_pose` 是否只在条件满足时显示虚拟手
- `reset_interaction` 是否能够把场景恢复到安全默认值

这一层测试的价值是把"场景可显示"进一步提升为"场景可重复、可恢复"。对课程设计答辩来说，稳定性往往比单纯的视觉效果更重要，因为它直接决定演示是否容易出错。

---

## 主循环联调

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

- **何子谦**（2025080905004）：架构设计。负责系统架构设计、技术选型、功能兜底、代码审查与合并。
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
