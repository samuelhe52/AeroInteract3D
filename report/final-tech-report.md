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

| 字段                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| `frame_id` / `timestamp_ms`         | 帧序号与时间戳                                               |
| `tracking_state`                    | `tracked` / `temporarily_lost` / `not_detected`              |
| `pinch_state`                       | `open` / `pinch_candidate` / `pinched` / `release_candidate` |
| `index_tip` / `thumb_tip` / `wrist` | 食指尖、拇指尖、腕部三维坐标                                 |
| `coordinate_space`                  | 坐标空间标记，下游必须据此做单位转换                         |

**`SceneCommand`**：描述渲染模块需要执行的场景更新命令。

| 字段           | 说明                                                                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| `command_id`   | 唯一命令标识                                                                                                |
| `command_type` | `init_scene` / `set_hand_pose` / `set_object_state` / `set_object_pose` / `reset_interaction` / `heartbeat` |
| `object_id`    | 命令目标对象 ID                                                                                             |
| `payload`      | 命令附带数据，结构随 `command_type` 不同而变化                                                              |

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
| `init_scene`        | 初始化场景                                                        |
| `set_hand_pose`     | 同步虚拟手姿态                                                    |
| `set_object_state`  | 切换对象交互状态（`idle`、`pending_grab`、`grabbed`、`rotating`） |
| `set_object_pose`   | 更新对象的位置、姿态或缩放状态                                    |
| `reset_interaction` | 输入无效或 Gesture 模块异常情况下重置物体状态                     |

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

### 双手缩放

当主手和副手同时处于 `pinched` 状态且均悬停在同一对象上时，Bridge 层基于两只手的捏合锚点间距变化，计算缩放比例，并通过 `set_object_pose` 命令更新对象的 `scale` 字段，实现跟手的缩放效果。缩放操作期间，对象平移被暂停，以避免位置和尺寸命令的冲突。

### 典型工程问题

**旋转和平移冲突**：抓取、旋转和平移可能被同一帧输入同时触发，导致对象既被拖动又被旋转的冲突行为。Bridge 层通过设置不同模式，将两类交互分开处理，旋转模式只输出姿态命令，平移模式只输出位置命令。

**过期命令与状态残留**：手已离开视野但对象仍停留在抓取状态，或上一帧命令延迟到达后覆盖当前状态。渲染端通过命令去重、过期帧忽略和 `reset_interaction` 恢复机制，把这类问题控制在安全范围内。

## Gesture 模块

### 模块职责

Gesture 模块是系统的输入端，负责把摄像头视频帧转化为稳定的手势状态输出。其核心职责包括：

- 从摄像头采集视频帧
- 调用 MediaPipe Hand Landmarker 检测手部关键点
- 在短时检测失败时预测运动轨迹作为 fallback，保证输出稳定性、连续性
- 根据契约，生成稳定的 `GesturePacket`
- 为调试视图提供原始相机帧、检测到的指标数据

### 核心算法链路

```{=latex}
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  >=Stealth,
  every node/.style={font=\small},
  pipeblock/.style={
    draw, rounded corners=3pt,
    minimum width=3.0cm, minimum height=1.0cm,
    text width=2.7cm, align=center, inner sep=4pt,
    fill=teal!20, draw=teal!70!black
  },
  fallblock/.style={
    draw, dashed, rounded corners=3pt,
    minimum width=3.0cm, minimum height=1.0cm,
    text width=2.7cm, align=center, inner sep=4pt,
    fill=teal!10, draw=teal!50!black
  },
  outputblock/.style={
    draw, rounded corners=3pt,
    minimum width=3.0cm, minimum height=1.0cm,
    text width=2.7cm, align=center, inner sep=4pt,
    fill=orange!20, draw=orange!70!black
  },
  lbl/.style={font=\tiny\itshape},
  arr/.style={->, thick},
]

%% ── Row 1: 检测链路 ─────────────────────────────────────
\node[pipeblock] (preproc)  at (0,   0) {图像预处理\\（灰度化·缩放）};
\node[pipeblock] (blur)     at (3.8,  0) {模糊度估计};
\node[pipeblock] (detect)   at (7.6,  0) {手部检测\\（MediaPipe）};

%% ── 检测成功/失败分支 ────────────────────────────────────
\node[pipeblock]  (lmnorm)   at (4.8,  -2.2) {关键点提取\\与尺度归一化};
\node[fallblock]  (fallback) at (10.4, -2.2) {回退逻辑\\（根据过往数据预测）};

%% ── Row 2: 时序归约（TemporalReducer）────────────────────
\node[pipeblock]   (smooth)  at (0,    -4.8) {坐标平滑};
\node[pipeblock]   (pscore)  at (3.8,  -4.8) {pinch 手势评分};
\node[pipeblock]   (pfsm)    at (7.6,  -4.8) {pinch\\状态机};
\node[outputblock] (output)  at (11.4, -4.8) {稳定输出\\GesturePacket};

%% ── Row 1 箭头 ──────────────────────────────────────────
\draw[arr] (preproc) -- (blur);
\draw[arr] (blur)    -- (detect);

%% ── 分支箭头（detect → 成功/失败路径）──────────────────
\draw[arr] (detect.south) -- ++(0,-0.4) -| (lmnorm.north);
\draw[arr] (detect.south) -- ++(0,-0.4) -| (fallback.north);

%% 分支标注
\node[lbl, above] at (5.8, -0.9) {成功};
\node[lbl, above] at (9.4, -0.9) {失败};

%% ── 两路汇聚至时序归约入口 ──────────────────────────────
\coordinate (merge) at (7.6, -3.6);
\draw[thick] (lmnorm.south)   -- ++(0,-0.55) -| (merge);
\draw[thick] (fallback.south) -- ++(0,-0.55) -| (merge);
\draw[arr]   (merge) -- ++(0,-0.2) -| (smooth.north);

%% ── Row 2 箭头 ──────────────────────────────────────────
\draw[arr] (smooth) -- (pscore);
\draw[arr] (pscore) -- (pfsm);
\draw[arr] (pfsm)   -- (output);

\end{tikzpicture}
\caption{手势模块核心算法链路}
\label{fig:gesture-pipeline}
\end{figure}
```

本项目的手势检测基于 MediaPipe Hand Landmarker Task API。项目工作重点在于其上的工程集成与鲁棒性算法设计，力图将不稳定的视觉观测转化为稳定的交互状态输出。核心算法链路如图 \ref{fig:gesture-pipeline} 所示。

### 输入预处理与模糊估计

系统从彩色图像中提取灰度图，用 Laplacian 方差衡量清晰度 `blur_level`（方差越低图像越模糊），再将输入图像进行缩放，控制负载。

`blur_level` 贯穿后续多个环节，以此实现自适应的稳定性调整。由于坐标稳定算法对系统端到端延迟有一定影响，系统需要在保证稳定性的同时尽量降低响应时间。`blur_level` 的引入使得系统能够在模糊帧时自动增强坐标平滑成都（通过动态调整参数实现），而在清晰帧时保持较低的平滑强度以保证响应速度。

```python
frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
blur_level = self._estimate_blur_level(frame_gray)
detect_frame = resize_for_detection(frame_bgr, max_side=GESTURE_DETECT_MAX_SIDE)
rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
result = self._landmarker.detect_for_video(image, timestamp_ms)

def _estimate_blur_level(self, frame_gray: np.ndarray) -> float:
    lap_var = float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
    # Higher blur means lower Laplacian variance, then invert to [0, 1].
    return _clamp(1.0 - (lap_var / (lap_var + 200.0)))
```

### 关键点检测和深度估算

MediaPipe Hand Landmarker 输出 21 个手部关键点，系统重点使用 `wrist`、`thumb_tip` 和 `index_tip` 作为判断 pinch 状态的依据；同时也采集其他几个手指尖的坐标以判断是否进入旋转模式。

由于单目摄像头无法直接获取深度，z 轴坐标只能从画面中估算。系统采用两路信号融合的方式进行深度估计：

- **手掌尺度**（`hand_scale`）：取所有关键点 x/y 坐标的最大跨度。手在画面中占比大说明离摄像头近，占比小说明离摄像头远，基于此可以推算一个全局的深度估计。
- **MediaPipe 局部 z 值**：MediaPipe 本身会输出各关键点的相对 z 坐标（以手掌根部为基准），可作为局部深度参考，但精度有限。

```python
def estimate_hand_depth(landmarks: list[Vec3], hand_scale: float) -> float:
    scale_weight = _normalized_scale(hand_scale)          # 全局：手掌大小推算距离
    local_depth = _clamp((-sum(lm.z for lm in landmarks)  # 局部：MediaPipe z 均值
                          / max(len(landmarks), 1)) / 0.25)
    blended = ((1.0 - DEPTH_ESTIMATION_LOCAL_Z_WEIGHT) * scale_weight
               + DEPTH_ESTIMATION_LOCAL_Z_WEIGHT * local_depth)
    return (2.0 * _clamp(blended)) - 1.0
```

手尺度归一化同样是 pinch 判定的前提：手离摄像头远近会显著改变原始图像空间中的指尖间距，因此必须根据手掌尺度进行距离归一化，保证 pinch 判定范围不受深度变化影响。

### 手部跟踪丢失时的 fallback 策略

当 MediaPipe 当帧未检测到手部时，系统不会立刻将跟踪状态置为丢失，而是通过两级 fallback 机制维持短时间内的位置预测和回退匹配，保证输出的连续性和稳定性：

- **第一级策略是记录检测点速度**，并合理进行坐标外推（extrapolation）。每帧检测成功时，系统记录腕部坐标点在图像像素坐标系中的位移和帧间时间差，推算速度向量。检测失败时，用上一帧速度乘以帧间时间估算腕部新像素坐标，再将整组关键点（食指尖、拇指尖等）按同样位移平移。这种方法在短时检测失败时能够保持位置的连续性和稳定性，但如果持续时间过长，预测误差会逐渐积累。
- **第二级策略是使用 OpenCV 中的经典模板匹配算法**（`cv2.matchTemplate`）在检测失败时进行局部搜索。具体原理是，在检测成功时，系统在腕部周围裁取一块 40×40 像素的灰度 patch 保存；检测失败时，以上一级策略中速度预测的新腕部坐标为中心，在半径 28 像素的搜索框内用 `cv2.matchTemplate` 算法做滑动窗口搜索，找到与存档 patch 相关系数最高的位置作为新腕部坐标。匹配分数归一化后作为 `appearance_match_score` 输出，供后续 pinch 判定使用。

先尝试模板匹配，若 `cv2.matchTemplate` 返回的最大相关系数（经 `(max_val + 1) * 0.5` 归一化到 [0, 1]）超过 0.45，则采用匹配位置（`source = "fallback"`）；否则退回速度预测（`source = "predicted"`）。连续 fallback 超过 2 帧后不再接受纯速度预测，超过 8 帧后整个链路终止，输出切换为 `not_detected`。

```python
if not result.hand_landmarks:
    fallback = self._detect_fallback(frame_bgr, frame_gray, timestamp_ms, blur_level)
    return [] if fallback is None else [fallback]
```

### 概率式 pinch 判定

直接用指尖坐标距离阈值判定 pinch 的问题在于：手指距离在阈值附近轻微抖动时，状态会反复翻转，导致对象频繁"抓住又松开"。为了解决这一问题，系统改用软概率评分：用高斯似然把指尖距离映射成连续的 `pinch_score`（代表 0-1 之间的概率），再结合先验状态（当前是否已处于稳定 pinch）和外观匹配分数三路加权，得到一个平滑变化的置信值，减少抖动带来的 pinch 状态不稳定问题。

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

更进一步的，为了对动态模糊强烈、画面质量差的情况做进一步优化，我们引入 `PINCH_CONFIRM_FRAMES` 参数，要求 pinch 状态必须连续数帧满足条件，超过一定帧数后才真正进入 `pinched` 状态。这层保险机制在一定程度上牺牲了 pinch 状态的响应速度，但大幅提升了在不稳定输入条件下的交互体验，避免了频繁误触和状态抖动。

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

### 旋转模式与双手缩放

在单目视觉输入下，直接实现旋转交互虽然能够使交互体验更加自然，但由于旋转操作对输入稳定性要求更高，且考虑到在手背朝摄像头时的跟踪丢失问题，本项目并没有通过“捏合并旋转手部”实现旋转，而是设计了一个独立的"旋转模式"。
旋转模式由“五指捏合”这一特定手势触发；在该模式下，捏合点的拖拽位移不再驱动对象位置移动，而是经由 Bridge 层转化为物体姿态参数，设置物体旋转模式。

注：对于缩放手势的实现，本项目选择不在 Gesture 层进行特殊处理；Gesture 模块只负责同时输出主手和副手两路 `GesturePacket`。当两只手都处于 `pinched` 状态时，Bridge 层检测双手间距变化并将其转化为缩放命令。

### 稳定策略总结

上述各环节共同构成分层的时序稳定机制：

- **基础坐标平滑**：对 x/y/z 三个方向做指数平滑，抑制逐帧抖动
- **动态调整平滑度**：根据 `blur_level` 动态调整平滑强度，模糊帧优先保证交互质量，清晰帧响应快
- **短时跟踪丢失 fallback**：检测中断时用上一帧速度外推位置，按缺失帧数逐步衰减
- **pinch 状态机**：pinch 四态状态机，进入和退出有独立确认条件，消除状态抖动

## Rendering 模块

### 模块职责

Rendering 模块是系统的输出端，负责把 Bridge 模块输出的 `SceneCommand` 转化为 Panda3D 场景更新。其核心职责包括：

- 初始化 Panda3D 窗口、相机和基础光照
- 创建场景对象并维护其视觉状态（悬停/抓取/旋转分别对应不同颜色）
- 按顺序消费来自 Bridge 的 `SceneCommand`，更新场景状态
- 实现手部可视化，提供视觉反馈
- 提供调试数据面板、摄像头预览
- 实现基本 UI 交互元素，验证可行性

### 命令消费与场景管理

渲染模块显式区分命令类型，每类命令有独立处理入口，并对 payload 做结构检查——只有字段满足契约要求时才执行更新，否则记录错误并跳过。这种设计有效阻断了上游错误的传播，也使各命令路径可独立测试。

场景对象由 `init_scene` 命令一次性建立缓存（含标识、初始位置/姿态、缩放、颜色、形状类型等），后续仅消费增量命令更新状态，对象定义与状态更新清晰分离。自定义模型与内置几何体通过同一套模型工厂注册，对材质、锚点偏移、双面渲染和懒加载做集中管理。

### UI 视图系统

Rendering 模块实现了一套完整的多视图 UI 系统，所有视图均基于指尖锚点进行手势驱动交互，无需键盘鼠标：

| 视图             | 源文件                  | 内容                                                                                            |
| ---------------- | ----------------------- | ----------------------------------------------------------------------------------------------- |
| Home 视图        | `home_view.py`          | 启动主页，含 table、setting 两个导航按钮                                                        |
| Table 视图       | `table_overlay_view.py` | 桌面场景叠加层，含 resume table、table options、return home 按钮及亮度/音量两个滑块             |
| Setting 视图     | `setting_view.py`       | 全局设置页，含光标缩放、光标透明度、亮度、音量四个滑块                                          |
| Calibration 视图 | `calibration_view.py`   | 光标标定页，支持对 cursor scale x/y 和 cursor offset x/y 四项参数的键盘精调，并实时预览标定效果 |

其中，`table` 代表桌面的真实交互场景；所有可交互控件（按钮、滑块）均实现了三态视觉样式（idle / hover / pressed 或 idle / hover / active）提供视觉反馈。

### 支持的交互能力

系统当前支持的完整手势交互功能如下：

| 功能       | 触发方式              | 说明                                                |
| ---------- | --------------------- | --------------------------------------------------- |
| 按钮       | 双指捏合              | 指尖锚点进入按钮区域后高亮，稳定 pinch 触发按钮激活 |
| 滑块调节   | 指尖悬停 + pinch 拖拽 | 悬停至滑块轨道后 pinch，拖拽改变参数值，释放提交    |
| 对象悬停   | 指尖接近对象          | 进入邻域后对象进入 pending_grab 状态，显示高亮      |
| 对象抓取   | 悬停中 pinch          | 稳定 pinch 后对象跟随两指尖中点移动                 |
| 对象平移   | 抓取中移动            | 拖拽期间对象实时跟随，并受桌面平面约束              |
| 对象旋转   | 抓取中切换旋转模式    | 激活旋转模式后，捏合拖拽转化为对对象姿态的控制      |
| 虚拟手显示 | 检测到手部            | 在场景中渲染虚拟手，实现视觉反馈                    |

### 自定义模型导入

**此部分后续根据具体实现完善**

渲染模块支持在运行时导入外部 3D 模型文件，作为可交互场景对象使用。用户可以将自定义模型放置于指定目录，系统会在初始化时自动发现并注册，与内置几何体一样参与场景管理和手势交互。这使得系统不局限于固定的演示场景，具备面向不同应用场景的扩展能力。

## App 主循环

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

可以看到，主循环中不断调用 `render_output.step()` 来驱动渲染更新；通过 `gesture_input.poll()` 获取最新的手势数据包；把数据包传给 Bridge 处理成场景命令；再把命令逐条推送给渲染模块。

项目运行时，支持通过 YAML 配置文件设置默认参数（摄像头编号、分辨率、画面是否镜像、灵敏度等）。

# 性能评价

本系统在 1080p 分辨率下的帧率能够稳定在 **25 FPS**；关闭渲染模块的调试视图后，帧率可以提升至 **30 FPS**，用户体验较为流畅。对于手部检测稳定性，在光线较为充足的环境下，系统极少出现跟踪丢失或跳变的情况；在光线较弱或手部快速移动时，如果叠加以清晰度/帧率较低的摄像头输入，检测稳定性会有所下降，但由于引入了 fallback 预测机制，整体上能够以延迟小幅增加、灵敏度小幅下降的代价保持交互的连续性和可用性。

整体来说，系统流畅度与延迟较低，能够做到实时的手势交互反馈；在多数常见使用场景下，手势检测的稳定性和准确性能够满足基本的交互需求。

# 系统展示

## 已完成的 UI 页面

本次展示中，桌面 UI 的三张截图已经整理完成并复制到 `assets/`。它们分别对应主页、设置页和光标标定页，可以直接用于报告排版。

```{=latex}
\begin{figure}[htbp]
\centering
\noindent\makebox[\textwidth][c]{%
\begin{minipage}[t]{0.40\textwidth}
\centering
\includegraphics[width=\linewidth]{assets/system-ui-home.jpg}
\par\vspace{0.35em}
{\small \textbf{主页}}
\par{\footnotesize table / setting 两个入口}
\end{minipage}
\hspace{0.05\textwidth}%
\begin{minipage}[t]{0.40\textwidth}
\centering
\includegraphics[width=\linewidth]{assets/system-ui-settings.jpg}
\par\vspace{0.35em}
{\small \textbf{设置页}}
\par{\footnotesize 全局参数与预览面板}
\end{minipage}
}
\par\vspace{0.95em}
\noindent\makebox[\textwidth][c]{%
\begin{minipage}[t]{0.40\textwidth}
\centering
\includegraphics[width=\linewidth]{assets/system-ui-calibration.jpg}
\par\vspace{0.35em}
{\small \textbf{光标标定页}}
\par{\footnotesize scale / offset 调整}
\end{minipage}
}
\caption{当前可直接展示的 UI 页面}
\label{fig:system-ui-showcase}
\end{figure}
```

## 桌面场景占位

实际桌面 table 场景仍在迭代中，因此这里先按最终演示顺序预留占位。建议补拍的最小闭环是 `初始态 → 悬停高亮 → 抓取拖拽 → 旋转 → 双手缩放`，这样可以完整覆盖报告中已经实现的对象交互主链路。

```{=latex}
\begin{figure}[htbp]
\centering
\noindent\makebox[\textwidth][c]{%
\begin{minipage}[t]{0.42\textwidth}
\centering
\fcolorbox{gray!55}{gray!10}{%
\begin{minipage}[c][3.35cm][c]{0.98\linewidth}
\centering
{\large \textbf{桌面场景}}\\[0.25em]
{\footnotesize 待补拍}
\end{minipage}}
\par\vspace{0.35em}
{\small \textbf{初始态}}
\end{minipage}
\hspace{0.03\textwidth}%
\begin{minipage}[t]{0.42\textwidth}
\centering
\fcolorbox{gray!55}{gray!10}{%
\begin{minipage}[c][3.35cm][c]{0.98\linewidth}
\centering
{\large \textbf{桌面场景}}\\[0.25em]
{\footnotesize 待补拍}
\end{minipage}}
\par\vspace{0.35em}
{\small \textbf{悬停高亮}}
\end{minipage}
}

\vspace{0.55em}

\noindent\makebox[\textwidth][c]{%
\begin{minipage}[t]{0.42\textwidth}
\centering
\fcolorbox{gray!55}{gray!10}{%
\begin{minipage}[c][3.35cm][c]{0.98\linewidth}
\centering
{\large \textbf{桌面场景}}\\[0.25em]
{\footnotesize 待补拍}
\end{minipage}}
\par\vspace{0.35em}
{\small \textbf{抓取拖拽}}
\end{minipage}
\hspace{0.03\textwidth}%
\begin{minipage}[t]{0.42\textwidth}
\centering
\fcolorbox{gray!55}{gray!10}{%
\begin{minipage}[c][3.35cm][c]{0.98\linewidth}
\centering
{\large \textbf{桌面场景}}\\[0.25em]
{\footnotesize 待补拍}
\end{minipage}}
\par\vspace{0.35em}
{\small \textbf{旋转}}
\end{minipage}
}

\vspace{0.55em}

\noindent\makebox[\textwidth][c]{%
\begin{minipage}[t]{0.42\textwidth}
\centering
\fcolorbox{gray!55}{gray!10}{%
\begin{minipage}[c][3.35cm][c]{0.98\linewidth}
\centering
{\large \textbf{桌面场景}}\\[0.25em]
{\footnotesize 待补拍}
\end{minipage}}
\par\vspace{0.35em}
{\small \textbf{双手缩放}}
\end{minipage}
}
\caption{桌面场景演示占位，后续按同序补拍后替换}
\label{fig:system-scene-placeholder}
\end{figure}
```

如果后续还想额外强调虚拟手或调试信息，可以再补一张叠加视图，但它不属于当前最小必需集。

# 总结

本项目完成了 MediaPipe 手势检测、坐标稳定、深度估算 Panda3D 场景渲染的完整系统链路。系统以共享契约划定模块边界，Gesture、Bridge、Rendering 三层可独立开发和测试。Gesture 层通过模糊感知平滑、fallback 预测和概率式 pinch 状态机，将不稳定的视觉观测转化为可靠的交互信号；Bridge 层处理坐标映射、对象状态管理和旋转/缩放语义转换；Rendering 层实现了抓取、平移、旋转、缩放的完整三维交互场景，以及多视图手势驱动 UI 和自定义模型导入能力。

团队在以下方面有显著收获：

- **算法层面**：认识到视觉交互系统的难点不只是调用现成检测器，更在于把不稳定的观测变成稳定的交互语义。模糊感知平滑、pinch 概率评分、短时预测和重捕获回接等实践，让团队对实时 CV 系统中的鲁棒性问题有了更直接的理解。
- **软件工程层面**：经历了契约设计、模块解耦、接口联调、异常处理和测试补齐等过程。系统级实现训练了成员对架构边界、状态管理和可维护性的判断能力。
- **协作层面**：体会到"统一数据契约"和"明确模块职责"的重要性。Gesture、Bridge 和 Rendering 之所以能够逐步收敛并联调成功，本质上依赖于前期对职责边界的约束，而非临时性的代码互相适配。

不足之处与改进方向：

- 场景完善性：当前系统的交互场景较为基础，未来可以允许用户自定义交互对象和场景布局，完善交互的视觉反馈和动画效果。
- 算法优化：虽然当前的稳定性策略在多数情况下表现良好，但在极端模糊或快速运动时仍有改进空间。

# 项目成员与分工

- **何子谦**（2025080905004）：架构设计。负责系统架构设计、技术选型、功能兜底、代码审查与合并。
- **徐逸博**（2025080905025）：核心算法。负责手势检测链路实现、稳定坐标相关算法实现、结题报告 PPT。
- **陈金龙**（2025080905001）：交互与 3D。负责 Panda3D 场景渲染实现、三维物体控制逻辑开发。
- **夏凡程**（2025080905020）：开题报告撰写、周报 PPT 制作。

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

本项目开源在 [Github 仓库](https://github.com/samuelhe52/AeroInteract3D)，包含完整的代码实现。

## 测试环境

系统在以下多个平台上完成了功能验证：

| 设备           | CPU                     | 内存  | 操作系统   |
| -------------- | ----------------------- | ----- | ---------- |
| MacBook Pro    | Apple M1 Pro            | 16 GB | macOS      |
| Windows 笔记本 | Intel Core Ultra 5 225H | 32 GB | Windows 11 |
| Windows 笔记本 | Intel Core i7-14650HX   | 32 GB | Windows 11 |

摄像头输入均使用电脑自带摄像头，分辨率为 720p 或 1080p。软件环境：Python 3.12、MediaPipe ≥ 0.10.32、OpenCV ≥ 4.13.0、Panda3D 1.11.0-dev。
