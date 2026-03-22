# Gesture 模块技术报告

## 1. 报告范围

本报告基于当前仓库中的 `gesture` 模块实现进行静态阅读和测试验证，覆盖以下代码与接口：

- `src/gesture/service.py`
- `src/gesture/runtime.py`
- `src/gesture/temporal.py`
- `src/gesture/debug/live_preview.py`
- `src/gesture/debug/live_preview_runtime.py`
- `src/contracts.py`
- `src/utils/contracts.py`
- `main.py`
- `tests/test_gesture_service.py`
- `tests/test_gesture_runtime.py`
- `tests/test_gesture_temporal.py`

目标是说明该模块“现在是如何工作的”，而不是给出未来方案设计。

## 2. 模块定位

`gesture` 模块是系统中的手势输入端，职责是：

1. 从摄像头采集帧。
2. 基于 MediaPipe Hand Landmarker 识别单手关键点。
3. 在检测短时失败时做有限度的 fallback / prediction。
4. 将原始观测送入时序归约器，输出稳定的 `GesturePacket`。
5. 向 Bridge 提供契约一致、时序有序、可降级的输入流。

它对外的正式端口是 `GestureInputPort`，当前实现为 `GestureServiceImpl`。

## 3. 整体架构

当前实现可以分成三层：

### 3.1 服务层

`src/gesture/service.py` 中的 `GestureServiceImpl` 负责生命周期、配置、错误收集、指标统计、对外 `poll()` 输出，以及可选 preview 渲染。

### 3.2 运行时检测层

`src/gesture/runtime.py` 中的 `CaptureRuntime` 和 `HandLandmarkerRuntime` 分别负责：

- 摄像头打开、读帧、镜像翻转。
- MediaPipe 模型加载与逐帧检测。
- 基于上一帧 wrist 模板和运动速度做短时 fallback。

这一层输出的是 `RawHandObservation`，它仍然是“观测值”，还不是最终契约包。

### 3.3 时序归约层

`src/gesture/temporal.py` 中的 `TemporalReducer` 是当前 gesture 模块的核心。它负责：

- 平滑坐标；
- 维护 pinch 状态机；
- 维护 tracking 状态机；
- 在缺帧时做位置预测和信心衰减；
- 生成最终 `GesturePacket`。

从实现上看，真正决定“交互稳定性”的逻辑主要都在这一层。

## 4. 数据流

当前一帧 gesture 数据的主路径如下：

1. `GestureServiceImpl.poll()` 递增 `frame_id`，生成单调 `timestamp_ms`。
2. `CaptureRuntime.read()` 读取摄像头帧，并做水平翻转。
3. `HandLandmarkerRuntime.detect()`：
   - 转灰度图，用于 blur 估计和 fallback；
   - 缩放图像送入 MediaPipe；
   - 若检测成功，提取 21 个 landmarks，并计算 `hand_scale`、depth hint、`raw_pinch_distance`；
   - 若检测失败，则尝试 template match / 速度预测生成 fallback observation。
4. `GestureServiceImpl` 将 observation 的运行时提示信息整理成 `runtime_hint`。
5. `TemporalReducer.reduce()` 结合 observation 和 hint 输出 `GesturePacket`。
6. `GestureServiceImpl` 对结果做契约校验、指标记录、健康状态更新和可选 preview 渲染。
7. 主应用在 `main.py` 中把 packet 交给 bridge，并把相机帧/观测数据同步给 rendering 调试视图。

## 5. 对外契约

当前 gesture 输出严格依赖共享契约 `src/contracts.py` 中的 `GesturePacket`。关键字段包括：

- `frame_id`
- `timestamp_ms`
- `hand_id`
- `tracking_state`
- `confidence`
- `pinch_state`
- `index_tip`
- `thumb_tip`
- `wrist`
- `coordinate_space`

模块内部还通过 `validate_gesture_packet()` 做一次运行时自检，确保输出至少满足：

- `contract_version` 正确；
- `frame_id`、`timestamp_ms` 非负；
- `confidence` 在 `[0, 1]`；
- `hand_id` 非空；
- `coordinate_space == "camera_norm"`；
- 三个关键点向量分量为数值。

这意味着 gesture 模块不仅产生数据，也承担了第一层契约守门职责。

## 6. 运行时检测设计

### 6.1 摄像头采集

`CaptureRuntime` 封装较薄，主要行为是：

- 用 OpenCV 打开摄像头；
- 设置期望分辨率和 FPS；
- 每帧做水平镜像。

镜像翻转很关键，因为这会影响最终交互方向，使预览更符合“镜像自拍”直觉。

### 6.2 MediaPipe 手部检测

`HandLandmarkerRuntime` 使用 MediaPipe Tasks 的 `VIDEO` 模式，固定 `num_hands=1`，符合项目 MVP 的单手约束。

检测成功后，它会做三件事：

1. 将 landmark 从 MediaPipe 归一化图像坐标转换到 `camera_norm`。
2. 用关键点包围盒估计 `hand_scale`。
3. 结合整体手尺度和局部 z 值估计深度。

这里没有直接输出 MediaPipe 原始坐标，而是提前转换成系统统一的摄像头归一化坐标，这有利于下游模块稳定消费。

### 6.3 检测失败时的 fallback

当 MediaPipe 当帧没有结果时，runtime 不会立刻返回 `None`，而是尝试两级补偿：

- 一级：围绕预测 wrist 位置做局部模板匹配。
- 二级：若模板匹配质量不够，则短时只用速度外推。

fallback 不是重新构造完整 landmarks，而是把上一帧 `index_tip`、`thumb_tip`、`wrist` 整体平移。这个策略的优点是常数时间、低延迟；缺点是无法表达手型变化，因此它更像“短时轨迹延续”，不是重新识别。

## 7. 时序归约器设计

### 7.1 Motion preset

`TemporalReducer` 支持 `high`、`medium`、`low` 三档 motion preset，本质上是对以下参数集的切换：

- XY / Z 平滑强度；
- deadzone；
- 丢失跟踪时的预测混合比例；
- 预测超前量；
- 丢失期间的速度阻尼。

从参数值看：

- `high` 更偏响应；
- `low` 更偏平滑；
- `medium` 是默认档。

### 7.2 pinch 状态机

pinch 并不是简单地按距离阈值二分，而是一个带确认帧数的状态机：

- `open`
- `pinch_candidate`
- `pinched`
- `release_candidate`

核心机制：

- 进入 pinch 需要连续多帧超过进入阈值；
- 已经 pinched 后，release 也需要独立确认；
- `aggressive_release_guard` 打开后，release 的质量要求更高、确认帧数更多。

这套机制明显是为了压制“刚捏住就抖开”或“快速运动造成误放手”的问题。

### 7.3 tracking 状态机

tracking 有三态：

- `tracked`
- `temporarily_lost`
- `not_detected`

状态转换逻辑不是完全由 detector 决定，而是 detector 与 temporal 共同决定：

- 观测正常时通常为 `tracked`；
- fallback / predicted 观测如果 pinch 不稳定，则降为 `temporarily_lost`；
- 连续缺失超过窗口后进入 `not_detected`；
- 如果手势已经稳定 pinched，短时缺失期间仍可能保持 `tracked`，以支撑抓取连续性。

这一点很重要。当前实现显式把“交互连续性”优先级放在“严格视觉真实性”之前。

### 7.4 坐标平滑

平滑逻辑有几个鲜明特点：

- `x/y/z` 采用不同响应策略；
- `y` 轴在高 blur 或低质量下仍尽量保持一定响应；
- 当出现较大横向跳变时，会对 `y` 使用保守混合，抑制横扫造成的垂向误差；
- 重捕获时会把预测位置与新检测位置做 blend，并限制单帧回跳幅度。

因此当前平滑不是简单 EMA，而是“质量感知 + 方向差异化 + 重捕获保护”的组合。

## 8. 输出包的语义特点

当前 `GesturePacket` 除契约必填字段外，还会带上较丰富的调试元数据：

- `pinch_distance`
- `velocity`
- `smoothing_hint`
- `debug`

其中 `debug` 内已经包含：

- `pinch_score`
- `appearance_match_score`
- `predicted_tracked`
- `blur_level`
- `missing_frames`
- `reacquire_blend_progress`
- `detector_source`
- `handedness`

这说明 gesture 模块现在不仅能给 bridge 供数，也在为调试和参数调优提供可观测性。

## 9. 与主应用的集成方式

在正式应用主循环中：

- `GestureServiceImpl` 作为 `gesture_input` 被轮询；
- 独立 OpenCV preview 默认关闭；
- gesture 模块会通过 `get_camera_data()` 把原始相机帧和当前 observation 交给 rendering；
- rendering 再负责窗口内摄像头预览和统计面板。

因此当前 gesture 模块实际上支持两种调试路径：

1. 独立的 `src/gesture/debug/live_preview.py`。
2. 主应用内嵌的渲染侧摄像头预览。

## 10. 测试现状

本次实际执行通过的测试：

- `uv run pytest -q tests/test_gesture_service.py tests/test_gesture_runtime.py tests/test_gesture_temporal.py`
- `uv run pytest -q tests/test_main.py -k 'motion_preset or aggressive_release_guard or build_app_disables_gesture_preview'`

结果：

- gesture 相关测试 `16 passed`
- main 相关集成测试 `3 passed`

现有测试主要覆盖：

- service 的生命周期、降级行为、preview 调用、health 字段；
- runtime 的坐标转换和灰度图复用；
- temporal 的 pinch/release 确认、丢失预测、纵向响应、motion preset；
- main 对 gesture 参数的透传。

## 11. 当前实现的优点

### 11.1 分层清晰

采集、检测、时序归约、服务编排四类职责总体分开，便于定位问题。

### 11.2 降级路径完整

从 detector 启动失败、读帧失败，到单帧检测失败，都有明确的降级行为，不是直接崩溃。

### 11.3 对交互连续性有明确设计

通过 fallback、prediction、grace frames、reacquire blend 等机制，当前实现明显围绕“抓取不中断”来优化。

### 11.4 可观测性较强

`health()`、`metrics`、`debug`、`smoothing_hint` 都比较完整，调试成本不高。

### 11.5 测试覆盖到关键状态机

最核心的 temporal 行为已被测试保护，这是当前模块最有价值的保障。

## 12. 当前风险与不足

### 12.1 文档与代码存在默认值漂移

`src/gesture/README.zh-CN.md` 仍写着目标 FPS 默认值为 `60`，但代码中的 `DEFAULT_TARGET_FPS` 已是 `30`。这类文档漂移会误导调参和联调。

### 12.2 常量存在重复来源风险

`runtime` 内部将 `_fallback_max_frames` 直接写成 `8`，而常量文件中也有 `FALLBACK_MAX_FRAMES = 8`。目前数值一致，但来源分裂，后续调参容易不一致。

### 12.3 fallback 只能平移关键点，不能表达姿态变化

当手快速旋转或 pinch 形状真实发生变化时，当前 fallback 只会平移上一帧关键点，因此：

- 位置连续性较好；
- 手型真实性较弱；
- pinch 状态更多依赖历史和质量门控，而不是新的几何证据。

### 12.4 service 带有少量历史调试接口残留

`get_last_frame()` / `get_preview_frame()` 是 best-effort 辅助接口，但默认 `CaptureRuntime` 和 `GesturePreviewWindow` 并未提供对应实现，说明这部分更像历史兼容接口，不在主路径上。

### 12.5 测试仍偏单元级

当前测试没有覆盖真实摄像头、真实 MediaPipe 模型加载、模板匹配 fallback 的端到端行为，因此“线上稳不稳”仍主要依赖实机验证。

## 13. 综合结论

就当前代码状态看，gesture 模块已经不是“纯检测器”，而是一个带时序理解和降级恢复能力的实时交互输入服务。

它的核心价值不在于识别了多少手势，而在于围绕单手 pinch 交互构建了一套相对完整的实时稳定化链路：

- 检测失败可短时补偿；
- pinch 有确认和释放保护；
- tracking 允许短暂失真但维持交互连续；
- 输出包契约稳定且带调试信息；
- 主应用与 debug preview 共用同一套时序语义。

如果后续继续演进，这个模块最值得保护的资产不是 MediaPipe 接入本身，而是 `TemporalReducer` 这套围绕交互稳定性形成的状态机和质量门控逻辑。

## 14. 建议的后续关注点

建议后续优先关注以下三类事项：

1. 清理文档与常量漂移，确保调参基线只有一个来源。
2. 补真实 fallback / reacquire 场景下的回归测试，尤其是模板匹配与快速运动。
3. 继续明确 gesture 与 rendering 的调试职责边界，避免服务层继续吸收过多展示侧接口。
