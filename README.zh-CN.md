# AeroInteract3D

[English Version](README.md)

这是一个基于摄像头手势输入的实时 3D 交互原型项目。

当前实现快照：

- Gesture：`src/gesture/service.py` 中的 `GestureServiceImpl`
- 共享手势时序归约器：`src/gesture/temporal.py`
- Bridge：`src/bridge/service.py` 中的 `BridgeServiceImpl`
- Rendering：`src/rendering/service.py` 中的 `RenderingServiceImpl`
- 渲染后端：**Python + Panda3D**

Gesture 实时服务与 live preview 现在共用同一套时序逻辑，包括：

- tracking 状态切换
- pinch 滞回与确认
- confidence 修正
- 坐标平滑

live preview 当前显示的是预览主循环实际测得的 FPS，而不是仅显示目标 FPS。

当前运行默认值：

- 目标 FPS 请求：`30`
- 请求采集分辨率：`1280x640`

开发环境配置请参考：[DEVELOPMENT.md](DEVELOPMENT.md)。
