# AeroInteract3D

[中文版本 / Chinese Version](README.zh-CN.md)

Early-stage project for real-time webcam-based 3D hand interaction.

Current implementation snapshot:

- Gesture: `GestureServiceImpl` in `src/gesture/service.py`
- Shared gesture temporal reducer: `src/gesture/temporal.py`
- Bridge: `BridgeServiceImpl` in `src/bridge/service.py`
- Rendering: `RenderingServiceImpl` in `src/rendering/service.py`
- Rendering backend: **Python + Panda3D**

The gesture runtime and live preview now share the same temporal logic for:

- tracking-state transitions
- pinch hysteresis and confirmation
- confidence shaping
- coordinate smoothing

The live preview also shows measured on-screen FPS from the actual preview loop, not just the configured target FPS.

Current runtime defaults:

- target FPS request: `30`
- requested capture resolution: `1280x640`

Developer environment setup: see [DEVELOPMENT.md](DEVELOPMENT.md).
