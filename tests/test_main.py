from __future__ import annotations

import main

from main import App, AppConfig, LIFECYCLE_RUNNING, build_config, parse_args


class FakeGestureInput:
    def __init__(self) -> None:
        self.app: App | None = None

    def start(self) -> None:
        return None

    def poll(self):
        assert self.app is not None
        self.app.request_stop()
        return None

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


class FakeBridge:
    def start(self) -> None:
        return None

    def process(self, packet):
        return []

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


class FakeRenderOutput:
    def __init__(self) -> None:
        self.step_calls = 0

    def start(self) -> None:
        return None

    def push(self, command) -> None:
        return None

    def step(self) -> None:
        self.step_calls += 1

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


def test_app_run_steps_render_output_every_loop_iteration() -> None:
    config = AppConfig(target_fps=60)
    gesture_input = FakeGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    app = App(config, gesture_input, bridge, render_output)
    gesture_input.app = app

    app.lifecycle_state = LIFECYCLE_RUNNING

    app.run()

    assert render_output.step_calls == 1


def test_parse_args_enables_live_preview_flag() -> None:
    args = parse_args(["--live-preview"])

    config = build_config(args)

    assert config.live_preview is True


def test_build_app_passes_live_preview_to_gesture_service(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    fake_gesture = object()
    fake_bridge = object()
    fake_render = object()

    class FakeGestureService:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(main, "GestureServiceImpl", FakeGestureService)
    monkeypatch.setattr(main, "BridgeServiceImpl", lambda: fake_bridge)
    monkeypatch.setattr(main, "RenderingServiceImpl", lambda: fake_render)

    app = main.build_app(AppConfig(live_preview=True))

    assert captured_kwargs["preview_enabled"] is True
    assert isinstance(app, App)
    assert app.gesture_input is not None
    assert app.bridge is fake_bridge
    assert app.render_output is fake_render