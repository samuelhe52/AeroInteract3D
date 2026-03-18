from __future__ import annotations

import main

from main import App, AppConfig, LIFECYCLE_RUNNING, build_config, parse_args
from src.gesture.constants import DEFAULT_TARGET_FPS


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


def test_build_config_uses_default_target_fps() -> None:
    config = build_config(parse_args([]))

    assert config.target_fps == DEFAULT_TARGET_FPS
    assert config.motion_preset == "medium"
    assert config.aggressive_release_guard is False


def test_parse_args_disables_live_preview_by_default() -> None:
    args = parse_args([])

    config = build_config(args)

    assert config.live_preview is False


def test_parse_args_uses_run_config_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_config = tmp_path / ".run.yaml"
    run_config.write_text(
        "\n".join(
            [
                "camera_index: 2",
                "target_fps: 55",
                "frame_width: 960",
                "frame_height: 540",
                "live_preview: true",
                "motion_preset: low",
                "aggressive_release_guard: true",
            ]
        ),
        encoding="utf-8",
    )

    config = build_config(parse_args([]))

    assert config.camera_index == 2
    assert config.target_fps == 55
    assert config.frame_width == 960
    assert config.frame_height == 540
    assert config.live_preview is True
    assert config.motion_preset == "low"
    assert config.aggressive_release_guard is True


def test_cli_flags_override_run_config_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".run.yaml").write_text(
        "\n".join(
            [
                "target_fps: 55",
                "live_preview: true",
                "aggressive_release_guard: true",
            ]
        ),
        encoding="utf-8",
    )

    config = build_config(
        parse_args(
            [
                "--target-fps",
                "24",
                "--no-live-preview",
                "--no-aggressive-release-guard",
            ]
        )
    )

    assert config.target_fps == 24
    assert config.live_preview is False
    assert config.aggressive_release_guard is False


def test_no_run_config_ignores_local_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".run.yaml").write_text("target_fps: 55\n", encoding="utf-8")

    config = build_config(parse_args(["--no-run-config"]))

    assert config.target_fps == DEFAULT_TARGET_FPS


def test_parse_args_disables_live_preview_flag() -> None:
    args = parse_args(["--no-live-preview"])

    config = build_config(args)

    assert config.live_preview is False


def test_build_app_passes_live_preview_to_gesture_service(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
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
    assert captured_kwargs["motion_preset"] == "medium"
    assert captured_kwargs["aggressive_release_guard"] is False
    assert isinstance(app, App)
    assert app.gesture_input is not None
    assert app.bridge is fake_bridge
    assert app.render_output is fake_render


def test_parse_args_accepts_motion_preset() -> None:
    args = parse_args(["--motion-preset", "low"])

    config = build_config(args)

    assert config.motion_preset == "low"


def test_parse_args_enables_aggressive_release_guard() -> None:
    args = parse_args(["--aggressive-release-guard"])

    config = build_config(args)

    assert config.aggressive_release_guard is True