from __future__ import annotations

import logging
import main
import pytest

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


class FakeDebugFrameSource:
    def get_camera_data(self):
        return None, None


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
        self.quit_callback = None
        self.updated_gesture_packets = []
        self.updated_camera_frames = []

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

    def update_gesture_data(self, packet) -> None:
        self.updated_gesture_packets.append(packet)

    def update_camera_frame(self, frame, observation=None, packet=None) -> None:
        self.updated_camera_frames.append((frame, observation, packet))

    def set_quit_callback(self, callback) -> None:
        self.quit_callback = callback


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


def test_app_shutdown_request_does_not_wait_for_next_frame(monkeypatch) -> None:
    config = AppConfig(target_fps=1)
    gesture_input = FakeGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    app = App(config, gesture_input, bridge, render_output)
    gesture_input.app = app
    app.lifecycle_state = LIFECYCLE_RUNNING
    monkeypatch.setattr(main.time, "sleep", lambda _: pytest.fail("shutdown should not sleep"))

    app.run()

    assert render_output.step_calls == 1


def test_app_run_reads_camera_data_through_port() -> None:
    class CameraGestureInput(FakeGestureInput):
        def __init__(self) -> None:
            super().__init__()
            self._emitted = False

        def poll(self):
            if self._emitted:
                assert self.app is not None
                self.app.request_stop()
                return None

            self._emitted = True
            return object()

    class CameraDebugFrameSource(FakeDebugFrameSource):
        def get_camera_data(self):
            return "frame", "observation"

    config = AppConfig(target_fps=60)
    gesture_input = CameraGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    debug_frame_source = CameraDebugFrameSource()
    app = App(config, gesture_input, bridge, render_output, debug_frame_source=debug_frame_source)
    gesture_input.app = app

    app.lifecycle_state = LIFECYCLE_RUNNING

    app.run()

    assert len(render_output.updated_gesture_packets) == 1
    assert render_output.updated_camera_frames == [("frame", "observation", render_output.updated_gesture_packets[0])]


def test_app_run_keeps_camera_data_when_debug_stats_disabled() -> None:
    class CameraGestureInput(FakeGestureInput):
        def __init__(self) -> None:
            super().__init__()
            self._emitted = False

        def poll(self):
            if self._emitted:
                assert self.app is not None
                self.app.request_stop()
                return None

            self._emitted = True
            return object()

    class CameraDebugFrameSource(FakeDebugFrameSource):
        def get_camera_data(self):
            return "frame", "observation"

    config = AppConfig(target_fps=60, debug_stats=False)
    gesture_input = CameraGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    debug_frame_source = CameraDebugFrameSource()
    app = App(config, gesture_input, bridge, render_output, debug_frame_source=debug_frame_source)
    gesture_input.app = app

    app.lifecycle_state = LIFECYCLE_RUNNING

    app.run()

    assert len(render_output.updated_gesture_packets) == 1
    assert render_output.updated_camera_frames == [("frame", "observation", render_output.updated_gesture_packets[0])]


def test_app_rotation_status_logging_is_debug_only(caplog) -> None:
    class RotationGestureInput(FakeGestureInput):
        def __init__(self) -> None:
            super().__init__()
            self._poll_count = 0

        def poll(self):
            self._poll_count += 1
            if self._poll_count > 6:
                assert self.app is not None
                self.app.request_stop()
                return None

            return type(
                "Packet",
                (),
                {
                    "rotation": {
                        "enabled": False,
                        "rotating": False,
                        "slot": 0,
                        "slot_count": 24,
                        "source": "none",
                    }
                },
            )()

    config = AppConfig(target_fps=60, debug_stats=True)
    gesture_input = RotationGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    app = App(config, gesture_input, bridge, render_output)
    gesture_input.app = app
    app.lifecycle_state = LIFECYCLE_RUNNING

    with caplog.at_level(logging.INFO, logger="main"):
        app.run()

    assert "rotation enabled=False rotating=False slot=0/24 src=none" not in caplog.text


def test_parse_args_enables_debug_stats_flag() -> None:
    args = parse_args(["--debug-stats"])

    config = build_config(args)

    assert config.debug_stats is True


def test_build_config_uses_default_target_fps() -> None:
    config = build_config(parse_args(["--no-run-config"]))

    assert config.flip_camera is True
    assert config.target_fps == DEFAULT_TARGET_FPS
    assert config.render_position_sensitivity == 1.35
    assert config.bridge_rotation_sensitivity == 1.5
    assert config.motion_preset == "medium"
    assert config.aggressive_release_guard is False


def test_parse_args_disables_debug_stats_by_default() -> None:
    args = parse_args(["--no-run-config"])

    config = build_config(args)

    assert config.debug_stats is True


def test_parse_args_uses_run_config_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_RUN_CONFIG_PATH", tmp_path / ".run.yaml")
    run_config = tmp_path / ".run.yaml"
    run_config.write_text(
        "\n".join(
            [
                "camera_index: 2",
                "flip_camera: false",
                "target_fps: 55",
                "frame_width: 960",
                "frame_height: 540",
                "debug_stats: true",
                "render_position_sensitivity: 1.5",
                "bridge_rotation_sensitivity: 1.8",
                "motion_preset: low",
                "aggressive_release_guard: true",
            ]
        ),
        encoding="utf-8",
    )

    config = build_config(parse_args([]))

    assert config.camera_index == 2
    assert config.flip_camera is False
    assert config.target_fps == 55
    assert config.frame_width == 960
    assert config.frame_height == 540
    assert config.debug_stats is True
    assert config.render_position_sensitivity == 1.5
    assert config.bridge_rotation_sensitivity == 1.8
    assert config.motion_preset == "low"
    assert config.aggressive_release_guard is True


def test_parse_args_rejects_removed_live_preview_run_config_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_RUN_CONFIG_PATH", tmp_path / ".run.yaml")
    (tmp_path / ".run.yaml").write_text(
        "\n".join(
            [
                "target_fps: 55",
                "live_preview: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        parse_args([])


def test_cli_flags_override_run_config_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_RUN_CONFIG_PATH", tmp_path / ".run.yaml")
    (tmp_path / ".run.yaml").write_text(
        "\n".join(
            [
                "target_fps: 55",
                "debug_stats: true",
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
                "--no-debug-stats",
                "--no-aggressive-release-guard",
            ]
        )
    )

    assert config.target_fps == 24
    assert config.debug_stats is False
    assert config.aggressive_release_guard is False


def test_no_run_config_ignores_local_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_RUN_CONFIG_PATH", tmp_path / ".run.yaml")
    (tmp_path / ".run.yaml").write_text("target_fps: 55\n", encoding="utf-8")

    config = build_config(parse_args(["--no-run-config"]))

    assert config.target_fps == DEFAULT_TARGET_FPS


def test_parse_args_uses_default_run_config_independent_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DEFAULT_RUN_CONFIG_PATH", tmp_path / ".run.yaml")
    monkeypatch.chdir(tmp_path.parent)
    (tmp_path / ".run.yaml").write_text("target_fps: 55\n", encoding="utf-8")

    config = build_config(parse_args([]))

    assert config.target_fps == 55


def test_parse_args_disables_debug_stats_flag() -> None:
    args = parse_args(["--no-debug-stats"])

    config = build_config(args)

    assert config.debug_stats is False


def test_parse_args_disables_camera_flip() -> None:
    args = parse_args(["--no-flip-camera"])

    config = build_config(args)

    assert config.flip_camera is False


def test_parse_args_accepts_render_position_sensitivity() -> None:
    args = parse_args(["--render-position-sensitivity", "1.75"])

    config = build_config(args)

    assert config.render_position_sensitivity == 1.75


def test_parse_args_accepts_bridge_rotation_sensitivity() -> None:
    args = parse_args(["--bridge-rotation-sensitivity", "1.9"])

    config = build_config(args)

    assert config.bridge_rotation_sensitivity == 1.9


@pytest.mark.parametrize(
    "args",
    [
        ["--camera-index", "-1"],
        ["--target-fps", "0"],
        ["--target-fps", "241"],
        ["--frame-width", "0"],
        ["--frame-height", "-1"],
        ["--render-position-sensitivity", "0"],
        ["--render-position-sensitivity", "nan"],
        ["--bridge-rotation-sensitivity", "-0.5"],
    ],
)
def test_parse_args_rejects_invalid_runtime_ranges(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--no-run-config", *args])


@pytest.mark.parametrize(
    "config_text",
    [
        "camera_index: -1\n",
        "target_fps: 0\n",
        "target_fps: 241\n",
        "frame_width: 0\n",
        "frame_height: -1\n",
        "log_level: verbose\n",
        "virtual_hand:\n  scale: 0\n",
        "virtual_hand:\n  stable_threshold: false\n",
        "virtual_hand:\n  bone_color: [1.2, 0.5, 0.5]\n",
    ],
)
def test_run_config_rejects_invalid_runtime_ranges(tmp_path, config_text: str) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError):
        main.load_run_config(config_path)


def test_build_app_disables_gesture_preview_and_passes_debug_stats_to_renderer(monkeypatch) -> None:
    captured_gesture_kwargs: dict[str, object] = {}
    captured_render_kwargs: dict[str, object] = {}
    captured_bridge_kwargs: dict[str, object] = {}
    fake_bridge = object()

    class FakeGestureService:
        def __init__(self, **kwargs) -> None:
            captured_gesture_kwargs.update(kwargs)

    class FakeRenderingService:
        def __init__(self, **kwargs) -> None:
            captured_render_kwargs.update(kwargs)
            self.quit_callback = None

        def set_quit_callback(self, callback) -> None:
            self.quit_callback = callback

    monkeypatch.setattr(main, "GestureServiceImpl", FakeGestureService)
    def fake_bridge_factory(**kwargs):
        captured_bridge_kwargs.update(kwargs)
        return fake_bridge

    monkeypatch.setattr(main, "BridgeServiceImpl", fake_bridge_factory)
    monkeypatch.setattr(main, "RenderingServiceImpl", FakeRenderingService)

    app = main.build_app(AppConfig(debug_stats=True))

    assert captured_gesture_kwargs["preview_enabled"] is False
    assert captured_gesture_kwargs["emit_debug_payload"] is True
    assert captured_gesture_kwargs["flip_camera"] is True
    assert captured_gesture_kwargs["motion_preset"] == "medium"
    assert captured_gesture_kwargs["aggressive_release_guard"] is False
    assert captured_bridge_kwargs["input_mirrored"] is True
    assert captured_bridge_kwargs["position_sensitivity"] == 1.35
    assert captured_bridge_kwargs["rotation_sensitivity"] == 1.5
    assert captured_render_kwargs["debug_stats_enabled"] is True
    assert captured_render_kwargs["position_sensitivity"] == 1.35
    assert isinstance(app, App)
    assert app.gesture_input is not None
    assert app.bridge is fake_bridge
    assert app.render_output is not None
    assert callable(app.render_output.quit_callback)


def test_build_app_keeps_gesture_debug_payloads_when_debug_stats_disabled(monkeypatch) -> None:
    captured_gesture_kwargs: dict[str, object] = {}
    captured_render_kwargs: dict[str, object] = {}

    class FakeGestureService:
        def __init__(self, **kwargs) -> None:
            captured_gesture_kwargs.update(kwargs)

    class FakeBridgeService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRenderingService:
        def __init__(self, **kwargs) -> None:
            captured_render_kwargs.update(kwargs)
            self.quit_callback = None

        def set_quit_callback(self, callback) -> None:
            self.quit_callback = callback

    monkeypatch.setattr(main, "GestureServiceImpl", FakeGestureService)
    monkeypatch.setattr(main, "BridgeServiceImpl", FakeBridgeService)
    monkeypatch.setattr(main, "RenderingServiceImpl", FakeRenderingService)

    app = main.build_app(AppConfig(debug_stats=False))

    assert captured_gesture_kwargs["emit_debug_payload"] is True
    assert captured_render_kwargs["debug_stats_enabled"] is False
    assert isinstance(app, App)


def test_parse_args_accepts_motion_preset() -> None:
    args = parse_args(["--motion-preset", "low"])

    config = build_config(args)

    assert config.motion_preset == "low"


def test_parse_args_enables_aggressive_release_guard() -> None:
    args = parse_args(["--aggressive-release-guard"])

    config = build_config(args)

    assert config.aggressive_release_guard is True
