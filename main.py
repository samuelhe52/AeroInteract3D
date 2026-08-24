from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import signal
import sys
import time
from dataclasses import dataclass

import yaml

from src.constants import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_TARGET_FPS,
)
from src.bridge.service import BridgeServiceImpl
from src.gesture.constants import GESTURE_MOTION_PRESET
from src.gesture.service import GestureServiceImpl
from src.ports import BridgeService, DebugFrameSource, GestureInputPort, RenderOutputPort
from src.rendering.service import RenderingServiceImpl


LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_STOPPED = "STOPPED"
DEFAULT_RUN_CONFIG_PATH = Path(__file__).resolve().with_name(".run.yaml")
RUN_CONFIG_HELP = "Path to a YAML file with machine-local default run options."
VALID_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
MAX_TARGET_FPS = 240
RUN_CONFIG_KEYS = frozenset(
    {
        "log_level",
        "camera_index",
        "flip_camera",
        "target_fps",
        "frame_width",
        "frame_height",
        "debug_stats",
        "render_position_sensitivity",
        "bridge_rotation_sensitivity",
        "motion_preset",
        "aggressive_release_guard",
        "virtual_hand",
    }
)


@dataclass(slots=True)
class AppConfig:
    contract_version: str = "0.1.0"
    log_level: str = "INFO"
    camera_index: int = DEFAULT_CAMERA_INDEX
    flip_camera: bool = True
    target_fps: int = DEFAULT_TARGET_FPS
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    debug_stats: bool = True
    render_position_sensitivity: float = 1.35
    bridge_rotation_sensitivity: float = 1.5
    motion_preset: str = GESTURE_MOTION_PRESET
    aggressive_release_guard: bool = False
    virtual_hand: dict | None = None


def _built_in_run_defaults() -> dict[str, object]:
    return {
        "log_level": "INFO",
        "camera_index": DEFAULT_CAMERA_INDEX,
        "flip_camera": True,
        "target_fps": DEFAULT_TARGET_FPS,
        "frame_width": DEFAULT_FRAME_WIDTH,
        "frame_height": DEFAULT_FRAME_HEIGHT,
        "debug_stats": True,
        "render_position_sensitivity": 1.35,
        "bridge_rotation_sensitivity": 1.5,
        "motion_preset": GESTURE_MOTION_PRESET,
        "aggressive_release_guard": False,
        "virtual_hand": {
            "scale": 4.0,
            "depth_scale": 4.0,
            "perspective_scale": 0.4,
            "stable_threshold": 3,
            "bone_color": [1.0, 0.5, 0.0],
            "bone_width": 2.0,
        },
    }


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _target_fps(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > MAX_TARGET_FPS:
        raise argparse.ArgumentTypeError(f"value must not exceed {MAX_TARGET_FPS}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than zero")
    return parsed


def _validate_run_config(config_data: object, path: Path) -> dict[str, object]:
    if config_data is None:
        return {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Run config {path} must contain a top-level mapping")

    unknown_keys = sorted(set(config_data) - RUN_CONFIG_KEYS)
    if unknown_keys:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(f"Run config {path} contains unsupported keys: {joined_keys}")

    validated: dict[str, object] = {}

    for key, value in config_data.items():
        if key in {"log_level", "motion_preset"}:
            if not isinstance(value, str):
                raise ValueError(f"Run config {path} field '{key}' must be a string")
            if key == "log_level":
                normalized_level = value.upper()
                if normalized_level not in VALID_LOG_LEVELS:
                    choices = ", ".join(level.lower() for level in VALID_LOG_LEVELS)
                    raise ValueError(
                        f"Run config {path} field 'log_level' must be one of: {choices}"
                    )
                value = normalized_level
            validated[key] = value
            continue

        if key == "flip_camera":
            if not isinstance(value, bool):
                raise ValueError(f"Run config {path} field '{key}' must be a boolean")
            validated[key] = value
            continue

        if key in {
            "camera_index",
            "target_fps",
            "frame_width",
            "frame_height",
        }:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Run config {path} field '{key}' must be an integer")
            if key == "camera_index" and value < 0:
                raise ValueError(f"Run config {path} field 'camera_index' must be zero or greater")
            if key == "target_fps" and not 1 <= value <= MAX_TARGET_FPS:
                raise ValueError(
                    f"Run config {path} field 'target_fps' must be between 1 and {MAX_TARGET_FPS}"
                )
            if key in {"frame_width", "frame_height"} and value <= 0:
                raise ValueError(f"Run config {path} field '{key}' must be greater than zero")
            validated[key] = value
            continue

        if key in {"render_position_sensitivity", "bridge_rotation_sensitivity"}:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Run config {path} field '{key}' must be a number")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"Run config {path} field '{key}' must be a finite number greater than zero"
                )
            validated[key] = float(value)
            continue

        if key in {"debug_stats", "aggressive_release_guard"}:
            if not isinstance(value, bool):
                raise ValueError(f"Run config {path} field '{key}' must be a boolean")
            validated[key] = value
            continue

        if key == "virtual_hand":
            if not isinstance(value, dict):
                raise ValueError(f"Run config {path} field '{key}' must be a dictionary")
            # Validate virtual_hand subfields
            if "scale" in value and not isinstance(value["scale"], (int, float)):
                raise ValueError(f"Run config {path} field 'virtual_hand.scale' must be a number")
            if "depth_scale" in value and not isinstance(value["depth_scale"], (int, float)):
                raise ValueError(f"Run config {path} field 'virtual_hand.depth_scale' must be a number")
            if "perspective_scale" in value and not isinstance(value["perspective_scale"], (int, float)):
                raise ValueError(f"Run config {path} field 'virtual_hand.perspective_scale' must be a number")
            if "stable_threshold" in value and not (
                isinstance(value["stable_threshold"], int)
                and not isinstance(value["stable_threshold"], bool)
                and value["stable_threshold"] > 0
            ):
                raise ValueError(f"Run config {path} field 'virtual_hand.stable_threshold' must be a positive integer")
            if "bone_color" in value and not (isinstance(value["bone_color"], list) and len(value["bone_color"]) == 3):
                raise ValueError(f"Run config {path} field 'virtual_hand.bone_color' must be a list of 3 numbers")
            if "bone_width" in value and not isinstance(value["bone_width"], (int, float)):
                raise ValueError(f"Run config {path} field 'virtual_hand.bone_width' must be a number")
            for numeric_key in ("scale", "depth_scale", "perspective_scale", "bone_width"):
                if numeric_key in value and (
                    isinstance(value[numeric_key], bool)
                    or not math.isfinite(float(value[numeric_key]))
                    or float(value[numeric_key]) <= 0.0
                ):
                    raise ValueError(
                        f"Run config {path} field 'virtual_hand.{numeric_key}' must be greater than zero"
                    )
            if "bone_color" in value and (
                not all(isinstance(channel, int | float) and not isinstance(channel, bool) for channel in value["bone_color"])
                or not all(0.0 <= float(channel) <= 1.0 for channel in value["bone_color"])
            ):
                raise ValueError(
                    f"Run config {path} field 'virtual_hand.bone_color' values must be numbers between 0 and 1"
                )
            validated[key] = value
            continue

    motion_preset = validated.get("motion_preset")
    if motion_preset is not None and motion_preset not in {"high", "medium", "low"}:
        raise ValueError(
            f"Run config {path} field 'motion_preset' must be one of: high, medium, low"
        )

    return validated


def load_run_config(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return _validate_run_config(yaml.safe_load(handle), path)


def _resolve_run_defaults(argv: list[str] | None) -> tuple[dict[str, object], Path | None]:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--run-config", default=str(DEFAULT_RUN_CONFIG_PATH))
    bootstrap_parser.add_argument("--no-run-config", action="store_true")
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)

    if bootstrap_args.no_run_config:
        return _built_in_run_defaults(), None

    run_config_path = Path(bootstrap_args.run_config)
    try:
        run_defaults = load_run_config(run_config_path)
    except ValueError as exc:
        bootstrap_parser.error(str(exc))

    defaults = _built_in_run_defaults()
    defaults.update(run_defaults)
    return defaults, run_config_path


class App:
    def __init__(
        self,
        config: AppConfig,
        gesture_input: GestureInputPort,
        bridge: BridgeService,
        render_output: RenderOutputPort,
        debug_frame_source: DebugFrameSource | None = None,
    ) -> None:
        self.config = config
        self.gesture_input = gesture_input
        self.bridge = bridge
        self.render_output = render_output
        self.debug_frame_source = debug_frame_source
        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._running = False

    def initialize(self) -> None:
        logging.info("Initializing application")
        self.gesture_input.start()
        self.bridge.start()
        self.render_output.start()
        self.lifecycle_state = LIFECYCLE_RUNNING

    def run(self) -> None:
        if self.lifecycle_state != LIFECYCLE_RUNNING:
            raise RuntimeError("App is not ready to run")

        frame_interval = 1.0 / max(self.config.target_fps, 1)
        next_frame_deadline = time.perf_counter()
        self._running = True
        logging.info("Application loop started")

        while self._running:
            self.render_output.step()

            packet = self.gesture_input.poll()
            if packet is not None:
                self.render_output.update_gesture_data(packet)

                if self.debug_frame_source is not None:
                    camera_frame, observation = self.debug_frame_source.get_camera_data()
                elif hasattr(self.gesture_input, "get_camera_data"):
                    camera_frame, observation = self.gesture_input.get_camera_data()
                else:
                    camera_frame, observation = None, None
                if camera_frame is not None:
                    self.render_output.update_camera_frame(camera_frame, observation, packet)

                commands = self.bridge.process(packet)
                for command in commands:
                    self.render_output.push(command)

            if not self._running:
                break

            next_frame_deadline += frame_interval
            now = time.perf_counter()
            if next_frame_deadline <= now:
                if now - next_frame_deadline >= frame_interval:
                    next_frame_deadline = now
                continue
            time.sleep(next_frame_deadline - now)

    def request_stop(self) -> None:
        self._running = False

    def shutdown(self) -> None:
        logging.info("Shutting down application")
        for component in (self.render_output, self.bridge, self.gesture_input):
            try:
                component.stop()
            except Exception:
                logging.exception("Component shutdown error")
        self.lifecycle_state = LIFECYCLE_STOPPED

    def health_snapshot(self) -> dict:
        return {
            "lifecycle_state": self.lifecycle_state,
            "gesture": self.gesture_input.health(),
            "bridge": self.bridge.health(),
            "render": self.render_output.health(),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults, run_config_path = _resolve_run_defaults(argv)
    parser = argparse.ArgumentParser(description="AeroInteract3D bootstrap entrypoint")
    parser.add_argument(
        "--run-config",
        default=None if run_config_path is None else str(run_config_path),
        help=RUN_CONFIG_HELP,
    )
    parser.add_argument(
        "--no-run-config",
        action="store_true",
        help="Ignore .run.yaml and use built-in defaults unless CLI flags override them.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=VALID_LOG_LEVELS,
        default=defaults["log_level"],
    )
    parser.add_argument("--camera-index", type=_non_negative_int, default=defaults["camera_index"])
    parser.add_argument(
        "--flip-camera",
        dest="flip_camera",
        action="store_true",
        help="Mirror the camera input horizontally.",
    )
    parser.add_argument(
        "--no-flip-camera",
        dest="flip_camera",
        action="store_false",
        help="Keep the camera input unmirrored.",
    )
    parser.add_argument("--target-fps", type=_target_fps, default=defaults["target_fps"])
    parser.add_argument("--frame-width", type=_positive_int, default=defaults["frame_width"])
    parser.add_argument("--frame-height", type=_positive_int, default=defaults["frame_height"])
    parser.add_argument(
        "--render-position-sensitivity",
        type=_positive_float,
        default=defaults["render_position_sensitivity"],
        help="Sensitivity multiplier applied to object translation in the rendering module.",
    )
    parser.add_argument(
        "--bridge-rotation-sensitivity",
        type=_positive_float,
        default=defaults["bridge_rotation_sensitivity"],
        help="Sensitivity multiplier applied to rotation deltas before the bridge emits object hpr updates.",
    )
    parser.add_argument(
        "--motion-preset",
        choices=["high", "medium", "low"],
        default=defaults["motion_preset"],
        help="Motion response preset for gesture smoothing and loss prediction.",
    )
    parser.add_argument(
        "--aggressive-release-guard",
        dest="aggressive_release_guard",
        action="store_true",
        help="Require higher-quality observations before pinch release is accepted.",
    )
    parser.add_argument(
        "--no-aggressive-release-guard",
        dest="aggressive_release_guard",
        action="store_false",
        help="Disable the stricter pinch release guard even if it is enabled in .run.yaml.",
    )
    parser.add_argument(
        "--debug-stats",
        dest="debug_stats",
        action="store_true",
        help="Show gesture statistics alongside the in-window camera preview.",
    )
    parser.add_argument(
        "--no-debug-stats",
        dest="debug_stats",
        action="store_false",
        help="Hide the gesture statistics overlay and keep only the in-window camera preview.",
    )
    parser.set_defaults(
        flip_camera=defaults["flip_camera"],
        debug_stats=defaults["debug_stats"],
        aggressive_release_guard=defaults["aggressive_release_guard"],
    )
    parsed = parser.parse_args(argv)
    parsed.virtual_hand = defaults.get("virtual_hand")
    return parsed


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        log_level=args.log_level.upper(),
        camera_index=args.camera_index,
        flip_camera=args.flip_camera,
        target_fps=args.target_fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        debug_stats=args.debug_stats,
        render_position_sensitivity=args.render_position_sensitivity,
        bridge_rotation_sensitivity=args.bridge_rotation_sensitivity,
        motion_preset=args.motion_preset,
        aggressive_release_guard=args.aggressive_release_guard,
        virtual_hand=getattr(args, "virtual_hand", None),
    )


def build_app(config: AppConfig) -> App:
    gesture_input = GestureServiceImpl(
        camera_index=config.camera_index,
        flip_camera=config.flip_camera,
        target_fps=float(config.target_fps),
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        preview_enabled=False,
        motion_preset=config.motion_preset,
        aggressive_release_guard=config.aggressive_release_guard,
        emit_debug_payload=True,
    )
    bridge = BridgeServiceImpl(
        input_mirrored=config.flip_camera,
        position_sensitivity=config.render_position_sensitivity,
        rotation_sensitivity=config.bridge_rotation_sensitivity,
    )
    render_output = RenderingServiceImpl(
        debug_stats_enabled=config.debug_stats,
        position_sensitivity=config.render_position_sensitivity,
        virtual_hand_config=config.virtual_hand,
    )
    app = App(config, gesture_input, bridge, render_output, debug_frame_source=gesture_input)
    if hasattr(render_output, "set_quit_callback"):
        render_output.set_quit_callback(app.request_stop)
    return app


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)
    app = build_app(config)

    def _handle_signal(signum: int, _frame: object) -> None:
        logging.info("Received signal %s, requesting shutdown", signum)
        app.request_stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        app.initialize()
        logging.info("Health snapshot: %s", app.health_snapshot())
        app.run()
        return 0
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 130
    except Exception:
        logging.exception("Fatal application error")
        return 1
    finally:
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
