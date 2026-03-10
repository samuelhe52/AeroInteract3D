from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass

from src.bridge.service import BridgeServiceImpl
from src.gesture.service_stub import GestureInputServiceStub
from src.ports import BridgeService, GestureInputPort, RenderOutputPort
from src.rendering.service import AeroRenderingService


LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_STOPPED = "STOPPED"


@dataclass(slots=True)
class AppConfig:
    contract_version: str = "0.1.0"
    log_level: str = "INFO"
    camera_index: int = 0
    target_fps: int = 60


class App:
    def __init__(
        self,
        config: AppConfig,
        gesture_input: GestureInputPort,
        bridge: BridgeService,
        render_output: RenderOutputPort,
    ) -> None:
        self.config = config
        self.gesture_input = gesture_input
        self.bridge = bridge
        self.render_output = render_output
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
        self._running = True
        logging.info("Application loop started")

        while self._running:
            loop_start = time.perf_counter()

            packet = self.gesture_input.poll()
            if packet is not None:
                commands = self.bridge.process(packet)
                for command in commands:
                    self.render_output.push(command)

            elapsed = time.perf_counter() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

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
    parser = argparse.ArgumentParser(description="AeroInteract3D bootstrap entrypoint")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=60)
    return parser.parse_args(argv)


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
        target_fps=args.target_fps,
    )


def build_app(config: AppConfig) -> App:
    gesture_input = GestureInputServiceStub()
    bridge = BridgeServiceImpl()
    render_output = AeroRenderingService()
    return App(config, gesture_input, bridge, render_output)


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
