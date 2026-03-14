from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import time
import sys
from pathlib import Path

from src.constants import DEFAULT_CAMERA_INDEX, DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, DEFAULT_TARGET_FPS
from src.gesture.service import GestureServiceImpl


@dataclass(slots=True)
class GesturePreviewConfig:
    camera_index: int = DEFAULT_CAMERA_INDEX
    target_fps: int = DEFAULT_TARGET_FPS
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    log_level: str = "INFO"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture live preview")
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GesturePreviewConfig:
    return GesturePreviewConfig(
        camera_index=args.camera_index,
        target_fps=args.target_fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        log_level=args.log_level.upper(),
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    setup_logging(config.log_level)

    service = GestureServiceImpl(
        camera_index=config.camera_index,
        target_fps=float(config.target_fps),
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        preview_enabled=True,
    )

    try:
        service.start()
        frame_interval = 1.0 / max(config.target_fps, 1)
        while service.preview_is_open or service.health()["stats"]["packets_emitted"] == 0:
            loop_started_at = time.perf_counter()
            service.poll()
            sleep_for = frame_interval - (time.perf_counter() - loop_started_at)
            if sleep_for > 0:
                time.sleep(sleep_for)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        service.stop()


__all__ = ["GesturePreviewConfig", "build_config", "main", "parse_args"]
