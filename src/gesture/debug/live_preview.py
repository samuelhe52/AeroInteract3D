from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass


if __package__ in {None, ""}:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.gesture.service import GestureInputServiceStub
from tests.debug_video import DebugVideoConfig, run_live_preview


@dataclass(slots=True)
class GesturePreviewConfig:
    log_level: str = "INFO"
    camera_index: int = 0
    target_fps: int = 30
    max_frames: int = 0
    hand_model: str | None = None
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture live preview debug entrypoint")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no frame limit")
    parser.add_argument("--window-name", default="Gesture Live Preview")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--hide-coordinates", action="store_true")
    parser.add_argument("--hand-model", help="path to a MediaPipe hand_landmarker.task model file")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_config(args: argparse.Namespace) -> GesturePreviewConfig:
    return GesturePreviewConfig(
        log_level=args.log_level.upper(),
        camera_index=args.camera_index,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        hand_model=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        window_name=args.window_name,
        mirror=not args.no_mirror,
        draw_coordinates=not args.hide_coordinates,
    )


def build_preview_config(config: GesturePreviewConfig) -> DebugVideoConfig:
    hand_model = config.hand_model
    if hand_model is None and GestureInputServiceStub.DEFAULT_HAND_MODEL_PATH.exists():
        hand_model = str(GestureInputServiceStub.DEFAULT_HAND_MODEL_PATH)

    return DebugVideoConfig(
        input_video=None,
        camera_index=config.camera_index,
        hand_model=hand_model,
        output_dir=None,
        target_fps=float(config.target_fps),
        max_frames=config.max_frames,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        model_complexity=config.model_complexity,
        window_name=config.window_name,
        mirror=config.mirror,
        draw_coordinates=config.draw_coordinates,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)
    try:
        run_live_preview(build_preview_config(config))
        return 0
    except Exception as exc:
        logging.exception(
            "Live gesture preview failed. Expected model path: %s. Error: %s",
            GestureInputServiceStub.DEFAULT_HAND_MODEL_PATH,
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))