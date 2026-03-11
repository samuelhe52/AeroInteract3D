from __future__ import annotations

from dataclasses import dataclass
import logging
import sys


if __package__ in {None, ""}:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.gesture.runtime import (
    GestureRuntimeConfig,
    HAND_MODEL_ENV_VAR,
    default_hand_model_path,
    resolve_hand_model_path,
)
from src.gesture.debug.live_preview_runtime import run_live_preview


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


def build_service(config: GesturePreviewConfig):
    from src.gesture.service import GestureServiceImpl

    return GestureServiceImpl(
        camera_index=config.camera_index,
        hand_model=config.hand_model,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        model_complexity=config.model_complexity,
    )


def build_preview_config(config: GesturePreviewConfig) -> GestureRuntimeConfig:
    hand_model = resolve_hand_model_path(config.hand_model)

    return GestureRuntimeConfig(
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


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


__all__ = [
    "HAND_MODEL_ENV_VAR",
    "GesturePreviewConfig",
    "build_preview_config",
    "build_service",
    "default_hand_model_path",
    "resolve_hand_model_path",
    "setup_logging",
]

def main(config: GesturePreviewConfig | None = None) -> int:
    preview_config = config or GesturePreviewConfig()
    setup_logging(preview_config.log_level)
    try:
        run_live_preview(build_preview_config(preview_config))
        return 0
    except Exception as exc:
        logging.exception(
            "Live gesture preview failed. Expected model path: %s. Error: %s",
            default_hand_model_path(),
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())