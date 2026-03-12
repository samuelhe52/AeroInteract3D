from __future__ import annotations

from dataclasses import dataclass
import logging
import sys

if __package__ in {None, ""}:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.constants import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_TARGET_FPS,
)

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
    camera_index: int = DEFAULT_CAMERA_INDEX
    target_fps: int = DEFAULT_TARGET_FPS
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    max_frames: int = DEFAULT_MAX_FRAMES
    hand_model: str | None = None
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    model_complexity: int = DEFAULT_MODEL_COMPLEXITY
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True


def build_service(config: GesturePreviewConfig):
    from src.gesture.service import GestureServiceImpl

    return GestureServiceImpl(
        camera_index=config.camera_index,
        target_fps=float(config.target_fps),
        frame_width=config.frame_width,
        frame_height=config.frame_height,
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
        frame_width=config.frame_width,
        frame_height=config.frame_height,
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