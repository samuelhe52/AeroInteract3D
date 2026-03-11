from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from src.contracts import Vec3

try:
    from mediapipe.tasks.python.core.base_options import BaseOptions as MpBaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode as MpVisionTaskRunningMode,
    )
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker as MpHandLandmarker,
        HandLandmarkerOptions as MpHandLandmarkerOptions,
        HandLandmarksConnections as MpHandLandmarksConnections,
    )
except ImportError:
    MpBaseOptions = None
    MpVisionTaskRunningMode = None
    MpHandLandmarker = None
    MpHandLandmarkerOptions = None
    MpHandLandmarksConnections = None


HAND_MODEL_ENV_VAR = "AEROINTERACT3D_HAND_MODEL"
DEFAULT_HAND_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

DEFAULT_PINCH_ENTER_THRESHOLD = 0.10
DEFAULT_PINCH_HOLD_THRESHOLD = 0.06
DEFAULT_PINCH_RELEASE_THRESHOLD = 0.12

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


@dataclass(slots=True)
class GestureRuntimeConfig:
    input_video: str | None
    camera_index: int | None
    hand_model: str | None
    output_dir: Path | None
    target_fps: float | None
    max_frames: int
    min_detection_confidence: float
    min_tracking_confidence: float
    model_complexity: int
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True
    pinch_enter_threshold: float = DEFAULT_PINCH_ENTER_THRESHOLD
    pinch_hold_threshold: float = DEFAULT_PINCH_HOLD_THRESHOLD
    release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD


def default_hand_model_path() -> Path:
    return DEFAULT_HAND_MODEL_PATH


def resolve_hand_model_path(hand_model: str | None) -> str | None:
    if hand_model:
        return hand_model

    env_hand_model = os.getenv(HAND_MODEL_ENV_VAR)
    if env_hand_model:
        return env_hand_model

    if DEFAULT_HAND_MODEL_PATH.exists():
        return str(DEFAULT_HAND_MODEL_PATH)
    return None


class LegacySolutionsHandDetector:
    backend_name = "mediapipe_solutions"

    def __init__(self, config: GestureRuntimeConfig) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def __enter__(self) -> "LegacySolutionsHandDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._hands.close()

    def detect(self, frame: Any, timestamp_ms: int) -> tuple[dict[str, Vec3] | None, float, Any]:
        del timestamp_ms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, 0.0, None

        hand_landmarks = results.multi_hand_landmarks[0]
        raw_confidence = 0.0
        if results.multi_handedness:
            raw_confidence = float(results.multi_handedness[0].classification[0].score)
        return extract_landmarks(hand_landmarks.landmark), raw_confidence, hand_landmarks

    def draw(self, frame: Any, raw_landmarks: Any) -> None:
        if raw_landmarks is None:
            return
        self._drawing.draw_landmarks(frame, raw_landmarks, self._connections)


class TaskHandLandmarkerDetector:
    backend_name = "mediapipe_tasks"

    def __init__(self, config: GestureRuntimeConfig) -> None:
        if (
            MpBaseOptions is None
            or MpVisionTaskRunningMode is None
            or MpHandLandmarker is None
            or MpHandLandmarkerOptions is None
            or MpHandLandmarksConnections is None
        ):
            raise RuntimeError("MediaPipe tasks API is unavailable in this environment")
        if not config.hand_model:
            raise RuntimeError("A hand landmarker model is required for MediaPipe tasks backend")
        model_path = Path(config.hand_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

        options = MpHandLandmarkerOptions(
            base_options=MpBaseOptions(model_asset_path=str(model_path)),
            running_mode=MpVisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = MpHandLandmarker.create_from_options(options)
        self._connections = list(MpHandLandmarksConnections.HAND_CONNECTIONS)

    def __enter__(self) -> "TaskHandLandmarkerDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._landmarker.close()

    def detect(self, frame: Any, timestamp_ms: int) -> tuple[dict[str, Vec3] | None, float, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None, 0.0, None

        raw_landmarks = result.hand_landmarks[0]
        raw_confidence = 0.0
        if result.handedness:
            raw_confidence = float(result.handedness[0][0].score)
        return extract_landmarks(raw_landmarks), raw_confidence, raw_landmarks

    def draw(self, frame: Any, raw_landmarks: Any) -> None:
        if raw_landmarks is None:
            return

        frame_height, frame_width = frame.shape[:2]
        for connection in self._connections:
            start = raw_landmarks[connection.start]
            end = raw_landmarks[connection.end]
            start_point = (int(start.x * frame_width), int(start.y * frame_height))
            end_point = (int(end.x * frame_width), int(end.y * frame_height))
            cv2.line(frame, start_point, end_point, (255, 180, 0), 2, cv2.LINE_AA)

        for landmark in raw_landmarks:
            center = (int(landmark.x * frame_width), int(landmark.y * frame_height))
            cv2.circle(frame, center, 3, (0, 255, 255), -1, cv2.LINE_AA)


def create_hand_detector(config: GestureRuntimeConfig) -> Any:
    if config.hand_model:
        detector = TaskHandLandmarkerDetector(config)
        logging.info("Using hand detector backend: %s", detector.backend_name)
        return detector

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        detector = LegacySolutionsHandDetector(config)
        logging.info("Using hand detector backend: %s", detector.backend_name)
        return detector

    raise RuntimeError(
        "No hand detector backend is available. MediaPipe solutions.hands is unavailable in this environment, "
        "and no hand_landmarker.task model was provided. Put the model at "
        f"{DEFAULT_HAND_MODEL_PATH}"
        " or pass --hand-model with a valid file."
    )


def landmark_to_vec3(landmark: Any) -> Vec3:
    return Vec3(
        x=max(-1.0, min(1.0, (float(landmark.x) - 0.5) * 2.0)),
        y=max(-1.0, min(1.0, (0.5 - float(landmark.y)) * 2.0)),
        z=max(-1.0, min(1.0, -float(landmark.z))),
    )


def extract_landmarks(hand_landmarks: Any) -> dict[str, Vec3]:
    if hasattr(hand_landmarks, "landmark"):
        source = hand_landmarks.landmark
    else:
        source = hand_landmarks
    points = [landmark_to_vec3(landmark) for landmark in source]
    return {name: points[index] for index, name in enumerate(LANDMARK_NAMES)}


def distance(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def open_capture(config: GestureRuntimeConfig) -> cv2.VideoCapture:
    if config.input_video is not None:
        return cv2.VideoCapture(config.input_video)
    assert config.camera_index is not None
    return cv2.VideoCapture(config.camera_index)


__all__ = [
    "DEFAULT_HAND_MODEL_PATH",
    "DEFAULT_PINCH_ENTER_THRESHOLD",
    "DEFAULT_PINCH_HOLD_THRESHOLD",
    "DEFAULT_PINCH_RELEASE_THRESHOLD",
    "GestureRuntimeConfig",
    "HAND_MODEL_ENV_VAR",
    "create_hand_detector",
    "default_hand_model_path",
    "distance",
    "extract_landmarks",
    "open_capture",
    "resolve_hand_model_path",
]