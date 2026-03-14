from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.constants import (
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEPTH_ESTIMATION_FAR_HAND_SCALE,
    DEPTH_ESTIMATION_LOCAL_Z_WEIGHT,
    DEPTH_ESTIMATION_NEAR_HAND_SCALE,
    GESTURE_MODEL_RELATIVE_PATH,
)
from src.contracts import Vec3


logger = logging.getLogger("gesture.runtime")

WRIST_LANDMARK_INDEX = 0
THUMB_TIP_LANDMARK_INDEX = 4
INDEX_TIP_LANDMARK_INDEX = 8


@dataclass(slots=True)
class RawHandObservation:
    index_tip: Vec3
    thumb_tip: Vec3
    wrist: Vec3
    confidence: float
    raw_pinch_distance: float
    hand_scale: float
    landmarks: list[Vec3]
    handedness: str | None = None
    detector_source: str = "mediapipe_tasks"


class CaptureRuntime:
    def __init__(
        self,
        *,
        camera_index: int,
        frame_width: int,
        frame_height: int,
        target_fps: float,
    ) -> None:
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._capture.set(cv2.CAP_PROP_FPS, target_fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)

    def close(self) -> None:
        self._capture.release()


class HandLandmarkerRuntime:
    def __init__(
        self,
        *,
        model_path: str | None = None,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - exercised only in real env failures.
            raise RuntimeError("mediapipe is not installed") from exc

        resolved_model_path = resolve_model_path(model_path)
        if not resolved_model_path.exists():
            raise RuntimeError(f"Gesture model file does not exist: {resolved_model_path}")

        self._mp = mp
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, *, timestamp_ms: int) -> RawHandObservation | None:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        landmarks = [
            Vec3(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
            )
            for landmark in hand_landmarks
        ]
        hand_scale = estimate_hand_scale(landmarks)
        depth_hint = estimate_hand_depth(landmarks, hand_scale)
        handedness = None
        confidence = 0.0

        if result.handedness:
            category = result.handedness[0][0]
            handedness = getattr(category, "category_name", None)
            confidence = float(getattr(category, "score", 0.0))

        index_tip = landmark_to_camera_vec3(landmarks[INDEX_TIP_LANDMARK_INDEX], depth_hint=depth_hint)
        thumb_tip = landmark_to_camera_vec3(landmarks[THUMB_TIP_LANDMARK_INDEX], depth_hint=depth_hint)
        wrist = landmark_to_camera_vec3(landmarks[WRIST_LANDMARK_INDEX], depth_hint=depth_hint)
        pinch_distance = normalized_pinch_distance(
            landmarks[INDEX_TIP_LANDMARK_INDEX],
            landmarks[THUMB_TIP_LANDMARK_INDEX],
            hand_scale=hand_scale,
        )

        return RawHandObservation(
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            confidence=_clamp(confidence),
            raw_pinch_distance=pinch_distance,
            hand_scale=hand_scale,
            landmarks=landmarks,
            handedness=handedness,
        )

    def close(self) -> None:
        self._landmarker.close()


def resolve_model_path(model_path: str | None = None) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / GESTURE_MODEL_RELATIVE_PATH


def estimate_hand_scale(landmarks: list[Vec3]) -> float:
    if not landmarks:
        return 0.0

    xs = [landmark.x for landmark in landmarks]
    ys = [landmark.y for landmark in landmarks]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def estimate_hand_depth(landmarks: list[Vec3], hand_scale: float) -> float:
    scale_weight = _normalized_scale(hand_scale)
    local_depth = _clamp((-sum(landmark.z for landmark in landmarks) / max(len(landmarks), 1)) / 0.25)
    blended = ((1.0 - DEPTH_ESTIMATION_LOCAL_Z_WEIGHT) * scale_weight) + (
        DEPTH_ESTIMATION_LOCAL_Z_WEIGHT * local_depth
    )
    return (2.0 * _clamp(blended)) - 1.0


def landmark_to_camera_vec3(landmark: Vec3, *, depth_hint: float) -> Vec3:
    local_depth = _clamp((-landmark.z) / 0.3)
    blended_depth = ((1.0 - DEPTH_ESTIMATION_LOCAL_Z_WEIGHT) * ((depth_hint + 1.0) * 0.5)) + (
        DEPTH_ESTIMATION_LOCAL_Z_WEIGHT * local_depth
    )
    return Vec3(
        x=_clamp((landmark.x * 2.0) - 1.0),
        y=_clamp(1.0 - (landmark.y * 2.0)),
        z=(2.0 * _clamp(blended_depth)) - 1.0,
    )


def distance(left: Vec3, right: Vec3) -> float:
    delta_x = left.x - right.x
    delta_y = left.y - right.y
    delta_z = left.z - right.z
    return float((delta_x * delta_x + delta_y * delta_y + delta_z * delta_z) ** 0.5)


def distance_2d(left: Vec3, right: Vec3) -> float:
    delta_x = left.x - right.x
    delta_y = left.y - right.y
    return float((delta_x * delta_x + delta_y * delta_y) ** 0.5)


def normalized_pinch_distance(index_tip: Vec3, thumb_tip: Vec3, *, hand_scale: float) -> float:
    return distance_2d(index_tip, thumb_tip) / max(hand_scale, 1e-6)


def _normalized_scale(scale: float) -> float:
    span = max(DEPTH_ESTIMATION_NEAR_HAND_SCALE - DEPTH_ESTIMATION_FAR_HAND_SCALE, 1e-6)
    return _clamp((scale - DEPTH_ESTIMATION_FAR_HAND_SCALE) / span)


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


__all__ = [
    "CaptureRuntime",
    "HandLandmarkerRuntime",
    "INDEX_TIP_LANDMARK_INDEX",
    "RawHandObservation",
    "THUMB_TIP_LANDMARK_INDEX",
    "WRIST_LANDMARK_INDEX",
    "distance",
    "distance_2d",
    "estimate_hand_depth",
    "estimate_hand_scale",
    "landmark_to_camera_vec3",
    "normalized_pinch_distance",
    "resolve_model_path",
]