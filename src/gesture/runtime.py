from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from src.constants import (
    DEPTH_ESTIMATION_FAR_HAND_SCALE,
    DEPTH_ESTIMATION_LOCAL_Z_WEIGHT,
    DEPTH_ESTIMATION_NEAR_HAND_SCALE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MODEL_COMPLEXITY,
    TEMPORAL_PINCH_ENTER_THRESHOLD as DEFAULT_PINCH_ENTER_THRESHOLD,
    TEMPORAL_PINCH_HOLD_THRESHOLD as DEFAULT_PINCH_HOLD_THRESHOLD,
    TEMPORAL_PINCH_RELEASE_THRESHOLD as DEFAULT_PINCH_RELEASE_THRESHOLD,
)
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
    frame_width: int | None
    frame_height: int | None
    max_frames: int = DEFAULT_MAX_FRAMES
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    model_complexity: int = DEFAULT_MODEL_COMPLEXITY
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True
    pinch_enter_threshold: float = DEFAULT_PINCH_ENTER_THRESHOLD
    pinch_hold_threshold: float = DEFAULT_PINCH_HOLD_THRESHOLD
    release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD


def configure_capture(
    capture: cv2.VideoCapture,
    *,
    target_fps: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> cv2.VideoCapture:
    if target_fps is not None and target_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(target_fps))
    if frame_width is not None and frame_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
    if frame_height is not None and frame_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
    return capture


def create_capture(
    *,
    input_video: str | None = None,
    camera_index: int | None = None,
    target_fps: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> cv2.VideoCapture:
    if input_video is not None:
        return cv2.VideoCapture(input_video)

    assert camera_index is not None
    return configure_capture(
        cv2.VideoCapture(camera_index),
        target_fps=target_fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )


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

        self._result_condition = threading.Condition()
        self._pending_results: dict[int, Any] = {}
        self._result_wait_timeout_s = self._resolve_result_wait_timeout(config.target_fps)

        options = MpHandLandmarkerOptions(
            base_options=MpBaseOptions(model_asset_path=str(model_path)),
            running_mode=MpVisionTaskRunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            result_callback=self._handle_result,
        )
        self._landmarker = MpHandLandmarker.create_from_options(options)
        self._connections = list(MpHandLandmarksConnections.HAND_CONNECTIONS)

    @staticmethod
    def _resolve_result_wait_timeout(target_fps: float | None) -> float:
        if target_fps is None or target_fps <= 0:
            return 0.1
        return max(0.05, 2.0 / float(target_fps))

    def _handle_result(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        with self._result_condition:
            self._pending_results[int(timestamp_ms)] = result
            self._result_condition.notify_all()

    def __enter__(self) -> "TaskHandLandmarkerDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._landmarker.close()

    def detect(self, frame: Any, timestamp_ms: int) -> tuple[dict[str, Vec3] | None, float, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        with self._result_condition:
            self._pending_results.pop(timestamp_ms, None)
            self._landmarker.detect_async(image, timestamp_ms)
            has_result = self._result_condition.wait_for(
                lambda: timestamp_ms in self._pending_results,
                timeout=self._result_wait_timeout_s,
            )
            result = self._pending_results.pop(timestamp_ms, None) if has_result else None

        if result is None:
            return None, 0.0, None
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


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def distance_2d(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def estimate_hand_scale(landmarks: dict[str, Vec3]) -> float:
    if not landmarks:
        return 0.0

    xs = [point.x for point in landmarks.values()]
    ys = [point.y for point in landmarks.values()]
    bbox_scale = max(max(xs) - min(xs), max(ys) - min(ys))

    wrist = landmarks.get("wrist")
    middle_mcp = landmarks.get("middle_finger_mcp")
    index_mcp = landmarks.get("index_finger_mcp")
    pinky_mcp = landmarks.get("pinky_mcp")

    palm_length = 0.0 if wrist is None or middle_mcp is None else distance_2d(wrist, middle_mcp)
    palm_width = 0.0 if index_mcp is None or pinky_mcp is None else distance_2d(index_mcp, pinky_mcp)
    stable_scale = max(palm_length, palm_width)
    if stable_scale > 0.0:
        return stable_scale
    return bbox_scale


def estimate_camera_depth_from_hand_scale(landmarks: dict[str, Vec3]) -> float:
    hand_scale = estimate_hand_scale(landmarks)
    scale_span = max(DEPTH_ESTIMATION_NEAR_HAND_SCALE - DEPTH_ESTIMATION_FAR_HAND_SCALE, 1e-6)
    normalized = (hand_scale - DEPTH_ESTIMATION_FAR_HAND_SCALE) / scale_span
    return _clip_unit(normalized * 2.0 - 1.0)


def estimate_palm_anchor(landmarks: dict[str, Vec3]) -> Vec3:
    wrist = landmarks.get("wrist")
    index_mcp = landmarks.get("index_finger_mcp")
    middle_mcp = landmarks.get("middle_finger_mcp")
    pinky_mcp = landmarks.get("pinky_mcp")

    if wrist is None:
        return Vec3(0.0, 0.0, 0.0)

    anchor_points = [point for point in (index_mcp, middle_mcp, pinky_mcp) if point is not None]
    if not anchor_points:
        return wrist

    averaged_mcp = Vec3(
        x=sum(point.x for point in anchor_points) / len(anchor_points),
        y=sum(point.y for point in anchor_points) / len(anchor_points),
        z=sum(point.z for point in anchor_points) / len(anchor_points),
    )
    return Vec3(
        x=_clip_unit(wrist.x * 0.35 + averaged_mcp.x * 0.65),
        y=_clip_unit(wrist.y * 0.35 + averaged_mcp.y * 0.65),
        z=_clip_unit(wrist.z * 0.60 + averaged_mcp.z * 0.40),
    )


def apply_camera_depth_heuristic(landmarks: dict[str, Vec3]) -> dict[str, Vec3]:
    if not landmarks:
        return landmarks

    estimated_depth = estimate_camera_depth_from_hand_scale(landmarks)
    wrist_z = landmarks.get("wrist", Vec3(0.0, 0.0, 0.0)).z
    adjusted_landmarks: dict[str, Vec3] = {}
    for name, point in landmarks.items():
        local_z = (point.z - wrist_z) * DEPTH_ESTIMATION_LOCAL_Z_WEIGHT
        adjusted_landmarks[name] = Vec3(
            x=point.x,
            y=point.y,
            z=_clip_unit(estimated_depth + local_z),
        )
    return adjusted_landmarks


def extract_landmarks(hand_landmarks: Any) -> dict[str, Vec3]:
    if hasattr(hand_landmarks, "landmark"):
        source = hand_landmarks.landmark
    else:
        source = hand_landmarks
    points = [landmark_to_vec3(landmark) for landmark in source]
    named_points = {name: points[index] for index, name in enumerate(LANDMARK_NAMES)}
    return apply_camera_depth_heuristic(named_points)


def distance(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def open_capture(config: GestureRuntimeConfig) -> cv2.VideoCapture:
    return create_capture(
        input_video=config.input_video,
        camera_index=config.camera_index,
        target_fps=config.target_fps,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
    )


__all__ = [
    "DEFAULT_HAND_MODEL_PATH",
    "DEFAULT_PINCH_ENTER_THRESHOLD",
    "DEFAULT_PINCH_HOLD_THRESHOLD",
    "DEFAULT_PINCH_RELEASE_THRESHOLD",
    "GestureRuntimeConfig",
    "HAND_MODEL_ENV_VAR",
    "apply_camera_depth_heuristic",
    "configure_capture",
    "create_hand_detector",
    "create_capture",
    "default_hand_model_path",
    "distance",
    "distance_2d",
    "estimate_camera_depth_from_hand_scale",
    "estimate_hand_scale",
    "estimate_palm_anchor",
    "extract_landmarks",
    "open_capture",
    "resolve_hand_model_path",
]