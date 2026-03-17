from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.gesture.constants import (
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEPTH_ESTIMATION_FAR_HAND_SCALE,
    DEPTH_ESTIMATION_LOCAL_Z_WEIGHT,
    DEPTH_ESTIMATION_NEAR_HAND_SCALE,
    GESTURE_DETECT_MAX_SIDE,
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
    observation_source: str = "detected"
    appearance_match_score: float = 1.0
    predicted_tracked: bool = False
    blur_level: float = 0.0
    quality_hint: float = 1.0


@dataclass(slots=True)
class _FallbackState:
    last_frame_gray: np.ndarray | None = None
    last_observation: RawHandObservation | None = None
    template_patch: np.ndarray | None = None
    last_wrist_px: tuple[float, float] | None = None
    velocity_px: tuple[float, float] = (0.0, 0.0)
    last_timestamp_ms: int | None = None
    fallback_frames: int = 0


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
        self._fallback_state = _FallbackState()

        # Small fixed limits keep fallback constant-time and low-latency.
        self._fallback_max_frames = 8
        self._fallback_search_radius_px = 28
        self._template_size_px = 40
        self._predicted_only_frames = 2
        self._min_template_match = 0.45

    def detect(self, frame_bgr: np.ndarray, *, timestamp_ms: int) -> RawHandObservation | None:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_level = self._estimate_blur_level(frame_gray)
        detect_frame = resize_for_detection(frame_bgr, max_side=GESTURE_DETECT_MAX_SIDE)
        rgb_frame = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return self._detect_fallback(
                frame_bgr,
                frame_gray,
                timestamp_ms=timestamp_ms,
                blur_level=blur_level,
            )

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

        observation = RawHandObservation(
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            confidence=_clamp(confidence),
            raw_pinch_distance=pinch_distance,
            hand_scale=hand_scale,
            landmarks=landmarks,
            handedness=handedness,
            observation_source="detected",
            appearance_match_score=1.0,
            predicted_tracked=False,
            blur_level=blur_level,
            quality_hint=max(0.0, 1.0 - (blur_level * 0.55)),
        )
        self._update_fallback_state(frame_gray, observation, timestamp_ms=timestamp_ms)
        return observation

    def _detect_fallback(
        self,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        *,
        timestamp_ms: int,
        blur_level: float,
    ) -> RawHandObservation | None:
        state = self._fallback_state
        if state.last_observation is None or state.fallback_frames >= self._fallback_max_frames:
            return None

        dt_scale = self._dt_scale(timestamp_ms)
        predicted_wrist_px = self._predict_wrist_px(dt_scale=dt_scale)
        source = "predicted"
        appearance_match_score = 0.2

        candidate_wrist_px = predicted_wrist_px
        if state.template_patch is not None:
            matched = self._local_template_match(frame_gray, predicted_wrist_px)
            if matched is not None:
                candidate_wrist_px, appearance_match_score = matched
                source = "fallback"

        if source == "fallback" and appearance_match_score < self._min_template_match:
            if state.fallback_frames >= self._predicted_only_frames:
                return None
            source = "predicted"
            appearance_match_score = max(0.0, appearance_match_score)

        if source == "predicted" and state.fallback_frames >= self._predicted_only_frames:
            return None

        fallback_observation = self._observation_from_wrist_shift(
            frame_bgr=frame_bgr,
            wrist_px=candidate_wrist_px,
            appearance_match_score=appearance_match_score,
            source=source,
            blur_level=blur_level,
        )
        if fallback_observation is None:
            return None

        state.fallback_frames += 1
        state.last_frame_gray = frame_gray
        state.last_timestamp_ms = timestamp_ms
        state.last_wrist_px = candidate_wrist_px
        state.last_observation = fallback_observation
        state.velocity_px = (
            state.velocity_px[0] * 0.75,
            state.velocity_px[1] * 0.75,
        )
        return fallback_observation

    def _update_fallback_state(
        self,
        frame_gray: np.ndarray,
        observation: RawHandObservation,
        *,
        timestamp_ms: int,
    ) -> None:
        state = self._fallback_state
        wrist_px = self._vec_to_pixel(observation.wrist, frame_gray.shape[1], frame_gray.shape[0])

        if state.last_wrist_px is not None and state.last_timestamp_ms is not None:
            dt_scale = self._dt_scale(timestamp_ms)
            if dt_scale > 0.0:
                state.velocity_px = (
                    (wrist_px[0] - state.last_wrist_px[0]) / dt_scale,
                    (wrist_px[1] - state.last_wrist_px[1]) / dt_scale,
                )
        else:
            state.velocity_px = (0.0, 0.0)

        state.last_observation = observation
        state.last_frame_gray = frame_gray
        state.last_wrist_px = wrist_px
        state.last_timestamp_ms = timestamp_ms
        state.template_patch = self._extract_patch(frame_gray, wrist_px)
        state.fallback_frames = 0

    def _observation_from_wrist_shift(
        self,
        *,
        frame_bgr: np.ndarray,
        wrist_px: tuple[float, float],
        appearance_match_score: float,
        source: str,
        blur_level: float,
    ) -> RawHandObservation | None:
        state = self._fallback_state
        last_observation = state.last_observation
        if last_observation is None or state.last_wrist_px is None:
            return None

        width = frame_bgr.shape[1]
        height = frame_bgr.shape[0]
        dx_norm = (wrist_px[0] - state.last_wrist_px[0]) * (2.0 / max(width, 1))
        dy_norm = -(wrist_px[1] - state.last_wrist_px[1]) * (2.0 / max(height, 1))

        shifted_index = self._shift_vec(last_observation.index_tip, dx_norm, dy_norm)
        shifted_thumb = self._shift_vec(last_observation.thumb_tip, dx_norm, dy_norm)
        shifted_wrist = self._shift_vec(last_observation.wrist, dx_norm, dy_norm)
        confidence = 0.62 if source == "fallback" else 0.58
        quality_hint = _clamp((appearance_match_score * 0.75) + ((1.0 - blur_level) * 0.25))

        return RawHandObservation(
            index_tip=shifted_index,
            thumb_tip=shifted_thumb,
            wrist=shifted_wrist,
            confidence=min(last_observation.confidence, confidence),
            raw_pinch_distance=last_observation.raw_pinch_distance,
            hand_scale=last_observation.hand_scale,
            landmarks=last_observation.landmarks,
            handedness=last_observation.handedness,
            detector_source="local_template" if source == "fallback" else "motion_predictor",
            observation_source=source,
            appearance_match_score=_clamp(appearance_match_score),
            predicted_tracked=True,
            blur_level=blur_level,
            quality_hint=quality_hint,
        )

    def _predict_wrist_px(self, *, dt_scale: float) -> tuple[float, float]:
        state = self._fallback_state
        if state.last_wrist_px is None:
            return (0.0, 0.0)
        return (
            state.last_wrist_px[0] + (state.velocity_px[0] * dt_scale),
            state.last_wrist_px[1] + (state.velocity_px[1] * dt_scale),
        )

    def _local_template_match(
        self,
        frame_gray: np.ndarray,
        predicted_center_px: tuple[float, float],
    ) -> tuple[tuple[float, float], float] | None:
        state = self._fallback_state
        template_patch = state.template_patch
        if template_patch is None:
            return None

        half = self._fallback_search_radius_px
        cx, cy = int(predicted_center_px[0]), int(predicted_center_px[1])
        min_x = max(cx - half, 0)
        max_x = min(cx + half, frame_gray.shape[1] - 1)
        min_y = max(cy - half, 0)
        max_y = min(cy + half, frame_gray.shape[0] - 1)

        roi = frame_gray[min_y : max_y + 1, min_x : max_x + 1]
        if (
            roi.shape[0] < template_patch.shape[0]
            or roi.shape[1] < template_patch.shape[1]
            or template_patch.size == 0
        ):
            return None

        result = cv2.matchTemplate(roi, template_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        top_left = (min_x + max_loc[0], min_y + max_loc[1])
        center = (
            top_left[0] + (template_patch.shape[1] * 0.5),
            top_left[1] + (template_patch.shape[0] * 0.5),
        )
        normalized_score = _clamp((float(max_val) + 1.0) * 0.5)
        return center, normalized_score

    def _extract_patch(self, frame_gray: np.ndarray, center_px: tuple[float, float]) -> np.ndarray | None:
        patch_half = self._template_size_px // 2
        cx, cy = int(center_px[0]), int(center_px[1])
        min_x = max(cx - patch_half, 0)
        min_y = max(cy - patch_half, 0)
        max_x = min(cx + patch_half, frame_gray.shape[1] - 1)
        max_y = min(cy + patch_half, frame_gray.shape[0] - 1)
        patch = frame_gray[min_y : max_y + 1, min_x : max_x + 1]
        if patch.shape[0] < 8 or patch.shape[1] < 8:
            return None
        return patch

    def _vec_to_pixel(self, vec: Vec3, width: int, height: int) -> tuple[float, float]:
        x = (vec.x + 1.0) * 0.5 * width
        y = (1.0 - (vec.y + 1.0) * 0.5) * height
        return x, y

    def _shift_vec(self, vec: Vec3, delta_x: float, delta_y: float) -> Vec3:
        return Vec3(
            x=_clamp_signed(vec.x + delta_x),
            y=_clamp_signed(vec.y + delta_y),
            z=vec.z,
        )

    def _estimate_blur_level(self, frame_gray: np.ndarray) -> float:
        lap_var = float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())
        # Higher blur means lower Laplacian variance, then invert to [0, 1].
        return _clamp(1.0 - (lap_var / (lap_var + 200.0)))

    def _dt_scale(self, timestamp_ms: int) -> float:
        state = self._fallback_state
        if state.last_timestamp_ms is None:
            return 1.0
        delta_ms = max(timestamp_ms - state.last_timestamp_ms, 1)
        return min(delta_ms / 16.0, 2.0)

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
        x=_clamp_signed((landmark.x * 2.0) - 1.0),
        y=_clamp_signed(1.0 - (landmark.y * 2.0)),
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


def resize_for_detection(frame_bgr: np.ndarray, *, max_side: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return frame_bgr

    scale = max_side / largest_side
    resized_width = max(int(width * scale), 1)
    resized_height = max(int(height * scale), 1)
    return cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _normalized_scale(scale: float) -> float:
    span = max(DEPTH_ESTIMATION_NEAR_HAND_SCALE - DEPTH_ESTIMATION_FAR_HAND_SCALE, 1e-6)
    return _clamp((scale - DEPTH_ESTIMATION_FAR_HAND_SCALE) / span)


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _clamp_signed(value: float, *, low: float = -1.0, high: float = 1.0) -> float:
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
    "resize_for_detection",
    "resolve_model_path",
]