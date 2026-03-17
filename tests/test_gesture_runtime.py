from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.contracts import Vec3
from src.gesture.runtime import HandLandmarkerRuntime, landmark_to_camera_vec3


def test_landmark_to_camera_vec3_preserves_negative_xy_coordinates() -> None:
    point = landmark_to_camera_vec3(Vec3(0.25, 0.75, 0.0), depth_hint=0.0)

    assert point.x == pytest.approx(-0.5)
    assert point.y == pytest.approx(-0.5)


def test_landmark_to_camera_vec3_clips_only_at_camera_space_bounds() -> None:
    point = landmark_to_camera_vec3(Vec3(-0.2, 1.4, 0.0), depth_hint=0.0)

    assert point.x == -1.0
    assert point.y == -1.0


def test_detect_reuses_single_gray_conversion_for_detected_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HandLandmarkerRuntime.__new__(HandLandmarkerRuntime)
    runtime._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda *, image_format, data: SimpleNamespace(image_format=image_format, data=data),
    )
    runtime._landmarker = SimpleNamespace(
        detect_for_video=lambda image, timestamp_ms: SimpleNamespace(
            hand_landmarks=[
                [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            ],
            handedness=[[SimpleNamespace(category_name="Right", score=0.9)]],
        )
    )
    runtime._fallback_state = SimpleNamespace()

    calls: list[int] = []
    captured_gray: list[np.ndarray] = []
    original_cvt_color = cv2.cvtColor

    def counting_cvt_color(frame: np.ndarray, code: int) -> np.ndarray:
        calls.append(code)
        return original_cvt_color(frame, code)

    def record_update(self, frame_gray: np.ndarray, observation, *, timestamp_ms: int) -> None:
        captured_gray.append(frame_gray)

    monkeypatch.setattr(cv2, "cvtColor", counting_cvt_color)
    monkeypatch.setattr(runtime, "_update_fallback_state", record_update.__get__(runtime, HandLandmarkerRuntime))

    frame = np.full((16, 16, 3), 32, dtype=np.uint8)
    observation = runtime.detect(frame, timestamp_ms=100)

    assert observation is not None
    assert calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert calls.count(cv2.COLOR_BGR2RGB) == 1
    assert len(captured_gray) == 1
    assert captured_gray[0].shape == frame.shape[:2]


def test_detect_reuses_single_gray_conversion_for_fallback_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HandLandmarkerRuntime.__new__(HandLandmarkerRuntime)
    runtime._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda *, image_format, data: SimpleNamespace(image_format=image_format, data=data),
    )
    runtime._landmarker = SimpleNamespace(
        detect_for_video=lambda image, timestamp_ms: SimpleNamespace(hand_landmarks=[], handedness=[])
    )
    runtime._fallback_state = SimpleNamespace()

    calls: list[int] = []
    captured_gray: list[np.ndarray] = []
    original_cvt_color = cv2.cvtColor

    def counting_cvt_color(frame: np.ndarray, code: int) -> np.ndarray:
        calls.append(code)
        return original_cvt_color(frame, code)

    def record_fallback(
        self,
        frame_bgr: np.ndarray,
        frame_gray: np.ndarray,
        *,
        timestamp_ms: int,
        blur_level: float,
    ):
        captured_gray.append(frame_gray)
        return None

    monkeypatch.setattr(cv2, "cvtColor", counting_cvt_color)
    monkeypatch.setattr(runtime, "_detect_fallback", record_fallback.__get__(runtime, HandLandmarkerRuntime))

    frame = np.full((16, 16, 3), 64, dtype=np.uint8)
    observation = runtime.detect(frame, timestamp_ms=200)

    assert observation is None
    assert calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert calls.count(cv2.COLOR_BGR2RGB) == 1
    assert len(captured_gray) == 1
    assert captured_gray[0].shape == frame.shape[:2]
