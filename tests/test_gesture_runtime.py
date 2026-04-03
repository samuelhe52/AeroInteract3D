from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.contracts import Vec3
from src.contracts import GesturePacket
from src.gesture.debug.live_preview_runtime import GesturePreviewWindow
from src.gesture.runtime import CaptureRuntime, HandLandmarkerRuntime, landmark_to_camera_vec3


class FakeVideoCapture:
    def __init__(self, camera_index: int) -> None:
        self.camera_index = camera_index
        self.frame = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))
        self.settings: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> None:
        self.settings[key] = value

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


def test_capture_runtime_flips_frames_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

    runtime = CaptureRuntime(camera_index=0, frame_width=640, frame_height=480, target_fps=30.0)
    frame = runtime.read()

    assert frame is not None
    assert np.array_equal(frame, np.flip(FakeVideoCapture(0).frame, axis=1))


def test_capture_runtime_can_leave_frames_unflipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

    runtime = CaptureRuntime(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        target_fps=30.0,
        flip_horizontal=False,
    )
    frame = runtime.read()

    assert frame is not None
    assert np.array_equal(frame, FakeVideoCapture(0).frame)


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


def test_detect_multi_returns_two_hands_when_detector_reports_both(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HandLandmarkerRuntime.__new__(HandLandmarkerRuntime)
    runtime._mp = SimpleNamespace(
        ImageFormat=SimpleNamespace(SRGB="srgb"),
        Image=lambda *, image_format, data: SimpleNamespace(image_format=image_format, data=data),
    )
    runtime._landmarker = SimpleNamespace(
        detect_for_video=lambda image, timestamp_ms: SimpleNamespace(
            hand_landmarks=[
                [SimpleNamespace(x=0.4, y=0.5, z=0.0) for _ in range(21)],
                [SimpleNamespace(x=0.6, y=0.5, z=0.0) for _ in range(21)],
            ],
            handedness=[
                [SimpleNamespace(category_name="Right", score=0.96)],
                [SimpleNamespace(category_name="Left", score=0.88)],
            ],
        )
    )
    runtime._fallback_state = SimpleNamespace()
    runtime._update_fallback_state = lambda *args, **kwargs: None

    frame = np.full((16, 16, 3), 96, dtype=np.uint8)
    observations = runtime.detect_multi(frame, timestamp_ms=300)

    assert len(observations) == 2
    assert {observation.handedness for observation in observations} == {"Right", "Left"}


def test_preview_window_draws_secondary_hand_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    window = GesturePreviewWindow()
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    circle_calls: list[tuple[int, int]] = []
    text_calls: list[str] = []

    monkeypatch.setattr(cv2, "circle", lambda image, center, radius, color, thickness=-1: circle_calls.append(center))
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args, **kwargs: text_calls.append(text))

    packet = GesturePacket(
        contract_version="2.0.0",
        frame_id=1,
        timestamp_ms=100,
        hand_id="hand-1",
        tracking_state="tracked",
        confidence=0.9,
        pinch_state="open",
        index_tip=Vec3(0.0, 0.0, 0.0),
        thumb_tip=Vec3(0.0, 0.0, 0.0),
        wrist=Vec3(0.0, 0.0, 0.0),
        coordinate_space="camera_norm",
        debug={
            "secondary_hand": {
                "hand_id": "hand-2",
                "handedness": "left",
                "tracking_state": "tracked",
                "confidence": 0.8,
                "pinch_state": "open",
                "index_tip": {"x": 0.2, "y": 0.2, "z": 0.0},
                "thumb_tip": {"x": 0.1, "y": 0.2, "z": 0.0},
                "wrist": {"x": 0.15, "y": 0.3, "z": 0.0},
            }
        },
    )

    window._draw_secondary_hand_overlay(canvas, packet=packet, width=32, height=32)

    assert len(circle_calls) == 4
    assert any(text.startswith("secondary:") for text in text_calls)


def test_preview_window_renders_explicit_hand_recognition_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    window = GesturePreviewWindow()
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    text_calls: list[str] = []

    monkeypatch.setattr(cv2, "putText", lambda image, text, *args, **kwargs: text_calls.append(text))
    monkeypatch.setattr(cv2, "circle", lambda *args, **kwargs: None)

    packet = GesturePacket(
        contract_version="2.0.0",
        frame_id=2,
        timestamp_ms=120,
        hand_id="hand-1",
        tracking_state="tracked",
        confidence=0.9,
        pinch_state="open",
        index_tip=Vec3(0.0, 0.0, 0.0),
        thumb_tip=Vec3(0.0, 0.0, 0.0),
        wrist=Vec3(0.0, 0.0, 0.0),
        coordinate_space="camera_norm",
        debug={
            "primary_hand": {
                "hand_id": "hand-1",
                "handedness": "right",
                "tracking_state": "tracked",
                "confidence": 0.9,
                "pinch_state": "open",
                "index_tip": {"x": 0.0, "y": 0.0, "z": 0.0},
                "thumb_tip": {"x": 0.0, "y": 0.0, "z": 0.0},
                "wrist": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "secondary_hand": None,
        },
    )

    window.render(canvas, observation=None, packet=packet)

    assert any(text.startswith("主手识别:") for text in text_calls)
    assert any(text.startswith("副手识别:") for text in text_calls)
    assert any(text.startswith("双手同时识别:") for text in text_calls)


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


def test_preview_window_draws_colored_scale_ratio_line(monkeypatch: pytest.MonkeyPatch) -> None:
    window = GesturePreviewWindow()
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    text_calls: list[tuple[str, tuple[int, int, int]]] = []

    def capture_text(image, text, org, font_face, font_scale, color, thickness=1, lineType=cv2.LINE_AA):
        text_calls.append((text, color))

    monkeypatch.setattr(cv2, "putText", capture_text)

    packet = GesturePacket(
        contract_version="2.0.0",
        frame_id=3,
        timestamp_ms=140,
        hand_id="hand-1",
        tracking_state="tracked",
        confidence=0.95,
        pinch_state="pinched",
        index_tip=Vec3(0.0, 0.0, 0.0),
        thumb_tip=Vec3(0.0, 0.0, 0.0),
        wrist=Vec3(0.0, 0.0, 0.0),
        coordinate_space="camera_norm",
        debug={"dual_hand": {"scale_ratio": 1.23}},
    )

    window._draw_status_text(canvas, packet=packet, fps=30.0)

    scale_lines = [entry for entry in text_calls if entry[0].startswith("scale_ratio:")]
    assert scale_lines
    assert scale_lines[-1][1] == window._colors.scale_up
