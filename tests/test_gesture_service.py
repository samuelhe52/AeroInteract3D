from __future__ import annotations

import logging

import numpy as np

import src.gesture.service as gesture_service
from src.gesture.runtime import RawHandObservation
from src.gesture.service import GestureServiceImpl
from src.gesture.constants import ROT_SLOT_COUNT
from src.utils.runtime import LIFECYCLE_DEGRADED, LIFECYCLE_RUNNING, LIFECYCLE_STOPPED
from src.contracts import Vec3


def make_observation(*, wrist_x: float = 0.0, pinch_gap: float = 0.05) -> RawHandObservation:
    return RawHandObservation(
        index_tip=Vec3(wrist_x + pinch_gap, 0.1, 0.0),
        thumb_tip=Vec3(wrist_x, 0.1, 0.0),
        wrist=Vec3(wrist_x, 0.0, 0.0),
        confidence=0.92,
        raw_pinch_distance=pinch_gap,
        hand_scale=0.30,
        landmarks=[Vec3(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
    )


class FakeCapture:
    def __init__(self, **_: object) -> None:
        self.frames = [np.zeros((8, 8, 3), dtype=np.uint8), None]
        self.closed = False

    def read(self):
        if not self.frames:
            return None
        return self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self, **_: object) -> None:
        self.calls = 0
        self.closed = False

    def detect(self, frame, *, timestamp_ms: int):
        assert frame is not None
        assert timestamp_ms > 0
        self.calls += 1
        return make_observation(wrist_x=0.4 if self.calls > 1 else 0.0)

    def close(self) -> None:
        self.closed = True


class FakeDualHandDetector:
    def __init__(self, **_: object) -> None:
        self.closed = False

    def detect_multi(self, frame, *, timestamp_ms: int):
        assert frame is not None
        assert timestamp_ms > 0
        return [
            make_observation(wrist_x=0.0),
            make_observation(wrist_x=0.4),
        ]

    def close(self) -> None:
        self.closed = True


class FakePreview:
    def __init__(self) -> None:
        self.calls = 0
        self.is_open = True

    def render(self, frame, *, observation, packet, secondary_observation=None) -> None:
        assert frame is not None
        assert packet.frame_id > 0
        self.calls += 1

    def close(self) -> None:
        self.is_open = False


def test_gesture_service_emits_valid_packets_and_updates_preview() -> None:
    preview = FakePreview()
    service = GestureServiceImpl(
        preview_enabled=True,
        capture_factory=FakeCapture,
        detector_factory=FakeDetector,
        preview_factory=lambda: preview,
        clock=iter([1.0, 1.01]).__next__,
    )

    service.start()
    packet = service.poll()

    assert service.lifecycle_state == LIFECYCLE_RUNNING
    assert packet is not None
    assert packet.tracking_state == "tracked"
    assert packet.coordinate_space == "camera_norm"
    assert preview.calls == 1

    degraded_packet = service.poll()

    assert degraded_packet is not None
    assert degraded_packet.tracking_state == "temporarily_lost"
    assert service.lifecycle_state == LIFECYCLE_DEGRADED
    assert service.health()["stats"]["capture_failures"] == 1


def test_gesture_service_enters_degraded_mode_when_backends_fail_to_start() -> None:
    def broken_capture(**_: object):
        raise RuntimeError("camera unavailable")

    def broken_detector(**_: object):
        raise RuntimeError("detector unavailable")

    service = GestureServiceImpl(
        capture_factory=broken_capture,
        detector_factory=broken_detector,
        clock=iter([2.0]).__next__,
    )

    service.start()
    packet = service.poll()

    assert service.lifecycle_state == LIFECYCLE_DEGRADED
    assert packet is not None
    assert packet.tracking_state == "temporarily_lost"
    assert service.health()["errors"][0]["code"] == "gesture.capture.start_failed"

    service.stop()

    assert service.lifecycle_state == LIFECYCLE_STOPPED


def test_gesture_summary_logging_is_debug_only(monkeypatch, caplog) -> None:
    preview = FakePreview()
    service = GestureServiceImpl(
        preview_enabled=True,
        capture_factory=FakeCapture,
        detector_factory=FakeDetector,
        preview_factory=lambda: preview,
        clock=iter([3.0]).__next__,
    )
    monkeypatch.setattr(gesture_service, "GESTURE_FRAME_SUMMARY_INTERVAL", 1)
    service.start()

    with caplog.at_level(logging.INFO, logger="gesture.service"):
        packet = service.poll()

    assert packet is not None
    assert "Gesture summary frame=" not in caplog.text


def test_gesture_service_relies_on_reducer_runtime_fields_without_post_mutation() -> None:
    service = GestureServiceImpl(
        capture_factory=FakeCapture,
        detector_factory=FakeDetector,
        clock=iter([4.0]).__next__,
    )

    service.start()
    packet = service.poll()

    assert packet is not None
    assert packet.smoothing_hint is not None
    assert packet.debug is not None
    assert packet.smoothing_hint["observation_source"] == "detected"
    assert packet.debug["observation_source"] == "detected"
    assert packet.debug["appearance_match_score"] == 1.0
    assert packet.debug["predicted_tracked"] is False
    assert packet.debug["blur_level"] == 0.0
    assert "runtime_quality_hint" not in packet.smoothing_hint


def test_gesture_service_health_reports_aggressive_release_guard() -> None:
    service = GestureServiceImpl(
        aggressive_release_guard=True,
        capture_factory=FakeCapture,
        detector_factory=FakeDetector,
        clock=iter([5.0]).__next__,
    )

    service.start()

    assert service.health()["stats"]["aggressive_release_guard"] is True


def test_gesture_service_passes_flip_camera_setting_to_capture() -> None:
    captured_kwargs: dict[str, object] = {}

    class RecordingCapture(FakeCapture):
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    service = GestureServiceImpl(
        flip_camera=False,
        capture_factory=RecordingCapture,
        detector_factory=FakeDetector,
        clock=iter([6.0]).__next__,
    )

    service.start()

    assert captured_kwargs["flip_horizontal"] is False


def test_gesture_service_emits_both_hands_in_debug_payload() -> None:
    service = GestureServiceImpl(
        capture_factory=FakeCapture,
        detector_factory=FakeDualHandDetector,
        clock=iter([7.0]).__next__,
    )

    service.start()
    packet = service.poll()

    assert packet is not None
    assert packet.debug is not None
    assert packet.debug["dual_hand"]["active_hand_count"] == 2
    assert packet.debug["primary_hand"] is not None
    assert packet.debug["secondary_hand"] is not None
    assert isinstance(packet.debug["dual_hand"]["pinch_distance_xy"], float)
    assert packet.debug["dual_hand"]["both_pinched"] is False
    assert packet.debug["dual_hand"]["scale_ratio"] == 1.0


def test_gesture_service_skips_debug_payload_when_disabled() -> None:
    service = GestureServiceImpl(
        emit_debug_payload=False,
        capture_factory=FakeCapture,
        detector_factory=FakeDetector,
        clock=iter([8.0]).__next__,
    )

    service.start()
    packet = service.poll()

    assert packet is not None
    assert packet.debug is None
    assert isinstance(packet.rotation, dict)
    assert service.health()["stats"]["emit_debug_payload"] is False


def test_rotation_slot_count_is_24() -> None:
    assert ROT_SLOT_COUNT == 24
