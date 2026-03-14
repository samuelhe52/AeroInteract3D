from __future__ import annotations

import numpy as np

from src.gesture.runtime import RawHandObservation
from src.gesture.service import GestureServiceImpl
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


class FakePreview:
    def __init__(self) -> None:
        self.calls = 0
        self.is_open = True

    def render(self, frame, *, observation, packet) -> None:
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