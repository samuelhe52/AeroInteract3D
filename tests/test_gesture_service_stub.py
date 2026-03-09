from __future__ import annotations

import pytest

from src.gesture.service_stub import GestureInputServiceStub


def test_first_pinched_frame_gets_confidence_bonus() -> None:
    service = GestureInputServiceStub()
    service.start()

    first_pinched_packet = None
    for _ in range(100):
        packet = service.poll()
        if packet is not None and packet.pinch_state == "pinched":
            first_pinched_packet = packet
            break

    service.stop()

    assert first_pinched_packet is not None
    assert first_pinched_packet.confidence == pytest.approx(0.95)


def test_packet_timestamp_matches_frame_timestamp() -> None:
    class CapturingGestureService(GestureInputServiceStub):
        def __init__(self) -> None:
            super().__init__()
            self.last_frame_timestamp_ms: int | None = None

        def _read_frame(self) -> dict[str, object]:
            frame = super()._read_frame()
            self.last_frame_timestamp_ms = int(frame["timestamp_ms"])
            return frame

    service = CapturingGestureService()
    service.start()
    packet = service.poll()
    service.stop()

    assert packet is not None
    assert packet.timestamp_ms == service.last_frame_timestamp_ms


def test_poll_raises_when_backend_fails() -> None:
    service = GestureInputServiceStub()
    service.start()
    service._read_frame = lambda: (_ for _ in ()).throw(RuntimeError("camera failed"))

    with pytest.raises(RuntimeError, match="Gesture input backend failure"):
        service.poll()

    health = service.health()
    service.stop()

    assert health["status"] == "DEGRADED"
    assert health["last_error"] == "camera failed"