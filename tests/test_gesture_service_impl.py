from __future__ import annotations

import pytest

from src.contracts import Vec3
from src.gesture.debug.live_preview import (
    GesturePreviewConfig,
    HAND_MODEL_ENV_VAR,
    build_preview_config,
    build_service,
)
from src.gesture.debug.live_preview_runtime import GestureDebugAnalyzer
from src.gesture.service import (
    GestureServiceImpl,
)
from src.utils.contracts import EXPECTED_CONTRACT_VERSION
from src.utils.runtime import LIFECYCLE_DEGRADED, LIFECYCLE_RUNNING


def test_health_uses_shared_runtime_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task")
    monkeypatch.setattr(service, "_setup_backend", lambda: None)

    service.start()

    health = service.health()

    assert health["component"] == "gesture"
    assert health["lifecycle_state"] == LIFECYCLE_RUNNING
    assert health["status"] == "ok"
    assert health["errors"] == []
    assert health["stats"]["started"] is True
    assert health["stats"]["packets_emitted"] == 0


def test_poll_emits_packet_with_expected_contract_version(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task")
    monkeypatch.setattr(service, "_setup_backend", lambda: None)
    monkeypatch.setattr(service, "_read_frame", lambda: {"timestamp_ms": 100, "tick": 1, "frame": object()})
    monkeypatch.setattr(
        service,
        "_detect_hand",
        lambda _raw_frame: {
            "index_tip": Vec3(0.2, 0.1, -0.1),
            "thumb_tip": Vec3(0.21, 0.11, -0.12),
            "palm_center": Vec3(0.0, 0.0, 0.0),
            "raw_confidence": 0.9,
        },
    )

    service.start()
    packet = service.poll()

    assert packet is not None
    assert packet.contract_version == EXPECTED_CONTRACT_VERSION
    health = service.health()
    assert health["stats"]["tracked_packets"] == 1
    assert health["stats"]["packets_emitted"] == 1


def test_poll_failure_records_structured_error_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task")
    monkeypatch.setattr(service, "_setup_backend", lambda: None)

    def raise_read_failure() -> dict[str, object]:
        raise RuntimeError("camera frame read failed")

    monkeypatch.setattr(service, "_read_frame", raise_read_failure)

    service.start()

    result = service.poll()

    assert result is None
    health = service.health()
    assert health["lifecycle_state"] == LIFECYCLE_DEGRADED
    assert health["status"] == "degraded"
    assert health["stats"]["backend_failures"] == 1
    assert health["stats"]["last_error"] == "Gesture input backend failure"
    assert health["errors"][-1]["code"] == "gesture.backend.failure"
    assert "timestamp" in health["errors"][-1]


def test_build_service_uses_preview_config_fields() -> None:
    config = GesturePreviewConfig(
        camera_index=2,
        frame_width=800,
        frame_height=600,
        hand_model="custom.task",
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8,
        model_complexity=2,
    )

    service = build_service(config)

    assert service._camera_index == 2
    assert service._target_fps == 60.0
    assert service._frame_width == 800
    assert service._frame_height == 600
    assert service._hand_model == "custom.task"
    assert service._min_detection_confidence == 0.7
    assert service._min_tracking_confidence == 0.8
    assert service._model_complexity == 2


def test_build_preview_config_uses_default_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv(HAND_MODEL_ENV_VAR, str(model_path))

    preview_config = build_preview_config(GesturePreviewConfig())

    assert preview_config.hand_model == str(model_path)
    assert preview_config.camera_index == 0
    assert preview_config.target_fps == 60.0
    assert preview_config.frame_width == 640
    assert preview_config.frame_height == 480


def test_poll_renders_live_preview_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task", preview_enabled=True)
    preview_calls: list[tuple[dict[str, object], dict[str, object] | None, object]] = []

    monkeypatch.setattr(service, "_setup_backend", lambda: None)
    monkeypatch.setattr(
        service,
        "_read_frame",
        lambda: {"timestamp_ms": 100, "tick": 1, "frame": object()},
    )
    monkeypatch.setattr(
        service,
        "_detect_hand",
        lambda _raw_frame: {
            "index_tip": Vec3(0.2, 0.1, -0.1),
            "thumb_tip": Vec3(0.21, 0.11, -0.12),
            "palm_center": Vec3(0.0, 0.0, 0.0),
            "raw_confidence": 0.9,
        },
    )
    monkeypatch.setattr(service, "_ensure_preview_window", lambda: None)
    monkeypatch.setattr(
        service,
        "_render_live_preview",
        lambda raw_frame, hand_data, packet: preview_calls.append((raw_frame, hand_data, packet)),
    )

    service.start()
    packet = service.poll()

    assert packet is not None
    assert len(preview_calls) == 1
    assert preview_calls[0][0]["tick"] == 1


def test_preview_failure_disables_preview_without_failing_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task", preview_enabled=True)

    monkeypatch.setattr(service, "_setup_backend", lambda: None)
    monkeypatch.setattr(
        service,
        "_read_frame",
        lambda: {"timestamp_ms": 100, "tick": 1, "frame": object()},
    )
    monkeypatch.setattr(
        service,
        "_detect_hand",
        lambda _raw_frame: {
            "index_tip": Vec3(0.2, 0.1, -0.1),
            "thumb_tip": Vec3(0.21, 0.11, -0.12),
            "palm_center": Vec3(0.0, 0.0, 0.0),
            "raw_confidence": 0.9,
        },
    )
    monkeypatch.setattr(service, "_ensure_preview_window", lambda: None)

    def raise_preview_failure(
        _raw_frame: dict[str, object],
        _hand_data: dict[str, object] | None,
        _packet,
    ) -> None:
        raise RuntimeError("preview backend failed")

    monkeypatch.setattr(service, "_render_live_preview", raise_preview_failure)

    service.start()
    packet = service.poll()

    assert packet is not None
    health = service.health()
    assert health["lifecycle_state"] == LIFECYCLE_RUNNING
    assert health["stats"]["preview_active"] is False
    assert health["stats"]["preview_failures"] == 1
    assert health["errors"][-1]["code"] == "gesture.preview.failure"


def test_live_preview_and_service_share_temporal_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureServiceImpl(hand_model="stub.task")
    preview = GestureDebugAnalyzer(build_preview_config(GesturePreviewConfig(hand_model="stub.task")))

    frames = iter(
        [
            {
                "timestamp_ms": 100,
                "tick": 1,
                "frame": object(),
                "hand": {
                    "index_tip": Vec3(0.2, 0.1, -0.1),
                    "thumb_tip": Vec3(0.21, 0.11, -0.12),
                    "palm_center": Vec3(0.0, 0.0, 0.0),
                    "raw_confidence": 0.9,
                },
            },
            {
                "timestamp_ms": 101,
                "tick": 2,
                "frame": object(),
                "hand": {
                    "index_tip": Vec3(0.25, 0.15, -0.15),
                    "thumb_tip": Vec3(0.26, 0.16, -0.16),
                    "palm_center": Vec3(0.3, 0.2, -0.1),
                    "raw_confidence": 0.9,
                },
            },
        ]
    )

    current_frame: dict[str, object] = {}

    def read_frame() -> dict[str, object]:
        nonlocal current_frame
        current_frame = next(frames)
        return {k: v for k, v in current_frame.items() if k != "hand"}

    def detect_hand(_raw_frame: dict[str, object]) -> dict[str, object]:
        return current_frame["hand"]  # type: ignore[return-value]

    monkeypatch.setattr(service, "_setup_backend", lambda: None)
    monkeypatch.setattr(service, "_read_frame", read_frame)
    monkeypatch.setattr(service, "_detect_hand", detect_hand)

    service.start()
    preview.start()

    packet_one = service.poll()
    preview_packet_one = preview.process_landmarks(
        {
            "index_finger_tip": Vec3(0.2, 0.1, -0.1),
            "thumb_tip": Vec3(0.21, 0.11, -0.12),
            "wrist": Vec3(0.0, 0.0, 0.0),
        },
        100,
        0.9,
    )
    packet_two = service.poll()
    preview_packet_two = preview.process_landmarks(
        {
            "index_finger_tip": Vec3(0.25, 0.15, -0.15),
            "thumb_tip": Vec3(0.26, 0.16, -0.16),
            "wrist": Vec3(0.3, 0.2, -0.1),
        },
        101,
        0.9,
    )

    assert packet_one is not None
    assert packet_two is not None
    assert packet_one.pinch_state == preview_packet_one.pinch_state
    assert packet_two.pinch_state == preview_packet_two.pinch_state
    assert packet_two.palm_center == preview_packet_two.palm_center
    assert packet_two.smoothing_hint == preview_packet_two.smoothing_hint