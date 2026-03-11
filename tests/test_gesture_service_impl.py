from __future__ import annotations

import pytest

from src.contracts import Vec3
from src.gesture.service_impl import (
    GestureInputServiceImpl,
    GesturePreviewConfig,
    build_preview_config,
    build_service,
)
from src.utils.contracts import EXPECTED_CONTRACT_VERSION
from src.utils.runtime import LIFECYCLE_DEGRADED, LIFECYCLE_RUNNING


def test_health_uses_shared_runtime_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureInputServiceImpl(hand_model="stub.task")
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
    service = GestureInputServiceImpl(hand_model="stub.task")
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


def test_poll_failure_records_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GestureInputServiceImpl(hand_model="stub.task")
    monkeypatch.setattr(service, "_setup_backend", lambda: None)

    def raise_read_failure() -> dict[str, object]:
        raise RuntimeError("camera frame read failed")

    monkeypatch.setattr(service, "_read_frame", raise_read_failure)

    service.start()

    with pytest.raises(RuntimeError, match="Gesture input backend failure"):
        service.poll()

    health = service.health()
    assert health["lifecycle_state"] == LIFECYCLE_DEGRADED
    assert health["status"] == "degraded"
    assert health["stats"]["backend_failures"] == 1
    assert health["stats"]["last_error"] == "Gesture input backend failure"
    assert health["errors"][-1]["code"] == "gesture.backend.failure"


def test_build_service_uses_preview_config_fields() -> None:
    config = GesturePreviewConfig(
        camera_index=2,
        hand_model="custom.task",
        min_detection_confidence=0.7,
        min_tracking_confidence=0.8,
        model_complexity=2,
    )

    service = build_service(config)

    assert service._camera_index == 2
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
    monkeypatch.setattr(GestureInputServiceImpl, "DEFAULT_HAND_MODEL_PATH", model_path)

    preview_config = build_preview_config(GesturePreviewConfig())

    assert preview_config.hand_model == str(model_path)
    assert preview_config.camera_index == 0
    assert preview_config.target_fps == 30.0