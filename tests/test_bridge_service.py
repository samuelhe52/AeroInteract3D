from __future__ import annotations

import logging

import pytest

from src.bridge.service import BridgeServiceImpl
from src.contracts import GesturePacket, Vec3


def make_packet(
    *,
    frame_id: int,
    timestamp_ms: int,
    pinch_state: str = "open",
    tracking_state: str = "tracked",
    confidence: float = 0.95,
    wrist: Vec3 | None = None,
) -> GesturePacket:
    return GesturePacket(
        contract_version="1.0.0",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        hand_id="hand-1",
        tracking_state=tracking_state,
        confidence=confidence,
        pinch_state=pinch_state,
        index_tip=Vec3(0.1, 0.2, 0.3),
        thumb_tip=Vec3(0.11, 0.19, 0.28),
        wrist=wrist or Vec3(0.0, 0.0, 0.0),
        coordinate_space="camera_norm",
        pinch_distance=0.02,
    )


def test_bridge_emits_init_scene_on_first_valid_packet() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    commands = bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    assert [command.command_type for command in commands] == ["init_scene"]
    assert commands[0].payload["objects"][0]["object_id"] == "primary_cube"


def test_bridge_enters_grab_and_emits_pose_updates() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100, pinch_state="open"))
    commands = bridge.process(make_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"))

    assert [command.command_type for command in commands] == ["set_object_state", "set_object_pose"]
    assert commands[0].payload["interaction_state"] == "grabbed"
    assert commands[1].payload["position"] == pytest.approx({"x": -0.105, "y": 0.195, "z": 0.29})
    assert commands[1].payload["coordinate_space"] == "world_norm"

    commands = bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            wrist=Vec3(0.4, 0.2, -0.1),
            
        )
    )

    assert [command.command_type for command in commands] == ["set_object_pose"]
    assert commands[0].payload["position"] == pytest.approx({"x": -0.105, "y": 0.195, "z": 0.29})
    assert commands[0].payload["coordinate_space"] == "world_norm"


def test_bridge_uses_pinch_midpoint_and_inverts_horizontal_axis() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100, pinch_state="open"))
    packet = make_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched")
    packet.index_tip = Vec3(0.6, 0.7, 0.1)
    packet.thumb_tip = Vec3(0.2, 0.5, 0.3)

    commands = bridge.process(packet)

    assert commands[1].payload["position"] == pytest.approx({"x": -0.4, "y": 0.6, "z": 0.2})


def test_bridge_resets_when_tracking_is_lost_during_grab() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100, pinch_state="open"))
    bridge.process(make_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            tracking_state="temporarily_lost",
        )
    )

    assert [command.command_type for command in commands] == ["reset_interaction", "set_object_state"]
    assert commands[0].payload["reason"] == "tracking_lost"
    assert commands[1].payload["interaction_state"] == "idle"


def test_bridge_ignores_duplicate_frames_and_records_health_error() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=5, timestamp_ms=200))
    commands = bridge.process(make_packet(frame_id=5, timestamp_ms=220))

    assert commands == []
    health = bridge.health()
    assert health["stats"]["duplicate_packets"] == 1
    assert health["errors"][-1]["code"] == "bridge.packet.duplicate"
    assert "timestamp" in health["errors"][-1]


def test_bridge_records_coordinate_transform_faults() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    transformed = bridge._camera_to_world_position(None)

    assert transformed == Vec3(0.0, 0.0, 0.0)
    health = bridge.health()
    assert health["errors"][-1]["code"] == "bridge.coordinate.position.missing"
    assert "timestamp" in health["errors"][-1]


def test_bridge_does_not_log_coordinate_clipped_for_axis_inversion_only(caplog) -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    with caplog.at_level(logging.WARNING, logger="bridge.coordinate_transformation"):
        transformed = bridge._camera_to_world_position(Vec3(0.25, 0.5, -0.5))

    assert transformed == Vec3(-0.25, 0.5, -0.5)
    assert "Coordinate clipped" not in caplog.text