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
    index_tip: Vec3 | None = None,
    thumb_tip: Vec3 | None = None,
    debug: dict | None = None,
) -> GesturePacket:
    return GesturePacket(
        contract_version="2.0.0",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        hand_id="hand-1",
        tracking_state=tracking_state,
        confidence=confidence,
        pinch_state=pinch_state,
        index_tip=index_tip or Vec3(0.1, 0.2, 0.3),
        thumb_tip=thumb_tip or Vec3(0.11, 0.19, 0.28),
        wrist=wrist or Vec3(0.0, 0.0, 0.0),
        coordinate_space="camera_norm",
        pinch_distance=0.02,
        debug=debug,
    )


def hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open") -> GesturePacket:
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(0.02, 0.01, 0.0),
        thumb_tip=Vec3(-0.02, -0.01, 0.0),
    )


def offset_hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open") -> GesturePacket:
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(0.06, 0.0, 0.0),
        thumb_tip=Vec3(0.02, 0.0, 0.0),
    )


def test_bridge_emits_init_scene_on_first_valid_packet() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    commands = bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    assert [command.command_type for command in commands] == ["init_scene", "set_hand_pose"]
    assert commands[0].payload["objects"][0]["object_id"] == "primary_cube"


def test_bridge_emits_hover_when_hand_moves_near_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    commands = bridge.process(hover_packet(frame_id=2, timestamp_ms=120))

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state"]
    assert commands[0].payload["visible"] is True
    assert commands[1].payload["interaction_state"] == "pending_grab"


def test_bridge_requires_hover_before_grab() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    commands = bridge.process(make_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"))

    assert [command.command_type for command in commands] == ["set_hand_pose"]


def test_bridge_enters_grab_from_hover_and_uses_relative_offset() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(offset_hover_packet(frame_id=2, timestamp_ms=120))
    commands = bridge.process(offset_hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_pose"]
    assert commands[1].payload["interaction_state"] == "grabbed"
    assert commands[2].payload["position"] == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.0})
    assert commands[2].payload["coordinate_space"] == "world_norm"

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            index_tip=Vec3(0.10, 0.0, 0.0),
            thumb_tip=Vec3(0.06, 0.0, 0.0),
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_pose"]
    assert commands[1].payload["position"] == pytest.approx({"x": -0.04, "y": 0.0, "z": 0.0})
    assert commands[1].payload["coordinate_space"] == "world_norm"


def test_bridge_uses_pinch_midpoint_and_inverts_horizontal_axis() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))
    packet = make_packet(frame_id=4, timestamp_ms=160, pinch_state="pinched")
    packet.index_tip = Vec3(0.6, 0.7, 0.1)
    packet.thumb_tip = Vec3(0.2, 0.5, 0.3)

    commands = bridge.process(packet)

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_pose"]
    assert commands[1].payload["position"] == pytest.approx({"x": -0.4, "y": 0.6, "z": -0.2})


def test_bridge_emits_hpr_only_in_rotation_mode() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    commands = bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            index_tip=Vec3(0.02, 0.01, 0.0),
            thumb_tip=Vec3(-0.02, -0.01, 0.0),
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 15.0,
                    "deg_y": -30.0,
                    "deg_z": 45.0,
                }
            },
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_pose"]
    assert commands[1].payload["interaction_state"] == "rotating"
    assert "position" not in commands[2].payload
    assert commands[2].payload["hpr"] == pytest.approx({"h": 0.0, "p": 0.0, "r": 0.0})
    assert commands[2].payload["coordinate_space"] == "world_norm"

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            index_tip=Vec3(0.02, 0.01, 0.0),
            thumb_tip=Vec3(-0.02, -0.01, 0.0),
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 25.0,
                    "deg_y": -10.0,
                    "deg_z": 60.0,
                }
            },
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_pose"]
    assert commands[1].payload["hpr"] == pytest.approx({"h": 10.0, "p": 20.0, "r": 15.0})


def test_bridge_rotation_mode_does_not_enter_grabbed_state() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))

    commands = bridge.process(
        hover_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
        )
    )

    assert commands[1].payload["interaction_state"] == "grabbed"

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            index_tip=Vec3(0.02, 0.01, 0.0),
            thumb_tip=Vec3(-0.02, -0.01, 0.0),
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 10.0,
                    "deg_y": 20.0,
                    "deg_z": 30.0,
                }
            },
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_pose"]
    assert commands[1].payload["interaction_state"] == "rotating"
    assert "position" not in commands[2].payload


def test_bridge_emits_rotation_updates_outside_grab_region_when_rotation_mode_is_active() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    commands = bridge.process(
        make_packet(
            frame_id=2,
            timestamp_ms=120,
            pinch_state="pinched",
            index_tip=Vec3(0.7, 0.7, 0.2),
            thumb_tip=Vec3(0.5, 0.5, 0.1),
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 25.0,
                    "deg_y": -10.0,
                    "deg_z": 5.0,
                }
            },
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_pose"]
    assert commands[1].payload["interaction_state"] == "rotating"
    assert "position" not in commands[2].payload
    assert commands[2].payload["hpr"] == pytest.approx({"h": 0.0, "p": 0.0, "r": 0.0})


def test_bridge_rotation_restarts_from_current_object_pose_instead_of_raw_hand_pose() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(
        make_packet(
            frame_id=2,
            timestamp_ms=120,
            pinch_state="pinched",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 10.0,
                    "deg_y": 20.0,
                    "deg_z": 30.0,
                }
            },
        )
    )
    bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 35.0,
                    "deg_y": 50.0,
                    "deg_z": 70.0,
                }
            },
        )
    )
    bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="open",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 35.0,
                    "deg_y": 50.0,
                    "deg_z": 70.0,
                }
            },
        )
    )

    commands = bridge.process(
        make_packet(
            frame_id=5,
            timestamp_ms=180,
            pinch_state="pinched",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 80.0,
                    "deg_y": 90.0,
                    "deg_z": 100.0,
                }
            },
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_pose"]
    assert commands[2].payload["hpr"] == pytest.approx({"h": 25.0, "p": 30.0, "r": 40.0})


def test_bridge_rotation_sensitivity_scales_rotation_delta() -> None:
    bridge = BridgeServiceImpl(rotation_sensitivity=2.0)
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(
        make_packet(
            frame_id=2,
            timestamp_ms=120,
            pinch_state="pinched",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 10.0,
                    "deg_y": 20.0,
                    "deg_z": 30.0,
                }
            },
        )
    )
    commands = bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            debug={
                "rotation": {
                    "mode_active": True,
                    "deg_x": 20.0,
                    "deg_y": 25.0,
                    "deg_z": 40.0,
                }
            },
        )
    )

    assert commands[1].payload["hpr"] == pytest.approx({"h": 20.0, "p": 10.0, "r": 20.0})


def test_bridge_resets_when_tracking_is_lost_during_grab() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            tracking_state="temporarily_lost",
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "reset_interaction", "set_object_state"]
    assert commands[0].payload["visible"] is False
    assert commands[1].payload["reason"] == "tracking_lost"
    assert commands[2].payload["interaction_state"] == "idle"


def test_bridge_returns_to_hover_on_release_when_hand_stays_near_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    commands = bridge.process(hover_packet(frame_id=4, timestamp_ms=160, pinch_state="open"))

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state"]
    assert commands[1].payload["interaction_state"] == "pending_grab"


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

    assert transformed == Vec3(-0.25, 0.5, 0.5)
    assert "Coordinate clipped" not in caplog.text
