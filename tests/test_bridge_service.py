from __future__ import annotations

import logging

import numpy as np
import pytest

from src.bridge.service import BridgeServiceImpl, TABLE_SCENE_OBJECTS, TABLE_SURFACE_Y
from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import RawHandObservation
from src.gesture.service import GestureServiceImpl
from src.utils.contracts import vec3_payload


def interactable_object_ids() -> list[str]:
    return [str(obj["object_id"]) for obj in TABLE_SCENE_OBJECTS if bool(obj.get("interactable", True))]


def default_test_object_id() -> str:
    return interactable_object_ids()[0]


def alternate_test_object_id() -> str:
    return interactable_object_ids()[1]


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


def with_secondary_hand(
    packet: GesturePacket,
    *,
    object_id: str | None = None,
    tracking_state: str = "tracked",
    pinch_state: str = "open",
    pinch_distance: float | None = None,
    confidence: float = 0.95,
    index_tip: Vec3 | None = None,
    thumb_tip: Vec3 | None = None,
    wrist: Vec3 | None = None,
    secondary_debug: dict | None = None,
) -> GesturePacket:
    anchor_camera = object_anchor_camera(object_id)
    debug = dict(packet.debug or {})
    debug["secondary_hand"] = {
        "hand_id": "hand-2",
        "handedness": "left",
        "tracking_state": tracking_state,
        "confidence": confidence,
        "pinch_state": pinch_state,
        "pinch_distance": pinch_distance,
        "index_tip": {
            "x": (index_tip or Vec3(anchor_camera.x + 0.06, anchor_camera.y, anchor_camera.z)).x,
            "y": (index_tip or Vec3(anchor_camera.x + 0.06, anchor_camera.y, anchor_camera.z)).y,
            "z": (index_tip or Vec3(anchor_camera.x + 0.06, anchor_camera.y, anchor_camera.z)).z,
        },
        "thumb_tip": {
            "x": (thumb_tip or Vec3(anchor_camera.x + 0.02, anchor_camera.y, anchor_camera.z)).x,
            "y": (thumb_tip or Vec3(anchor_camera.x + 0.02, anchor_camera.y, anchor_camera.z)).y,
            "z": (thumb_tip or Vec3(anchor_camera.x + 0.02, anchor_camera.y, anchor_camera.z)).z,
        },
        "wrist": {
            "x": (wrist or Vec3(anchor_camera.x + 0.04, anchor_camera.y - 0.04, anchor_camera.z)).x,
            "y": (wrist or Vec3(anchor_camera.x + 0.04, anchor_camera.y - 0.04, anchor_camera.z)).y,
            "z": (wrist or Vec3(anchor_camera.x + 0.04, anchor_camera.y - 0.04, anchor_camera.z)).z,
        },
        "debug": {} if secondary_debug is None else secondary_debug,
    }
    packet.debug = debug
    return packet


def scene_object_position(object_id: str) -> Vec3:
    descriptor = next(obj for obj in TABLE_SCENE_OBJECTS if obj["object_id"] == object_id)
    init_pos = descriptor["init_pos"]
    return Vec3(float(init_pos["x"]), float(init_pos["y"]), float(init_pos["z"]))


def scene_object_descriptor(object_id: str) -> dict:
    return next(obj for obj in TABLE_SCENE_OBJECTS if obj["object_id"] == object_id)


def target_object_position(object_id: str | None = None) -> Vec3:
    return scene_object_position(default_test_object_id() if object_id is None else object_id)


def target_object_initial_hpr(object_id: str | None = None) -> tuple[float, float, float]:
    descriptor = scene_object_descriptor(default_test_object_id() if object_id is None else object_id)
    init_hpr = descriptor["init_hpr"]
    return (float(init_hpr["h"]), float(init_hpr["p"]), float(init_hpr["r"]))


def target_object_grabbed_y(object_id: str | None = None) -> float:
    descriptor = scene_object_descriptor(default_test_object_id() if object_id is None else object_id)
    init_pos_y = float(descriptor["init_pos"]["y"])
    half_height = float(descriptor.get("collision_half_height", float(descriptor["scale"]["y"]) * 0.5))
    return max(init_pos_y, TABLE_SURFACE_Y + half_height)


def world_to_camera_position(position: Vec3) -> Vec3:
    return Vec3(-position.x, position.y, -position.z)


def object_anchor_camera(object_id: str | None = None) -> Vec3:
    return world_to_camera_position(target_object_position(object_id))


def object_camera_point(dx: float, dy: float = 0.0, dz: float = 0.0, *, object_id: str | None = None) -> Vec3:
    anchor_camera = object_anchor_camera(object_id)
    return Vec3(anchor_camera.x + dx, anchor_camera.y + dy, anchor_camera.z + dz)


def hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open", object_id: str | None = None) -> GesturePacket:
    anchor_camera = object_anchor_camera(object_id)
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(anchor_camera.x + 0.02, anchor_camera.y, anchor_camera.z),
        thumb_tip=Vec3(anchor_camera.x - 0.02, anchor_camera.y, anchor_camera.z),
    )


def offset_hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open", object_id: str | None = None) -> GesturePacket:
    anchor_camera = object_anchor_camera(object_id)
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(anchor_camera.x + 0.10, anchor_camera.y, anchor_camera.z),
        thumb_tip=Vec3(anchor_camera.x + 0.06, anchor_camera.y, anchor_camera.z),
    )


def test_table_scene_interactable_objects_start_on_collision_surface() -> None:
    for descriptor in TABLE_SCENE_OBJECTS:
        if not bool(descriptor.get("interactable", True)):
            continue
        init_y = float(descriptor["init_pos"]["y"])
        half_height = float(descriptor.get("collision_half_height", float(descriptor["scale"]["y"]) * 0.5))

        assert init_y == pytest.approx(TABLE_SURFACE_Y + half_height, abs=0.002)


def test_bridge_emits_init_scene_on_first_valid_packet() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    commands = bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    assert [command.command_type for command in commands] == ["init_scene", "set_hand_pose"]
    assert commands[0].payload["objects"][0]["object_id"] == "table_plane"


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
    target_position = target_object_position()
    assert commands[2].payload["position"] == pytest.approx({"x": target_position.x, "y": target_object_grabbed_y(), "z": target_position.z})
    assert commands[2].payload["coordinate_space"] == "world_norm"

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            index_tip=object_camera_point(0.14),
            thumb_tip=object_camera_point(0.10),
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_pose"]
    assert commands[1].payload["position"] == pytest.approx({"x": target_position.x - 0.04, "y": target_object_grabbed_y(), "z": target_position.z})
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
    h, p, r = target_object_initial_hpr()
    assert commands[2].payload["hpr"] == pytest.approx({"h": h, "p": p, "r": r})
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
    assert commands[1].payload["hpr"] == pytest.approx({"h": 2.0, "p": -12.0, "r": 15.0})


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
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    commands = bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
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
    assert commands[2].payload["hpr"] == pytest.approx({"h": 12.0, "p": 8.0, "r": 0.0})


def test_bridge_rotation_restarts_from_current_object_pose_instead_of_raw_hand_pose() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=110))
    bridge.process(
        make_packet(
            frame_id=3,
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
            frame_id=4,
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
            frame_id=5,
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
            frame_id=6,
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
    assert commands[2].payload["hpr"] == pytest.approx({"h": -13.0, "p": -22.0, "r": 40.0})


def test_bridge_rotation_sensitivity_scales_rotation_delta() -> None:
    bridge = BridgeServiceImpl(rotation_sensitivity=2.0)
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=110))
    bridge.process(
        make_packet(
            frame_id=3,
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
            frame_id=4,
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

    assert commands[1].payload["hpr"] == pytest.approx({"h": -8.0, "p": -2.0, "r": 20.0})


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

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state"]
    assert commands[0].payload["visible"] is False
    assert commands[1].payload["interaction_state"] == "idle"


def test_bridge_returns_to_hover_on_release_when_hand_stays_near_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    commands = bridge.process(hover_packet(frame_id=4, timestamp_ms=160, pinch_state="open"))

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_state", "set_object_state"]
    assert commands[-1].payload["interaction_state"] == "pending_grab"


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


def test_bridge_raw_coordinate_transform_ignores_position_sensitivity() -> None:
    bridge = BridgeServiceImpl(position_sensitivity=2.0)

    transformed = bridge._camera_to_world_position(Vec3(0.25, 0.2, -0.3))

    assert transformed == Vec3(-0.25, 0.2, 0.3)


def test_bridge_position_sensitivity_scales_control_coordinates() -> None:
    bridge = BridgeServiceImpl(position_sensitivity=2.0)

    transformed = bridge._camera_to_world_control_position(Vec3(0.25, 0.2, -0.3))

    assert transformed == Vec3(-0.5, 0.4, 0.6)


def test_bridge_position_sensitivity_moves_hand_without_scaling_geometry() -> None:
    bridge = BridgeServiceImpl(position_sensitivity=2.0)
    bridge.start()

    packet = hover_packet(frame_id=1, timestamp_ms=100)
    commands = bridge.process(packet)

    primary_hand_pose = next(
        command for command in commands if command.command_type == "set_hand_pose" and command.object_id == "hand-1"
    )
    points = primary_hand_pose.payload["points"]

    base_anchor = bridge._camera_to_world_position(bridge._interaction_anchor(packet))
    scaled_anchor = bridge._camera_to_world_control_position(bridge._interaction_anchor(packet))

    assert points["anchor"] == pytest.approx(vec3_payload(scaled_anchor))
    assert base_anchor != scaled_anchor
    assert target_object_position().y == pytest.approx(scene_object_position(default_test_object_id()).y)


def test_bridge_emits_secondary_hand_pose_when_secondary_is_present() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    packet = with_secondary_hand(make_packet(frame_id=1, timestamp_ms=100))
    commands = bridge.process(packet)

    hand_pose_commands = [command for command in commands if command.command_type == "set_hand_pose"]
    assert len(hand_pose_commands) == 2
    assert hand_pose_commands[0].object_id == "hand-1"
    assert hand_pose_commands[1].object_id == "hand-2"
    assert hand_pose_commands[1].payload["visible"] is True


def test_bridge_emits_hidden_secondary_hand_pose_when_secondary_slot_clears() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(with_secondary_hand(make_packet(frame_id=1, timestamp_ms=100)))

    packet = make_packet(
        frame_id=2,
        timestamp_ms=120,
        debug={
            "secondary_hand": None,
            "dual_hand": {"secondary_hand": None},
        },
    )
    commands = bridge.process(packet)

    hand_pose_commands = [command for command in commands if command.command_type == "set_hand_pose"]
    assert len(hand_pose_commands) == 2
    assert hand_pose_commands[1].object_id == "hand-2"
    assert hand_pose_commands[1].payload["visible"] is False


def test_bridge_secondary_hand_does_not_trigger_rotation_mode() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    packet = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="open"),
        pinch_state="pinched",
    )

    commands = bridge.process(packet)

    assert all(
        not (command.command_type == "set_object_state" and command.payload.get("interaction_state") == "rotating")
        for command in commands
    )


def test_bridge_dual_scale_emits_scale_without_hpr_and_freezes_position() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    primary_pinched = hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched")
    packet = with_secondary_hand(primary_pinched, pinch_state="pinched")
    commands = bridge.process(packet)

    pose_commands = [command for command in commands if command.command_type == "set_object_pose"]
    assert pose_commands
    scale_payload = pose_commands[-1].payload
    assert "scale" in scale_payload
    assert "hpr" not in scale_payload
    target_position = target_object_position()
    assert scale_payload["position"] == pytest.approx(
        {"x": target_position.x, "y": target_position.y, "z": target_position.z}
    )
    assert scale_payload["debug"]["dual_scale"]["active"] is True


def test_bridge_secondary_pinch_distance_fallback_triggers_dual_scale() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    primary_pinched = hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched")
    packet = with_secondary_hand(
        primary_pinched,
        pinch_state="open",  # Simulate secondary pinch_state jitter/misclassification.
        pinch_distance=0.05,   # Strict fallback threshold should still treat this as pinched.
    )

    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_pose" and "scale" in command.payload
        for command in commands
    )


def test_bridge_dual_scale_accepts_primary_pinch_candidate_from_gesture_service() -> None:
    def make_observation(*, wrist_x: float, handedness: str) -> RawHandObservation:
        return RawHandObservation(
            index_tip=Vec3(wrist_x + 0.04, -0.08, -0.18),
            thumb_tip=Vec3(wrist_x, -0.08, -0.18),
            wrist=Vec3(wrist_x, -0.12, -0.18),
            confidence=0.95,
            raw_pinch_distance=0.04,
            hand_scale=0.30,
            landmarks=[Vec3(0.5, 0.5, 0.0) for _ in range(21)],
            handedness=handedness,
        )

    class FakeCapture:
        def __init__(self, **_: object) -> None:
            self.frames = [
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.uint8),
            ]

        def read(self):
            if not self.frames:
                return None
            return self.frames.pop(0)

        def close(self) -> None:
            return None

    class FakeDualPinchDetector:
        def __init__(self, **_: object) -> None:
            return None

        def detect_multi(self, frame, *, timestamp_ms: int):
            assert frame is not None
            assert timestamp_ms > 0
            return [
                make_observation(wrist_x=0.0, handedness="Right"),
                make_observation(wrist_x=0.04, handedness="Left"),
            ]

        def close(self) -> None:
            return None

    gesture = GestureServiceImpl(
        capture_factory=FakeCapture,
        detector_factory=FakeDualPinchDetector,
        clock=iter([1.0, 1.01]).__next__,
    )
    bridge = BridgeServiceImpl()

    gesture.start()
    bridge.start()

    first_packet = gesture.poll()
    assert first_packet is not None
    assert first_packet.pinch_state == "pinch_candidate"
    assert first_packet.debug is not None
    assert first_packet.debug["secondary_hand"]["pinch_state"] == "pinch_candidate"

    commands = bridge.process(first_packet)

    assert any(
        command.command_type == "set_object_pose" and "scale" in command.payload
        for command in commands
    )


def test_bridge_dual_scale_tracks_distance_during_primary_pinch_candidate_frames() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    first_packet = make_packet(
        frame_id=2,
        timestamp_ms=120,
        pinch_state="pinch_candidate",
        index_tip=Vec3(0.02, -0.08, -0.18),
        thumb_tip=Vec3(-0.02, -0.08, -0.18),
    )
    first_packet.pinch_distance = 0.04
    first_packet = with_secondary_hand(
        first_packet,
        pinch_state="open",
        pinch_distance=0.05,
        index_tip=Vec3(0.06, -0.08, -0.18),
        thumb_tip=Vec3(0.02, -0.08, -0.18),
        wrist=Vec3(0.04, -0.12, -0.18),
    )

    first_commands = bridge.process(first_packet)
    first_pose = [
        command
        for command in first_commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]

    assert first_pose.payload["debug"]["dual_scale"]["distance_xy"] > 0.0
    assert first_pose.payload["debug"]["dual_scale"]["ratio"] == pytest.approx(1.0)

    second_packet = make_packet(
        frame_id=3,
        timestamp_ms=140,
        pinch_state="pinch_candidate",
        index_tip=Vec3(0.00, -0.08, -0.18),
        thumb_tip=Vec3(-0.04, -0.08, -0.18),
    )
    second_packet.pinch_distance = 0.04
    second_packet = with_secondary_hand(
        second_packet,
        pinch_state="open",
        pinch_distance=0.05,
        index_tip=Vec3(0.08, -0.08, -0.18),
        thumb_tip=Vec3(0.04, -0.08, -0.18),
        wrist=Vec3(0.06, -0.12, -0.18),
    )

    second_commands = bridge.process(second_packet)
    second_pose = [
        command
        for command in second_commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]

    assert second_pose.payload["debug"]["dual_scale"]["distance_xy"] > first_pose.payload["debug"]["dual_scale"]["distance_xy"]
    assert second_pose.payload["debug"]["dual_scale"]["ratio"] > first_pose.payload["debug"]["dual_scale"]["ratio"]


def test_bridge_dual_scale_has_no_per_session_ratio_cap() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    baseline_packet = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
        index_tip=Vec3(0.03, -0.08, -0.18),
        thumb_tip=Vec3(0.01, -0.08, -0.18),
    )
    bridge.process(baseline_packet)

    expanded_packet = with_secondary_hand(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            index_tip=Vec3(-0.23, -0.08, -0.18),
            thumb_tip=Vec3(-0.27, -0.08, -0.18),
        ),
        pinch_state="pinched",
        index_tip=Vec3(0.27, -0.08, -0.18),
        thumb_tip=Vec3(0.23, -0.08, -0.18),
    )
    commands = bridge.process(expanded_packet)
    pose = [
        command
        for command in commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]

    ratio = pose.payload["debug"]["dual_scale"]["ratio"]
    assert ratio > 2.8


def test_bridge_dual_scale_uses_xy_distance_only() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    packet1 = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
            index_tip=Vec3(0.06, -0.08, 0.42),
            thumb_tip=Vec3(0.02, -0.08, -0.78),
    )
    commands1 = bridge.process(packet1)
    ratio1 = [command for command in commands1 if command.command_type == "set_object_pose" and "scale" in command.payload][-1].payload["debug"]["dual_scale"]["ratio"]

    packet2 = with_secondary_hand(
        hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"),
        pinch_state="pinched",
        index_tip=Vec3(0.06, -0.08, -0.98),
        thumb_tip=Vec3(0.02, -0.08, 0.62),
    )
    commands2 = bridge.process(packet2)
    ratio2 = [command for command in commands2 if command.command_type == "set_object_pose" and "scale" in command.payload][-1].payload["debug"]["dual_scale"]["ratio"]

    assert ratio1 == pytest.approx(ratio2)


def test_bridge_stops_dual_scale_when_either_hand_releases() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    dual_pinched = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
    )
    commands = bridge.process(dual_pinched)
    assert any(command.command_type == "set_object_pose" and "scale" in command.payload for command in commands)

    primary_open = with_secondary_hand(
        hover_packet(frame_id=3, timestamp_ms=140, pinch_state="open"),
        pinch_state="pinched",
    )
    commands = bridge.process(primary_open)
    assert not any(command.command_type == "set_object_pose" and "scale" in command.payload for command in commands)


def test_bridge_dual_scale_blocks_rotation_mode_even_when_rotation_debug_is_active() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    packet = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
    )
    packet.debug = dict(packet.debug or {})
    packet.debug["rotation"] = {
        "mode_active": True,
        "deg_x": 55.0,
        "deg_y": 30.0,
        "deg_z": -25.0,
    }

    commands = bridge.process(packet)

    pose_commands = [command for command in commands if command.command_type == "set_object_pose"]
    assert pose_commands
    assert any("scale" in command.payload for command in pose_commands)
    assert all("hpr" not in command.payload for command in pose_commands)


def test_bridge_dual_scale_activates_when_only_primary_hovers_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    # Primary hand hovers the object; secondary hand is pinched but far from any object.
    primary_pinched = hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched")
    packet = with_secondary_hand(
        primary_pinched,
        pinch_state="pinched",
        index_tip=Vec3(0.90, 0.90, -0.18),
        thumb_tip=Vec3(0.88, 0.90, -0.18),
        wrist=Vec3(0.90, 0.85, -0.18),
    )
    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_pose" and "scale" in command.payload
        for command in commands
    )


def test_bridge_dual_scale_activates_when_only_secondary_hovers_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    # Primary hand is pinched but far from any object; secondary hand hovers the object.
    far_primary = make_packet(
        frame_id=2,
        timestamp_ms=120,
        pinch_state="pinched",
        index_tip=Vec3(0.90, 0.90, -0.18),
        thumb_tip=Vec3(0.88, 0.90, -0.18),
    )
    packet = with_secondary_hand(
        far_primary,
        pinch_state="pinched",
        index_tip=Vec3(0.02, -0.08, -0.18),
        thumb_tip=Vec3(-0.02, -0.08, -0.18),
        wrist=Vec3(0.0, -0.12, -0.18),
    )
    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_pose" and "scale" in command.payload
        for command in commands
    )


def test_bridge_dual_scale_binds_to_last_interacted_object() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    # Interact with the default object first so it becomes the sticky scale target
    # for the current interaction session.
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    commands = bridge.process(
        with_secondary_hand(
            make_packet(
                frame_id=4,
                timestamp_ms=160,
                pinch_state="pinched",
                index_tip=object_camera_point(0.02, object_id=alternate_test_object_id()),
                thumb_tip=object_camera_point(-0.02, object_id=alternate_test_object_id()),
            ),
            pinch_state="pinched",
            object_id=alternate_test_object_id(),
        )
    )

    scale_pose = [
        command
        for command in commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]
    assert scale_pose.object_id == default_test_object_id()


def test_bridge_rotation_binds_to_last_interacted_object_when_not_hovering() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"))

    far_open = make_packet(
        frame_id=4,
        timestamp_ms=160,
        pinch_state="open",
        index_tip=Vec3(0.70, 0.70, 0.20),
        thumb_tip=Vec3(0.62, 0.62, 0.20),
    )
    bridge.process(far_open)

    commands = bridge.process(
        make_packet(
            frame_id=5,
            timestamp_ms=180,
            pinch_state="pinched",
            index_tip=Vec3(0.72, 0.72, 0.22),
            thumb_tip=Vec3(0.64, 0.64, 0.22),
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

    state_command = next(
        command
        for command in commands
        if command.command_type == "set_object_state"
    )
    pose_command = next(
        command
        for command in commands
        if command.command_type == "set_object_pose"
    )

    assert state_command.object_id == default_test_object_id()
    assert state_command.payload["interaction_state"] == "rotating"
    assert pose_command.object_id == default_test_object_id()
    assert "hpr" in pose_command.payload
    assert "position" not in pose_command.payload


def test_bridge_rotation_does_not_seed_last_interacted_without_direct_grab() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))
    bridge.process(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="pinched",
            index_tip=object_camera_point(0.02),
            thumb_tip=object_camera_point(-0.02),
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

    commands = bridge.process(
        with_secondary_hand(
            make_packet(
                frame_id=4,
                timestamp_ms=160,
                pinch_state="pinched",
                index_tip=object_camera_point(0.02, object_id=alternate_test_object_id()),
                thumb_tip=object_camera_point(-0.02, object_id=alternate_test_object_id()),
            ),
            pinch_state="pinched",
            object_id=alternate_test_object_id(),
        )
    )

    scale_pose = [
        command
        for command in commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]
    assert scale_pose.object_id == alternate_test_object_id()


def test_bridge_two_hand_detected_does_not_block_primary_translation_without_dual_pinch() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))

    commands = bridge.process(
        with_secondary_hand(
            hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"),
            pinch_state="open",
        )
    )

    # Secondary hand alone should not enable dual-scale gating. Primary pinch
    # can still drive normal translation updates.
    assert any(
        command.command_type == "set_object_pose" and "position" in command.payload
        for command in commands
    )


def test_bridge_open_secondary_can_drive_hover_capture_when_primary_misses() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    # Primary hand is away from interactable objects; secondary is near the default test object.
    primary_far = make_packet(
        frame_id=2,
        timestamp_ms=120,
        pinch_state="open",
        index_tip=Vec3(0.70, 0.70, 0.20),
        thumb_tip=Vec3(0.62, 0.62, 0.20),
    )
    packet = with_secondary_hand(
        primary_far,
        pinch_state="open",
        object_id=default_test_object_id(),
    )

    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_state"
        and command.payload.get("interaction_state") == "pending_grab"
        for command in commands
    )


def test_bridge_secondary_pinched_can_grab_and_move_when_primary_open() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    primary_far = make_packet(
        frame_id=2,
        timestamp_ms=120,
        pinch_state="open",
        index_tip=Vec3(0.70, 0.70, 0.20),
        thumb_tip=Vec3(0.62, 0.62, 0.20),
    )
    packet = with_secondary_hand(
        primary_far,
        pinch_state="pinched",
        object_id=default_test_object_id(),
    )

    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_state"
        and command.payload.get("interaction_state") == "grabbed"
        for command in commands
    )
    assert any(
        command.command_type == "set_object_pose" and "position" in command.payload
        for command in commands
    )


def test_bridge_secondary_release_stops_motion_when_primary_open() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    grabbed = with_secondary_hand(
        make_packet(
            frame_id=2,
            timestamp_ms=120,
            pinch_state="open",
            index_tip=Vec3(0.70, 0.70, 0.20),
            thumb_tip=Vec3(0.62, 0.62, 0.20),
        ),
        pinch_state="pinched",
        object_id=default_test_object_id(),
    )
    bridge.process(grabbed)

    released = with_secondary_hand(
        make_packet(
            frame_id=3,
            timestamp_ms=140,
            pinch_state="open",
            index_tip=Vec3(0.70, 0.70, 0.20),
            thumb_tip=Vec3(0.62, 0.62, 0.20),
        ),
        pinch_state="open",
        pinch_distance=0.30,
        object_id=default_test_object_id(),
    )
    commands = bridge.process(released)

    assert any(
        command.command_type == "set_object_state"
        and command.payload.get("interaction_state") == "idle"
        for command in commands
    )
    assert not any(
        command.command_type == "set_object_pose" and "position" in command.payload
        for command in commands
    )


def test_bridge_secondary_can_enter_rotation_mode_when_primary_open() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    packet = with_secondary_hand(
        make_packet(
            frame_id=2,
            timestamp_ms=120,
            pinch_state="open",
            index_tip=Vec3(0.70, 0.70, 0.20),
            thumb_tip=Vec3(0.62, 0.62, 0.20),
        ),
        pinch_state="pinched",
        object_id=default_test_object_id(),
        secondary_debug={
            "rotation": {
                "mode_active": True,
                "deg_x": 15.0,
                "deg_y": -30.0,
                "deg_z": 45.0,
            }
        },
    )

    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_state" and command.payload.get("interaction_state") == "rotating"
        for command in commands
    )
    assert any(
        command.command_type == "set_object_pose" and "hpr" in command.payload
        for command in commands
    )


def test_bridge_two_hand_detected_does_not_block_primary_rotation_without_dual_pinch() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    bridge.process(hover_packet(frame_id=2, timestamp_ms=120))

    packet = with_secondary_hand(
        hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"),
        pinch_state="open",
    )
    packet.debug = dict(packet.debug or {})
    packet.debug["rotation"] = {
        "mode_active": True,
        "deg_x": 33.0,
        "deg_y": 44.0,
        "deg_z": 55.0,
    }
    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_pose" and "hpr" in command.payload
        for command in commands
    )


def test_bridge_dual_scale_ratio_is_absolute_against_initial_scale() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    # Session 1 frame A: establish the per-session baseline.
    baseline_packet = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
    )
    bridge.process(baseline_packet)

    # Session 1 frame B: change distance to create a non-1.0 absolute ratio.
    shrink_packet = with_secondary_hand(
        hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"),
        pinch_state="pinched",
        index_tip=object_camera_point(0.03),
        thumb_tip=object_camera_point(0.01),
    )
    commands = bridge.process(shrink_packet)
    first_pose = [
        command
        for command in commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]
    first_ratio = first_pose.payload["debug"]["dual_scale"]["ratio"]
    assert first_ratio < 1.0

    # Release to end first session.
    bridge.process(
        with_secondary_hand(
            hover_packet(frame_id=4, timestamp_ms=160, pinch_state="open"),
            pinch_state="pinched",
        )
    )

    # Session 2 starts with the same pinch distance as session-1 baseline.
    # Relative ratio would be ~1.0, but absolute ratio must stay at first_ratio.
    restart_packet = with_secondary_hand(
        hover_packet(frame_id=5, timestamp_ms=180, pinch_state="pinched"),
        pinch_state="pinched",
    )
    bridge.process(restart_packet)

    # Session 2 frame B: keep the same changed distance as session-1 frame B.
    # Relative ratio would again be ~1.0 against the new baseline, but absolute
    # ratio should stay aligned with the initial object scale.
    restart_shrink_packet = with_secondary_hand(
        hover_packet(frame_id=6, timestamp_ms=200, pinch_state="pinched"),
        pinch_state="pinched",
        index_tip=object_camera_point(0.03),
        thumb_tip=object_camera_point(0.01),
    )
    commands = bridge.process(restart_shrink_packet)
    second_pose = [
        command
        for command in commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]
    second_ratio = second_pose.payload["debug"]["dual_scale"]["ratio"]

    assert second_ratio == pytest.approx(first_ratio * first_ratio)


def test_bridge_dual_scale_ratio_is_not_reused_when_switching_objects() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))

    # Scale object A away from 1.0 so stale-ratio carryover is visible.
    bridge.process(
        with_secondary_hand(
            hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
            pinch_state="pinched",
        )
    )
    commands_a = bridge.process(
        with_secondary_hand(
            hover_packet(frame_id=3, timestamp_ms=140, pinch_state="pinched"),
            pinch_state="pinched",
            index_tip=object_camera_point(0.03),
            thumb_tip=object_camera_point(0.01),
        )
    )
    ratio_a = [
        command
        for command in commands_a
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1].payload["debug"]["dual_scale"]["ratio"]
    assert ratio_a < 1.0

    # Release first object.
    bridge.process(
        with_secondary_hand(
            hover_packet(frame_id=4, timestamp_ms=160, pinch_state="open"),
            pinch_state="pinched",
        )
    )

    # Start dual-scale on a different object. First frame is that object's own
    # baseline, so absolute ratio must be 1.0 instead of reusing ratio_a.
    moved_primary = make_packet(
        frame_id=5,
        timestamp_ms=180,
        pinch_state="pinched",
        index_tip=object_camera_point(0.02, object_id=alternate_test_object_id()),
        thumb_tip=object_camera_point(-0.02, object_id=alternate_test_object_id()),
    )
    moved_packet = with_secondary_hand(
        moved_primary,
        pinch_state="pinched",
        object_id=alternate_test_object_id(),
    )
    commands_b = bridge.process(moved_packet)
    ratio_b = [
        command
        for command in commands_b
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1].payload["debug"]["dual_scale"]["ratio"]

    assert ratio_b == pytest.approx(1.0)


def test_bridge_dual_scale_does_not_switch_to_another_object_while_pinched() -> None:
    bridge = BridgeServiceImpl()
    bridge.start()

    bridge.process(make_packet(frame_id=1, timestamp_ms=100))
    first_packet = with_secondary_hand(
        hover_packet(frame_id=2, timestamp_ms=120, pinch_state="pinched"),
        pinch_state="pinched",
    )
    first_commands = bridge.process(first_packet)
    first_scale_pose = [
        command
        for command in first_commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ][-1]
    locked_object_id = first_scale_pose.object_id

    # Move both hands near another object while keeping pinch locked; bridge must not
    # switch capture target during the same pinch session.
    moved_primary = make_packet(
        frame_id=3,
        timestamp_ms=140,
        pinch_state="pinched",
        index_tip=Vec3(-0.44, -0.14, -0.02),
        thumb_tip=Vec3(-0.52, -0.14, -0.02),
    )
    moved_packet = with_secondary_hand(
        moved_primary,
        pinch_state="pinched",
        index_tip=Vec3(-0.40, -0.14, -0.02),
        thumb_tip=Vec3(-0.48, -0.14, -0.02),
        wrist=Vec3(-0.44, -0.18, -0.02),
    )

    moved_commands = bridge.process(moved_packet)
    moved_scale_pose = [
        command
        for command in moved_commands
        if command.command_type == "set_object_pose" and "scale" in command.payload
    ]

    # Bridge should keep scaling the locked object instead of switching targets.
    assert moved_scale_pose
    assert all(command.object_id == locked_object_id for command in moved_scale_pose)
    assert all(command.object_id == locked_object_id for command in moved_commands if command.command_type == "set_object_state")
