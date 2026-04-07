from __future__ import annotations

import logging

import numpy as np
import pytest

from src.bridge.service import BridgeServiceImpl
from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import RawHandObservation
from src.gesture.service import GestureServiceImpl


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
    tracking_state: str = "tracked",
    pinch_state: str = "open",
    pinch_distance: float | None = None,
    confidence: float = 0.95,
    index_tip: Vec3 | None = None,
    thumb_tip: Vec3 | None = None,
    wrist: Vec3 | None = None,
    secondary_debug: dict | None = None,
) -> GesturePacket:
    debug = dict(packet.debug or {})
    debug["secondary_hand"] = {
        "hand_id": "hand-2",
        "handedness": "left",
        "tracking_state": tracking_state,
        "confidence": confidence,
        "pinch_state": pinch_state,
        "pinch_distance": pinch_distance,
        "index_tip": {
            "x": (index_tip or Vec3(0.06, -0.08, -0.18)).x,
            "y": (index_tip or Vec3(0.06, -0.08, -0.18)).y,
            "z": (index_tip or Vec3(0.06, -0.08, -0.18)).z,
        },
        "thumb_tip": {
            "x": (thumb_tip or Vec3(0.02, -0.08, -0.18)).x,
            "y": (thumb_tip or Vec3(0.02, -0.08, -0.18)).y,
            "z": (thumb_tip or Vec3(0.02, -0.08, -0.18)).z,
        },
        "wrist": {
            "x": (wrist or Vec3(0.04, -0.12, -0.18)).x,
            "y": (wrist or Vec3(0.04, -0.12, -0.18)).y,
            "z": (wrist or Vec3(0.04, -0.12, -0.18)).z,
        },
        "debug": {} if secondary_debug is None else secondary_debug,
    }
    packet.debug = debug
    return packet


def hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open") -> GesturePacket:
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(0.02, -0.08, -0.18),
        thumb_tip=Vec3(-0.02, -0.08, -0.18),
    )


def offset_hover_packet(*, frame_id: int, timestamp_ms: int, pinch_state: str = "open") -> GesturePacket:
    return make_packet(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        pinch_state=pinch_state,
        index_tip=Vec3(0.10, -0.08, -0.18),
        thumb_tip=Vec3(0.06, -0.08, -0.18),
    )


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
    assert commands[2].payload["position"] == pytest.approx({"x": 0.0, "y": -0.08, "z": 0.18})
    assert commands[2].payload["coordinate_space"] == "world_norm"

    commands = bridge.process(
        make_packet(
            frame_id=4,
            timestamp_ms=160,
            pinch_state="pinched",
            index_tip=Vec3(0.14, -0.08, -0.18),
            thumb_tip=Vec3(0.10, -0.08, -0.18),
        )
    )

    assert [command.command_type for command in commands] == ["set_hand_pose", "set_object_pose"]
    assert commands[1].payload["position"] == pytest.approx({"x": -0.04, "y": -0.08, "z": 0.18})
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
    assert commands[2].payload["hpr"] == pytest.approx({"h": 12.0, "p": 8.0, "r": 0.0})
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
    assert commands[1].payload["hpr"] == pytest.approx({"h": 2.0, "p": 28.0, "r": 15.0})


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
    assert commands[2].payload["hpr"] == pytest.approx({"h": -13.0, "p": 38.0, "r": 40.0})


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

    assert commands[1].payload["hpr"] == pytest.approx({"h": -8.0, "p": 18.0, "r": 20.0})


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
    assert scale_payload["position"] == pytest.approx({"x": 0.0, "y": -0.08, "z": 0.18})
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

    packet = gesture.poll()

    assert packet is not None
    assert packet.pinch_state == "pinch_candidate"
    assert packet.debug is not None
    assert packet.debug["secondary_hand"]["pinch_state"] == "pinch_candidate"

    commands = bridge.process(packet)

    assert any(
        command.command_type == "set_object_pose" and "scale" in command.payload
        for command in commands
    )


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
    # Primary hand is away from interactable objects; secondary is near primary_cube.
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
        index_tip=Vec3(0.06, -0.08, -0.18),
        thumb_tip=Vec3(0.02, -0.08, -0.18),
        wrist=Vec3(0.04, -0.12, -0.18),
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
        index_tip=Vec3(0.06, -0.08, -0.18),
        thumb_tip=Vec3(0.02, -0.08, -0.18),
        wrist=Vec3(0.04, -0.12, -0.18),
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
        index_tip=Vec3(0.06, -0.08, -0.18),
        thumb_tip=Vec3(0.02, -0.08, -0.18),
        wrist=Vec3(0.04, -0.12, -0.18),
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
        index_tip=Vec3(0.06, -0.08, -0.18),
        thumb_tip=Vec3(0.02, -0.08, -0.18),
        wrist=Vec3(0.04, -0.12, -0.18),
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
        index_tip=Vec3(0.02, -0.08, -0.18),
        thumb_tip=Vec3(-0.02, -0.08, -0.18),
        wrist=Vec3(0.0, -0.12, -0.18),
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
        index_tip=Vec3(0.03, -0.08, -0.18),
        thumb_tip=Vec3(0.01, -0.08, -0.18),
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
        index_tip=Vec3(0.03, -0.08, -0.18),
        thumb_tip=Vec3(0.01, -0.08, -0.18),
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
            index_tip=Vec3(0.03, -0.08, -0.18),
            thumb_tip=Vec3(0.01, -0.08, -0.18),
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
