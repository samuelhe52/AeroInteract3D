from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
import time
from typing import Any, Optional

from src.constants import (
    BRIDGE_HEARTBEAT_INTERVAL_FRAMES,
    BRIDGE_MIN_TRACKING_CONFIDENCE,
    BRIDGE_STATE_IDLE,
    BRIDGE_STATE_GRABBING,
    BRIDGE_STATE_PENDING_GRAB,
    BRIDGE_STATE_ROTATING,
    GRAB_RELEASE_DISTANCE_THRESHOLD,
    HOVER_DISTANCE_THRESHOLD,
    INTERACTION_IDLE,
    INTERACTION_GRABBED,
    INTERACTION_PENDING_GRAB,
    INTERACTION_ROTATING,
    MAX_ERROR_HISTORY,
)
from src.contracts import GesturePacket, SceneCommand, Vec3
from src.ports import BridgeService
from src.utils.contracts import EXPECTED_CONTRACT_VERSION, validate_gesture_packet, vec3_payload
from src.utils.runtime import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_INITIALIZING,
    LIFECYCLE_RUNNING,
    LIFECYCLE_STOPPED,
    build_health,
    classify_frame,
    error_entry,
    make_command_id,
)

# Create a dedicated logger for bridge service
logger = logging.getLogger("bridge.service")

# Create a dedicated logger for coordinate transformation
coordinate_logger = logging.getLogger("bridge.coordinate_transformation")

INITIAL_OBJECT_POSITION = Vec3(0.0, 0.0, 0.0)
BRIDGE_SCENE_OBJECT_ID = "scene"
CURRENT_TABLE_SURFACE_Y = -0.05


_TABLETOP_PROP_Y_BY_SHAPE = {
    "apple": 0.032,
    "burger_bun01": -0.05,
    "burger_bun02": -0.05,
    "burger_bun03": -0.05,
    "croissant": -0.048,
    "lemon": 0.041,
    "frame": -0.05,
    "potted_plant": -0.05,
}


def _tabletop_prop_y(shape: str) -> float:
    return _TABLETOP_PROP_Y_BY_SHAPE.get(shape, CURRENT_TABLE_SURFACE_Y)


def _tabletop_collision_half_height(init_y: float) -> float:
    return max(init_y - CURRENT_TABLE_SURFACE_Y, 0.0)


def _scene_collision_surface_y(descriptor: dict[str, Any]) -> float:
    explicit_surface_y = descriptor.get("collision_surface_y")
    if isinstance(explicit_surface_y, int | float):
        return float(explicit_surface_y)
    return float(descriptor["init_pos"]["y"]) + (float(descriptor["scale"]["y"]) * 0.5)


TABLE_SCENE_OBJECTS: tuple[dict[str, Any], ...] = (
    {
        "object_id": "table_plane",
        "init_pos": {"x": 0.0, "y": -1.94, "z": 0.18},
        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "wooden_table",
        "scale": {
            "x": 2.2919410217835164,
            "y": 2.3640589645458817,
            "z": 2.436176907308247,
        },
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "collision_surface_y": CURRENT_TABLE_SURFACE_Y,
        "interactable": False,
    },
    {
        "object_id": "apple_model",
        "init_pos": {"x": -0.12, "y": _tabletop_prop_y("apple"), "z": 0.54},
        "init_hpr": {"h": 12.0, "p": 8.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "apple",
        "scale": {"x": 0.225, "y": 0.225, "z": 0.225},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("apple")),
        "interaction_radius": 0.18,
    },
    {
        "object_id": "bun_left",
        "init_pos": {"x": 0.49, "y": _tabletop_prop_y("burger_bun01"), "z": 0.58},
        "init_hpr": {"h": -12.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "burger_bun01",
        "scale": {"x": 0.1875, "y": 0.1875, "z": 0.1875},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("burger_bun01")),
        "interaction_radius": 0.14,
    },
    {
        "object_id": "bun_center",
        "init_pos": {"x": 0.46, "y": _tabletop_prop_y("burger_bun02"), "z": 0.54},
        "init_hpr": {"h": 6.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "burger_bun02",
        "scale": {"x": 0.1875, "y": 0.1875, "z": 0.1875},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("burger_bun02")),
        "interaction_radius": 0.14,
    },
    {
        "object_id": "bun_right",
        "init_pos": {"x": 0.52, "y": _tabletop_prop_y("burger_bun03"), "z": 0.50},
        "init_hpr": {"h": 16.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "burger_bun03",
        "scale": {"x": 0.1875, "y": 0.1875, "z": 0.1875},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("burger_bun03")),
        "interaction_radius": 0.14,
    },
    {
        "object_id": "croissant_left",
        "init_pos": {"x": -0.30, "y": _tabletop_prop_y("croissant"), "z": -0.21},
        "init_hpr": {"h": -38.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "croissant",
        "scale": {"x": 0.495, "y": 0.495, "z": 0.495},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("croissant")),
        "interaction_radius": 0.22,
    },
    {
        "object_id": "lemon_right",
        "init_pos": {"x": 0.10, "y": _tabletop_prop_y("lemon"), "z": 0.52},
        "init_hpr": {"h": 34.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "lemon",
        "scale": {"x": 0.20625, "y": 0.20625, "z": 0.20625},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("lemon")),
        "interaction_radius": 0.15,
    },
    {
        "object_id": "desk_frame",
        "init_pos": {"x": -0.58, "y": _tabletop_prop_y("frame"), "z": 0.52},
        "init_hpr": {"h": 22.0, "p": -12.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "frame",
        "scale": {"x": 0.24, "y": 0.24, "z": 0.24},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("frame")),
        "interaction_radius": 0.18,
    },
    {
        "object_id": "plant_back_left",
        "init_pos": {"x": 0.48, "y": _tabletop_prop_y("potted_plant"), "z": 0.04},
        "init_hpr": {"h": 24.0, "p": 0.0, "r": 0.0},
        "coordinate_space": "world_norm",
        "interaction_state": INTERACTION_IDLE,
        "shape": "potted_plant",
        "scale": {"x": 0.36, "y": 0.36, "z": 0.36},
        "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
        "interactable": True,
        "collision_half_height": _tabletop_collision_half_height(_tabletop_prop_y("potted_plant")),
        "interaction_radius": 0.20,
    },
)
TABLE_SURFACE_Y = _scene_collision_surface_y(TABLE_SCENE_OBJECTS[0])
DEFAULT_INTERACTABLE_OBJECT_SCALE = (0.22, 0.22, 0.22)


@dataclass(slots=True)
class BridgeMetrics:
    packets_seen: int = 0
    commands_emitted: int = 0
    duplicate_packets: int = 0
    stale_packets: int = 0
    rejected_packets: int = 0
    resets_emitted: int = 0
    pose_updates: int = 0


@dataclass(slots=True)
class ObjectInteractionState:
    object_id: str
    world_position: Vec3
    interaction_radius: float = HOVER_DISTANCE_THRESHOLD
    half_height: float = 0.1
    world_scale: tuple[float, float, float] = DEFAULT_INTERACTABLE_OBJECT_SCALE
    initial_world_scale: tuple[float, float, float] = DEFAULT_INTERACTABLE_OBJECT_SCALE
    absolute_scale_ratio: float = 1.0
    world_hpr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    interaction_state: str = BRIDGE_STATE_IDLE
    grab_offset_world: Vec3 | None = None
    rotation_reference_hpr: tuple[float, float, float] | None = None
    rotation_reference_input: tuple[float, float, float] | None = None
    initialized: bool = False


@dataclass(slots=True)
class SecondaryHandState:
    tracking_state: str
    confidence: float
    pinch_state: str
    pinch_distance: float | None
    index_tip: Vec3
    thumb_tip: Vec3
    wrist: Vec3
    rotation: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


DUAL_SCALE_RATIO_EXPONENT = 0.85
DUAL_SCALE_RATIO_EPSILON = 1e-6
PRIMARY_PINCH_FALLBACK_ENTER_DISTANCE = 0.12
SECONDARY_PINCH_FALLBACK_ENTER_DISTANCE = 0.12
SECONDARY_PINCH_FALLBACK_RELEASE_DISTANCE = 0.24


class BridgeServiceImpl(BridgeService):
    def __init__(
        self,
        *,
        input_mirrored: bool = True,
        position_sensitivity: float = 1.0,
        rotation_sensitivity: float = 1.0,
    ) -> None:
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._expected_contract_version = EXPECTED_CONTRACT_VERSION
        self._input_mirrored = bool(input_mirrored)
        self._position_sensitivity = max(float(position_sensitivity), 0.001)
        self._rotation_sensitivity = max(float(rotation_sensitivity), 0.001)
        self._interaction_state = BRIDGE_STATE_IDLE
        self._last_frame_id: int | None = None
        self._last_timestamp_ms: int | None = None
        self._errors: list[dict[str, Any]] = []
        self._metrics = BridgeMetrics()
        self._pending_init = False
        self._object_states: dict[str, ObjectInteractionState] = {}
        self._hovered_object_id: str | None = None
        self._grabbed_object_id: str | None = None
        self._rotation_object_id: str | None = None
        self._last_interacted_object_id: str | None = None
        self._dual_scale_active = False
        self._dual_scale_object_id: str | None = None
        self._dual_scale_baseline_distance_xy: float | None = None
        self._dual_scale_baseline_scale: tuple[float, float, float] | None = None
        self._dual_scale_ratio: float = 1.0
        self._pinch_capture_lock_object_id: str | None = None
        self._dual_scale_rotation_blocked_until_open: bool = False

    def start(self) -> None:
        if self.lifecycle_state == LIFECYCLE_RUNNING:
            return None

        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._interaction_state = BRIDGE_STATE_IDLE
        self._last_frame_id = None
        self._last_timestamp_ms = None
        self._errors = []
        self._metrics = BridgeMetrics()
        self._pending_init = True
        self._object_states = {}
        self._hovered_object_id = None
        self._grabbed_object_id = None
        self._rotation_object_id = None
        self._last_interacted_object_id = None
        self._dual_scale_active = False
        self._dual_scale_object_id = None
        self._dual_scale_baseline_distance_xy = None
        self._dual_scale_baseline_scale = None
        self._dual_scale_ratio = 1.0
        self._pinch_capture_lock_object_id = None
        self._dual_scale_rotation_blocked_until_open = False
        self.lifecycle_state = LIFECYCLE_RUNNING
        return None

    def process(self, packet: GesturePacket) -> list[SceneCommand]:
        if self.lifecycle_state not in {LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED}:
            raise RuntimeError("Bridge must be running before processing packets")

        self._metrics.packets_seen += 1
        commands: list[SceneCommand] = []

        frame_status = classify_frame(self._last_frame_id, packet.frame_id)
        if frame_status == "duplicate":
            self._metrics.duplicate_packets += 1
            self._record_error(
                error_entry(
                    "bridge.packet.duplicate",
                    "Ignoring duplicate gesture packet",
                    recoverable=True,
                    hint="Ensure frame_id is incremented exactly once per produced packet.",
                    details={"frame_id": packet.frame_id},
                )
            )
            return commands

        if frame_status == "stale":
            self._metrics.stale_packets += 1
            self._record_error(
                error_entry(
                    "bridge.packet.stale",
                    "Ignoring stale gesture packet",
                    recoverable=True,
                    hint="Do not replay older frames into the live bridge pipeline.",
                    details={"frame_id": packet.frame_id, "last_frame_id": self._last_frame_id},
                )
            )
            return commands

        packet_errors = validate_gesture_packet(
            packet,
            expected_version=self._expected_contract_version,
        )
        if self._last_timestamp_ms is not None and packet.timestamp_ms < self._last_timestamp_ms:
            packet_errors.append(
                error_entry(
                    "bridge.packet.timestamp.stale",
                    "Ignoring packet with stale timestamp",
                    recoverable=True,
                    hint="Emit a monotonic timestamp for every gesture packet.",
                    details={"timestamp_ms": packet.timestamp_ms, "last_timestamp_ms": self._last_timestamp_ms},
                )
            )

        if packet_errors:
            self._metrics.rejected_packets += 1
            for packet_error in packet_errors:
                self._record_error(packet_error)
            self.lifecycle_state = LIFECYCLE_DEGRADED
            return commands

        self.lifecycle_state = LIFECYCLE_RUNNING
        self._last_frame_id = packet.frame_id
        self._last_timestamp_ms = packet.timestamp_ms

        if self._pending_init:
            commands.append(self._make_init_scene(packet))
            self._pending_init = False

        commands.append(self._make_hand_pose(packet))
        secondary_hand_pose = self._make_secondary_hand_pose(packet)
        if secondary_hand_pose is not None:
            commands.append(secondary_hand_pose)
        commands.extend(self._step_state_machine(packet))
        if packet.frame_id % BRIDGE_HEARTBEAT_INTERVAL_FRAMES == 0:
            commands.append(self._make_heartbeat(packet))

        self._metrics.commands_emitted += len(commands)
        return commands

    def health(self) -> dict[str, Any]:
        return build_health(
            component="bridge",
            lifecycle_state=self.lifecycle_state,
            errors=self._errors,
            stats={
                "interaction_state": self._interaction_state,
                "last_frame_id": self._last_frame_id,
                "last_timestamp_ms": self._last_timestamp_ms,
                "pending_init": self._pending_init,
                "dual_scale_active": self._dual_scale_active,
                "dual_scale_ratio": self._dual_scale_ratio,
                "object_scale_ratios": {
                    object_id: object_state.absolute_scale_ratio
                    for object_id, object_state in self._object_states.items()
                },
                "packets_seen": self._metrics.packets_seen,
                "commands_emitted": self._metrics.commands_emitted,
                "duplicate_packets": self._metrics.duplicate_packets,
                "stale_packets": self._metrics.stale_packets,
                "rejected_packets": self._metrics.rejected_packets,
                "resets_emitted": self._metrics.resets_emitted,
                "pose_updates": self._metrics.pose_updates,
            },
        )

    def stop(self) -> None:
        self._pending_init = False
        self._interaction_state = BRIDGE_STATE_IDLE
        self._object_states = {}
        self._hovered_object_id = None
        self._grabbed_object_id = None
        self._rotation_object_id = None
        self._last_interacted_object_id = None
        self._dual_scale_active = False
        self._dual_scale_object_id = None
        self._dual_scale_baseline_distance_xy = None
        self._dual_scale_baseline_scale = None
        self._dual_scale_ratio = 1.0
        self._pinch_capture_lock_object_id = None
        self._dual_scale_rotation_blocked_until_open = False
        self.lifecycle_state = LIFECYCLE_STOPPED
        return None

    def _step_state_machine(self, packet: GesturePacket) -> list[SceneCommand]:
        commands: list[SceneCommand] = []
        secondary_hand = self._secondary_hand_state(packet)

        primary_available = packet.tracking_state == "tracked" and packet.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE
        secondary_detected = self._secondary_detected(secondary_hand)
        secondary_available = (
            secondary_detected
            and secondary_hand is not None
            and secondary_hand.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE
        )
        secondary_pinched = self._secondary_is_pinched(secondary_hand)
        primary_dual_scale_pinched = self._primary_allows_dual_scale(packet)

        if not primary_available and not secondary_available:
            if self._grabbed_object_id is not None or self._hovered_object_id is not None:
                return self._release_interaction(packet)
            self._stop_dual_scale_mode()
            return commands

        hand_anchor_world = self._camera_to_world_control_position(self._interaction_anchor(packet)) if primary_available else Vec3(0.0, 0.0, 0.0)
        hovered_object = self._select_hovered_object(hand_anchor_world) if primary_available else None
        hovered_object_id = hovered_object.object_id if hovered_object is not None else None
        secondary_anchor_world = None
        if secondary_hand is not None:
            secondary_anchor_world = self._camera_to_world_control_position(
                self._anchor_from_tips(secondary_hand.index_tip, secondary_hand.thumb_tip)
            )
        secondary_hovered_object_id = self._secondary_hovered_object_id(secondary_hand) if secondary_available else None

        # Single-hand control arbitration: if primary is open/unavailable and the
        # secondary hand is pinched, let secondary drive grab/move exactly like primary.
        controls_from_secondary = (
            secondary_available
            and secondary_pinched
            and (not primary_available or not primary_dual_scale_pinched)
            and not self._dual_scale_active
        )
        if controls_from_secondary and secondary_anchor_world is not None:
            hand_anchor_world = secondary_anchor_world
            hovered_object_id = secondary_hovered_object_id
            hovered_object = self._object_state(secondary_hovered_object_id) if secondary_hovered_object_id is not None else None
            effective_debug = dict(packet.debug or {})
            if isinstance(secondary_hand.rotation, dict):
                effective_debug["rotation"] = secondary_hand.rotation
            packet = replace(
                packet,
                hand_id="hand-2",
                tracking_state="tracked",
                confidence=secondary_hand.confidence,
                pinch_state="pinched",
                index_tip=secondary_hand.index_tip,
                thumb_tip=secondary_hand.thumb_tip,
                wrist=secondary_hand.wrist,
                rotation=secondary_hand.rotation,
                debug=effective_debug,
            )

        if packet.pinch_state == "open":
            self._pinch_capture_lock_object_id = None
            self._dual_scale_rotation_blocked_until_open = False

        dual_scale_gate_active = (
            primary_available
            and secondary_available
            and primary_dual_scale_pinched
            and secondary_pinched
        )

        # Hard gate: while both hands are available and both are pinched, disable
        # rotation and translation entirely. Only dual-scale path is allowed.
        if dual_scale_gate_active:
            if self._rotation_object_id is not None:
                rotating_object = self._object_state(self._rotation_object_id)
                self._rotation_object_id = None
                next_state = BRIDGE_STATE_GRABBING if packet.pinch_state == "pinched" else BRIDGE_STATE_IDLE
                commands.extend(self._set_object_interaction_state(packet, rotating_object, next_state))

        # Dual-scale object lock: once scaling starts, remain bound to the same object
        # until pinch opens, and suppress any rotation/translation side effects while
        # pinched after scaling interruption.
        if self._dual_scale_active and self._dual_scale_object_id is not None:
            lock_matches = (
                dual_scale_gate_active
                and secondary_hand is not None
                and self._secondary_is_pinched(secondary_hand)
                and secondary_hand.tracking_state == "tracked"
                and secondary_hand.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE
            )
            if lock_matches:
                target_object = self._object_state(self._dual_scale_object_id)
                commands.extend(
                    self._handle_dual_scale_mode(
                        packet,
                        target_object,
                        hand_anchor_world,
                        secondary_hand,
                    )
                )
                return commands

            self._stop_dual_scale_mode()
            self._dual_scale_rotation_blocked_until_open = packet.pinch_state == "pinched"
            if packet.pinch_state == "open" and self._grabbed_object_id is not None:
                released_object = self._object_state(self._grabbed_object_id)
                self._grabbed_object_id = None
                self._pinch_capture_lock_object_id = None
                self._last_interacted_object_id = None
                commands.extend(self._set_object_interaction_state(packet, released_object, BRIDGE_STATE_IDLE))
                self._hovered_object_id = None
                commands.extend(self._sync_hover_state(packet, hovered_object_id))
                return commands
            if self._grabbed_object_id is not None or dual_scale_gate_active:
                # Keep object frozen and do not allow mode switching while still pinched.
                return commands

        dual_scale_target = None
        if dual_scale_gate_active:
            dual_scale_target = self._dual_scale_target_object(
                packet,
                hovered_object_id=hovered_object_id,
                secondary_hovered_object_id=secondary_hovered_object_id,
                secondary_hand=secondary_hand,
            )
        if dual_scale_target is not None:
            commands.extend(
                self._handle_dual_scale_mode(
                    packet,
                    dual_scale_target,
                    hand_anchor_world,
                    secondary_hand,
                )
            )
            return commands

        self._stop_dual_scale_mode()
        if dual_scale_gate_active:
            if packet.pinch_state == "pinched" and self._grabbed_object_id is None and hovered_object_id is not None:
                self._grabbed_object_id = hovered_object_id
                self._hovered_object_id = hovered_object_id
                self._pinch_capture_lock_object_id = hovered_object_id
                commands.extend(
                    self._set_object_interaction_state(
                        packet,
                        self._object_state(hovered_object_id),
                        BRIDGE_STATE_GRABBING,
                    )
                )
            # Freeze while two hands are present: no rotation and no translation.
            return commands

        # When both hands are tracked and both are open, allow either hand to
        # drive hover capture so pending-grab feels symmetric.
        both_open = secondary_detected and packet.pinch_state == "open" and not secondary_pinched
        if both_open and hovered_object_id is None and secondary_hovered_object_id is not None and secondary_anchor_world is not None:
            hand_anchor_world = secondary_anchor_world
            hovered_object_id = secondary_hovered_object_id
            hovered_object = self._object_state(secondary_hovered_object_id)

        if self._dual_scale_rotation_blocked_until_open and packet.pinch_state == "pinched":
            return commands

        rotation_mode_active = self._rotation_mode_active(packet)

        if rotation_mode_active:
            target_object = self._rotation_target_object(hovered_object_id)
            if target_object is None:
                return self._sync_hover_state(packet, hovered_object_id)
            commands.extend(
                self._handle_rotation_mode(
                    packet,
                    target_object,
                    hovered_object_id == target_object.object_id,
                    hand_anchor_world,
                )
            )
            return commands

        if self._grabbed_object_id is None:
            commands.extend(self._sync_hover_state(packet, hovered_object_id))
            if hovered_object is None:
                return commands
            if packet.pinch_state == "pinched":
                if self._pinch_capture_lock_object_id is None:
                    self._pinch_capture_lock_object_id = hovered_object.object_id
                if hovered_object.object_id != self._pinch_capture_lock_object_id:
                    return commands
                self._grabbed_object_id = hovered_object.object_id
                self._hovered_object_id = hovered_object.object_id
                hovered_object.grab_offset_world = self._subtract_vec3(hovered_object.world_position, hand_anchor_world)
                commands.extend(
                    self._set_object_interaction_state(packet, hovered_object, BRIDGE_STATE_GRABBING)
                )
                commands.append(self._make_object_pose(packet, hovered_object, hand_anchor_world))
            return commands

        grabbed_object = self._object_state(self._grabbed_object_id)
        self._pinch_capture_lock_object_id = grabbed_object.object_id
        if packet.pinch_state == "open":
            self._grabbed_object_id = None
            self._last_interacted_object_id = None
            commands.extend(self._set_object_interaction_state(packet, grabbed_object, BRIDGE_STATE_IDLE))
            self._hovered_object_id = None
            commands.extend(self._sync_hover_state(packet, hovered_object_id))
            return commands

        constrained_position, blocked_by_table = self._drag_world_position(packet, grabbed_object, hand_anchor_world)
        if blocked_by_table and self._distance(hand_anchor_world, constrained_position) >= GRAB_RELEASE_DISTANCE_THRESHOLD:
            self._grabbed_object_id = None
            grabbed_object.grab_offset_world = None
            self._last_interacted_object_id = None
            commands.extend(self._set_object_interaction_state(packet, grabbed_object, BRIDGE_STATE_IDLE))
            self._hovered_object_id = None
            commands.extend(self._sync_hover_state(packet, hovered_object_id))
            return commands

        commands.append(self._make_object_pose(packet, grabbed_object, hand_anchor_world))
        return commands

    def _reset_interaction(self, packet: GesturePacket, *, reason: str) -> list[SceneCommand]:
        commands = [self._make_reset_interaction(packet, reason=reason)]
        for object_state in self._object_states.values():
            if object_state.interaction_state != BRIDGE_STATE_IDLE:
                object_state.interaction_state = BRIDGE_STATE_IDLE
                object_state.grab_offset_world = None
                object_state.rotation_reference_hpr = None
                object_state.rotation_reference_input = None
                commands.append(self._make_object_state(packet, object_state.object_id, INTERACTION_IDLE))
            else:
                object_state.grab_offset_world = None
                object_state.rotation_reference_hpr = None
                object_state.rotation_reference_input = None
        self._hovered_object_id = None
        self._grabbed_object_id = None
        self._rotation_object_id = None
        self._last_interacted_object_id = None
        self._pinch_capture_lock_object_id = None
        self._dual_scale_rotation_blocked_until_open = False
        self._stop_dual_scale_mode()
        self._interaction_state = BRIDGE_STATE_IDLE
        self._metrics.resets_emitted += 1
        return commands

    def _release_interaction(self, packet: GesturePacket) -> list[SceneCommand]:
        commands: list[SceneCommand] = []
        for object_state in self._object_states.values():
            if object_state.interaction_state != BRIDGE_STATE_IDLE:
                commands.extend(self._set_object_interaction_state(packet, object_state, BRIDGE_STATE_IDLE))
            object_state.grab_offset_world = None
            object_state.rotation_reference_hpr = None
            object_state.rotation_reference_input = None
        self._hovered_object_id = None
        self._grabbed_object_id = None
        self._rotation_object_id = None
        self._last_interacted_object_id = None
        self._pinch_capture_lock_object_id = None
        self._dual_scale_rotation_blocked_until_open = False
        self._stop_dual_scale_mode()
        self._interaction_state = BRIDGE_STATE_IDLE
        return commands

    def _make_init_scene(self, packet: GesturePacket) -> SceneCommand:
        self._object_states = {}
        self._hovered_object_id = None
        self._grabbed_object_id = None
        self._rotation_object_id = None
        self._last_interacted_object_id = None
        self._pinch_capture_lock_object_id = None
        self._dual_scale_rotation_blocked_until_open = False
        self._stop_dual_scale_mode()
        for descriptor in TABLE_SCENE_OBJECTS:
            if not bool(descriptor.get("interactable", True)):
                continue
            object_state = self._ensure_object_state(
                str(descriptor["object_id"]),
                world_position=Vec3(
                    x=float(descriptor["init_pos"]["x"]),
                    y=float(descriptor["init_pos"]["y"]),
                    z=float(descriptor["init_pos"]["z"]),
                ),
                interaction_radius=float(descriptor.get("interaction_radius", HOVER_DISTANCE_THRESHOLD)),
                half_height=float(descriptor.get("collision_half_height", float(descriptor["scale"]["y"]) * 0.5)),
                world_scale=(
                    float(descriptor["scale"]["x"]),
                    float(descriptor["scale"]["y"]),
                    float(descriptor["scale"]["z"]),
                ),
                interaction_state=BRIDGE_STATE_IDLE,
                initialized=True,
            )
            object_state.world_hpr = (
                float(descriptor["init_hpr"]["h"]),
                float(descriptor["init_hpr"]["p"]),
                float(descriptor["init_hpr"]["r"]),
            )
            object_state.grab_offset_world = None
            object_state.rotation_reference_hpr = None
            object_state.rotation_reference_input = None
            object_state.initialized = True
        self._interaction_state = BRIDGE_STATE_IDLE
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("init-scene", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="init_scene",
            object_id=BRIDGE_SCENE_OBJECT_ID,
            payload={"objects": [dict(scene_object) for scene_object in TABLE_SCENE_OBJECTS]},
        )

    def _make_object_pose(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        hand_anchor_world: Vec3 | None = None,
    ) -> SceneCommand:
        self._metrics.pose_updates += 1
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-pose", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_object_pose",
            object_id=object_state.object_id,
            payload=self._pose_payload(packet, object_state, hand_anchor_world),
        )

    def _pose_payload(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        hand_anchor_world: Vec3 | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"coordinate_space": "world_norm"}
        rotation_hpr = self._rotation_hpr_payload(packet, object_state)
        if rotation_hpr is not None:
            payload["hpr"] = rotation_hpr
            return payload

        world_position, _ = self._drag_world_position(packet, object_state, hand_anchor_world)
        object_state.world_position = world_position
        payload["position"] = vec3_payload(world_position)
        return payload

    def _make_hand_pose(self, packet: GesturePacket) -> SceneCommand:
        payload: dict[str, Any] = {
            "coordinate_space": "world_norm",
            "visible": False,
        }
        if packet.tracking_state == "tracked" and packet.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE:
            anchor_world = self._camera_to_world_position(self._interaction_anchor(packet))
            sensitivity_offset = self._subtract_vec3(
                self._camera_to_world_control_position(self._interaction_anchor(packet)),
                anchor_world,
            )
            index_tip_world = self._translate_world_position(
                self._camera_to_world_position(packet.index_tip),
                sensitivity_offset,
            )
            thumb_tip_world = self._translate_world_position(
                self._camera_to_world_position(packet.thumb_tip),
                sensitivity_offset,
            )
            wrist_world = self._translate_world_position(
                self._camera_to_world_position(packet.wrist),
                sensitivity_offset,
            )
            anchor_world = self._translate_world_position(anchor_world, sensitivity_offset)
            thumb_base_world, index_base_world = self._finger_base_points(
                wrist_world,
                anchor_world,
                thumb_tip_world,
                index_tip_world,
            )
            payload["visible"] = True
            payload["points"] = {
                "wrist": vec3_payload(wrist_world),
                "thumb_tip": vec3_payload(thumb_tip_world),
                "index_tip": vec3_payload(index_tip_world),
                "anchor": vec3_payload(anchor_world),
                "thumb_base": vec3_payload(thumb_base_world),
                "index_base": vec3_payload(index_base_world),
            }

        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-hand-pose", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_hand_pose",
            object_id=packet.hand_id,
            payload=payload,
        )

    def _make_secondary_hand_pose(self, packet: GesturePacket) -> SceneCommand | None:
        secondary_hand = self._secondary_hand_state(packet)
        if secondary_hand is None and not self._has_secondary_hand_slot(packet):
            return None

        payload: dict[str, Any] = {
            "coordinate_space": "world_norm",
            "visible": False,
        }
        if secondary_hand is not None and (
            secondary_hand.tracking_state == "tracked"
            and secondary_hand.confidence >= BRIDGE_MIN_TRACKING_CONFIDENCE
        ):
            anchor_world = self._camera_to_world_position(self._anchor_from_tips(secondary_hand.index_tip, secondary_hand.thumb_tip))
            sensitivity_offset = self._subtract_vec3(
                self._camera_to_world_control_position(
                    self._anchor_from_tips(secondary_hand.index_tip, secondary_hand.thumb_tip)
                ),
                anchor_world,
            )
            index_tip_world = self._translate_world_position(
                self._camera_to_world_position(secondary_hand.index_tip),
                sensitivity_offset,
            )
            thumb_tip_world = self._translate_world_position(
                self._camera_to_world_position(secondary_hand.thumb_tip),
                sensitivity_offset,
            )
            wrist_world = self._translate_world_position(
                self._camera_to_world_position(secondary_hand.wrist),
                sensitivity_offset,
            )
            anchor_world = self._translate_world_position(anchor_world, sensitivity_offset)
            thumb_base_world, index_base_world = self._finger_base_points(
                wrist_world,
                anchor_world,
                thumb_tip_world,
                index_tip_world,
            )
            payload["visible"] = True
            payload["points"] = {
                "wrist": vec3_payload(wrist_world),
                "thumb_tip": vec3_payload(thumb_tip_world),
                "index_tip": vec3_payload(index_tip_world),
                "anchor": vec3_payload(anchor_world),
                "thumb_base": vec3_payload(thumb_base_world),
                "index_base": vec3_payload(index_base_world),
            }

        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-hand-pose-secondary", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_hand_pose",
            object_id="hand-2",
            payload=payload,
        )

    def _has_secondary_hand_slot(self, packet: GesturePacket) -> bool:
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return False
        if "secondary_hand" in debug_payload:
            return True
        dual_hand = debug_payload.get("dual_hand")
        return isinstance(dual_hand, dict) and "secondary_hand" in dual_hand

    def _make_dual_scale_pose(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        *,
        ratio: float,
        distance_xy: float,
    ) -> SceneCommand:
        self._metrics.pose_updates += 1
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-pose-scale", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_object_pose",
            object_id=object_state.object_id,
            payload={
                "coordinate_space": "world_norm",
                # Keep position frozen while dual-scale is active.
                "position": vec3_payload(object_state.world_position),
                "scale": {
                    "x": object_state.world_scale[0],
                    "y": object_state.world_scale[1],
                    "z": object_state.world_scale[2],
                },
                "debug": {
                    "dual_scale": {
                        "active": True,
                        "ratio": ratio,
                        "distance_xy": distance_xy,
                    }
                },
            },
        )

    def _finger_base_points(
        self,
        wrist_world: Vec3,
        anchor_world: Vec3,
        thumb_tip_world: Vec3,
        index_tip_world: Vec3,
    ) -> tuple[Vec3, Vec3]:
        palm_pull = self._scale_vec3(self._subtract_vec3(wrist_world, anchor_world), 0.34)
        thumb_forward = self._scale_vec3(
            self._normalized_vec3(self._subtract_vec3(thumb_tip_world, anchor_world)),
            0.085,
        )
        index_forward = self._scale_vec3(
            self._normalized_vec3(self._subtract_vec3(index_tip_world, anchor_world)),
            0.085,
        )
        thumb_base = self._add_vec3(anchor_world, self._add_vec3(palm_pull, thumb_forward))
        index_base = self._add_vec3(anchor_world, self._add_vec3(palm_pull, index_forward))
        return thumb_base, index_base

    @staticmethod
    def _anchor_from_tips(index_tip: Vec3, thumb_tip: Vec3) -> Vec3:
        return Vec3(
            x=(index_tip.x + thumb_tip.x) * 0.5,
            y=(index_tip.y + thumb_tip.y) * 0.5,
            z=(index_tip.z + thumb_tip.z) * 0.5,
        )

    @staticmethod
    def _vec3_from_payload(payload: dict[str, Any] | None) -> Vec3 | None:
        if not isinstance(payload, dict):
            return None
        try:
            return Vec3(float(payload["x"]), float(payload["y"]), float(payload["z"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _secondary_hand_payload(self, packet: GesturePacket) -> dict[str, Any] | None:
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return None
        secondary = debug_payload.get("secondary_hand")
        if isinstance(secondary, dict):
            return secondary
        dual_hand = debug_payload.get("dual_hand")
        if isinstance(dual_hand, dict):
            nested = dual_hand.get("secondary_hand")
            if isinstance(nested, dict):
                return nested
        return None

    def _secondary_hand_state(self, packet: GesturePacket) -> SecondaryHandState | None:
        payload = self._secondary_hand_payload(packet)
        if payload is None:
            return None

        index_tip = self._vec3_from_payload(payload.get("index_tip"))
        thumb_tip = self._vec3_from_payload(payload.get("thumb_tip"))
        wrist = self._vec3_from_payload(payload.get("wrist"))
        if index_tip is None or thumb_tip is None or wrist is None:
            return None

        tracking_state = str(payload.get("tracking_state", "not_detected"))
        pinch_state = str(payload.get("pinch_state", "open")).lower()
        pinch_distance_raw = payload.get("pinch_distance")
        pinch_distance: float | None = None
        if isinstance(pinch_distance_raw, (int, float)):
            pinch_distance = float(pinch_distance_raw)
        if pinch_distance is not None:
            if pinch_state == "pinched" and pinch_distance >= SECONDARY_PINCH_FALLBACK_RELEASE_DISTANCE:
                pinch_state = "open"
            elif pinch_state != "pinched" and pinch_distance <= SECONDARY_PINCH_FALLBACK_ENTER_DISTANCE:
                pinch_state = "pinched"
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        rotation_payload = payload.get("rotation")
        rotation = rotation_payload if isinstance(rotation_payload, dict) else None
        debug_payload = payload.get("debug")
        debug = debug_payload if isinstance(debug_payload, dict) else None
        if rotation is None and debug is not None:
            nested_rotation = debug.get("rotation")
            if isinstance(nested_rotation, dict):
                rotation = nested_rotation
        return SecondaryHandState(
            tracking_state=tracking_state,
            confidence=confidence,
            pinch_state=pinch_state,
            pinch_distance=pinch_distance,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            rotation=rotation,
            debug=debug,
        )

    def _secondary_hovered_object_id(self, secondary_hand: SecondaryHandState | None) -> str | None:
        if secondary_hand is None:
            return None
        if (
            secondary_hand.tracking_state != "tracked"
            or secondary_hand.confidence < BRIDGE_MIN_TRACKING_CONFIDENCE
        ):
            return None
        secondary_anchor_world = self._camera_to_world_control_position(
            self._anchor_from_tips(secondary_hand.index_tip, secondary_hand.thumb_tip)
        )
        secondary_hovered = self._select_hovered_object(secondary_anchor_world)
        return None if secondary_hovered is None else secondary_hovered.object_id

    @staticmethod
    def _secondary_detected(secondary_hand: SecondaryHandState | None) -> bool:
        if secondary_hand is None:
            return False
        if secondary_hand.tracking_state != "tracked":
            return False
        return True

    def _secondary_is_pinched(self, secondary_hand: SecondaryHandState | None) -> bool:
        if secondary_hand is None:
            return False
        if secondary_hand.pinch_state == "pinched":
            return True
        if secondary_hand.tracking_state != "tracked":
            return False
        if secondary_hand.confidence < BRIDGE_MIN_TRACKING_CONFIDENCE:
            return False
        if secondary_hand.pinch_distance is None:
            return False
        return secondary_hand.pinch_distance <= SECONDARY_PINCH_FALLBACK_ENTER_DISTANCE

    @staticmethod
    def _primary_allows_dual_scale(packet: GesturePacket) -> bool:
        if packet.pinch_state == "pinched":
            return True
        if packet.pinch_state != "pinch_candidate":
            return False
        if packet.tracking_state != "tracked":
            return False
        if packet.confidence < BRIDGE_MIN_TRACKING_CONFIDENCE:
            return False
        if packet.pinch_distance is None:
            return False
        return packet.pinch_distance <= PRIMARY_PINCH_FALLBACK_ENTER_DISTANCE

    def _dual_scale_target_object(
        self,
        packet: GesturePacket,
        *,
        hovered_object_id: str | None,
        secondary_hovered_object_id: str | None,
        secondary_hand: SecondaryHandState | None,
    ) -> ObjectInteractionState | None:
        if not self._primary_allows_dual_scale(packet):
            return None
        if secondary_hand is None or not self._secondary_is_pinched(secondary_hand):
            return None
        if secondary_hand.tracking_state != "tracked" or secondary_hand.confidence < BRIDGE_MIN_TRACKING_CONFIDENCE:
            return None

        # Prefer the last object the user actually interacted with, then fall
        # back to the current hover target when no sticky interaction target exists.
        if self._last_interacted_object_id is not None:
            return self._object_state(self._last_interacted_object_id)

        target_id = hovered_object_id or secondary_hovered_object_id
        if target_id is None:
            return None
        return self._object_state(target_id)

    @staticmethod
    def _distance_xy(a: Vec3, b: Vec3) -> float:
        return math.sqrt(((a.x - b.x) ** 2) + ((a.y - b.y) ** 2))

    def _stop_dual_scale_mode(self) -> None:
        self._dual_scale_active = False
        self._dual_scale_object_id = None
        self._dual_scale_baseline_distance_xy = None
        self._dual_scale_baseline_scale = None

    def _handle_dual_scale_mode(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        primary_anchor_world: Vec3,
        secondary_hand: SecondaryHandState | None,
    ) -> list[SceneCommand]:
        if secondary_hand is None:
            self._stop_dual_scale_mode()
            return []

        secondary_anchor_world = self._camera_to_world_control_position(
            self._anchor_from_tips(secondary_hand.index_tip, secondary_hand.thumb_tip)
        )
        distance_xy = self._distance_xy(primary_anchor_world, secondary_anchor_world)

        if not self._dual_scale_active or self._dual_scale_object_id != object_state.object_id:
            self._dual_scale_active = True
            self._dual_scale_object_id = object_state.object_id
            self._dual_scale_baseline_distance_xy = max(distance_xy, 1e-4)
            self._dual_scale_baseline_scale = object_state.world_scale
            self._last_interacted_object_id = object_state.object_id

        baseline_distance = max(self._dual_scale_baseline_distance_xy or 1e-4, 1e-4)
        baseline_scale = self._dual_scale_baseline_scale or object_state.world_scale
        ratio_raw = distance_xy / baseline_distance
        ratio = max(ratio_raw, DUAL_SCALE_RATIO_EPSILON) ** DUAL_SCALE_RATIO_EXPONENT
        object_state.world_scale = (
            baseline_scale[0] * ratio,
            baseline_scale[1] * ratio,
            baseline_scale[2] * ratio,
        )
        object_state.absolute_scale_ratio = self._absolute_scale_ratio(object_state)
        self._dual_scale_ratio = object_state.absolute_scale_ratio
        self._grabbed_object_id = object_state.object_id
        self._hovered_object_id = object_state.object_id
        self._rotation_object_id = None
        object_state.grab_offset_world = None
        object_state.rotation_reference_hpr = None
        object_state.rotation_reference_input = None

        commands: list[SceneCommand] = []
        commands.extend(self._set_object_interaction_state(packet, object_state, BRIDGE_STATE_GRABBING))
        commands.append(
            self._make_dual_scale_pose(
                packet,
                object_state,
                ratio=self._dual_scale_ratio,
                distance_xy=distance_xy,
            )
        )
        return commands

    @staticmethod
    def _absolute_scale_ratio(object_state: ObjectInteractionState) -> float:
        initial_scale_x = max(float(object_state.initial_world_scale[0]), 1e-6)
        return float(object_state.world_scale[0]) / initial_scale_x

    @staticmethod
    def _rotation_mode_active(packet: GesturePacket) -> bool:
        rotation = getattr(packet, "rotation", None)
        if not isinstance(rotation, dict):
            debug_payload = getattr(packet, "debug", None)
            if not isinstance(debug_payload, dict):
                return False
            rotation = debug_payload.get("rotation")
        if not isinstance(rotation, dict):
            return False

        return bool(rotation.get("mode_active", False))

    def _handle_rotation_mode(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        is_hovering: bool,
        hand_anchor_world: Vec3,
    ) -> list[SceneCommand]:
        if packet.pinch_state != "pinched":
            if self._grabbed_object_id == object_state.object_id:
                self._grabbed_object_id = None
            self._rotation_object_id = None
            next_state = BRIDGE_STATE_PENDING_GRAB if is_hovering else BRIDGE_STATE_IDLE
            object_state.grab_offset_world = None
            object_state.rotation_reference_hpr = None
            object_state.rotation_reference_input = None
            return self._set_object_interaction_state(packet, object_state, next_state)

        commands: list[SceneCommand] = []
        self._grabbed_object_id = object_state.object_id
        self._rotation_object_id = object_state.object_id
        self._hovered_object_id = object_state.object_id
        object_state.grab_offset_world = None
        commands.extend(self._set_object_interaction_state(packet, object_state, BRIDGE_STATE_ROTATING))
        commands.append(self._make_object_pose(packet, object_state, hand_anchor_world))
        return commands

    @staticmethod
    def _rotation_input_hpr(packet: GesturePacket) -> tuple[float, float, float] | None:
        rotation = getattr(packet, "rotation", None)
        if not isinstance(rotation, dict):
            debug_payload = getattr(packet, "debug", None)
            if not isinstance(debug_payload, dict):
                return None
            rotation = debug_payload.get("rotation")
        if not isinstance(rotation, dict):
            return None

        if not bool(rotation.get("mode_active", False)):
            return None

        return (
            float(rotation.get("deg_x", 0.0)),
            float(rotation.get("deg_y", 0.0)),
            float(rotation.get("deg_z", 0.0)),
        )

    def _rotation_hpr_payload(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
    ) -> dict[str, float] | None:
        rotation_input = self._rotation_input_hpr(packet)
        if rotation_input is None:
            object_state.rotation_reference_hpr = None
            object_state.rotation_reference_input = None
            return None

        if object_state.rotation_reference_hpr is None or object_state.rotation_reference_input is None:
            object_state.rotation_reference_hpr = object_state.world_hpr
            object_state.rotation_reference_input = rotation_input

        base_h, base_p, base_r = object_state.rotation_reference_hpr
        ref_h, ref_p, ref_r = object_state.rotation_reference_input
        cur_h, cur_p, cur_r = rotation_input
        heading_delta = cur_h - ref_h
        if self._input_mirrored:
            heading_delta *= -1.0
        next_hpr = (
            base_h + (heading_delta * self._rotation_sensitivity),
            base_p + ((cur_p - ref_p) * self._rotation_sensitivity),
            base_r + ((cur_r - ref_r) * self._rotation_sensitivity),
        )
        object_state.world_hpr = next_hpr
        return {
            "h": next_hpr[0],
            "p": next_hpr[1],
            "r": next_hpr[2],
        }

    def _interaction_anchor(self, packet: GesturePacket) -> Vec3:
        return Vec3(
            x=(packet.index_tip.x + packet.thumb_tip.x) * 0.5,
            y=(packet.index_tip.y + packet.thumb_tip.y) * 0.5,
            z=(packet.index_tip.z + packet.thumb_tip.z) * 0.5,
        )

    def _drag_world_position(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        hand_anchor_world: Vec3 | None = None,
    ) -> tuple[Vec3, bool]:
        if hand_anchor_world is None:
            hand_anchor_world = self._camera_to_world_control_position(self._interaction_anchor(packet))
        if object_state.grab_offset_world is None:
            return object_state.world_position, False
        unconstrained_position = self._add_vec3(hand_anchor_world, object_state.grab_offset_world)
        return self._constrain_object_to_table(unconstrained_position, object_state)

    def _constrain_object_to_table(self, world_position: Vec3, object_state: ObjectInteractionState) -> tuple[Vec3, bool]:
        minimum_center_y = TABLE_SURFACE_Y + object_state.half_height
        if world_position.y >= minimum_center_y:
            return world_position, False
        return Vec3(world_position.x, minimum_center_y, world_position.z), True

    def _camera_to_world_position(self, position: Optional[Vec3]) -> Vec3:
        '''
        Complete camera_norm → world_norm coordinate transformation with full fault tolerance.
        
        camera_norm definition (gesture input space relative to camera frame):
        - +x: right (camera horizontal)
        - +y: up (camera vertical)
        - +z: toward the user/camera (camera depth)
        
        world_norm definition (renderer-facing scene space after bridge mapping):
        - +x: user-right in the scene
        - +y: up
        - +z: out of the screen toward the user
        
        :param position: Original coordinates in camera_norm (Vec3), None is allowed
        :return: Transformed coordinates in world_norm (Vec3)
        '''
        # 1. Null/illegal input fault tolerance
        if position is None:
            self._record_error(
                error_entry(
                    "bridge.coordinate.position.missing",
                    "Coordinate transformation failed because input position is missing",
                    recoverable=True,
                    hint="Ensure wrist is present before emitting pose updates.",
                    details={"position": None},
                )
            )
            coordinate_logger.error("Coordinate transformation failed: input position is None")
            return Vec3(0.0, 0.0, 0.0)
        
        # 2. Invalid value (NaN/Inf) validation
        def is_valid_num(v: float) -> bool:
            return not (math.isnan(v) or math.isinf(v))
        
        x = position.x if is_valid_num(position.x) else 0.0
        y = position.y if is_valid_num(position.y) else 0.0
        z = position.z if is_valid_num(position.z) else 0.0
        
        # 3. Convert from camera-centric coordinates into user-centric scene motion.
        # The scene camera views the origin from +Y, so world +x appears on the
        # left side of the screen. Mirrored input therefore needs an x inversion
        # to preserve user-perceived left/right motion, while unmirrored input
        # should keep x as-is. Positive camera z means "toward camera", which the
        # scene should interpret as moving farther into the screen.
        world_x = -x if self._input_mirrored else x
        world_y = y
        world_z = -z
        return Vec3(world_x, world_y, world_z)

    def _camera_to_world_control_position(self, position: Optional[Vec3]) -> Vec3:
        base_position = self._camera_to_world_position(position)
        return Vec3(
            x=base_position.x * self._position_sensitivity,
            y=base_position.y * self._position_sensitivity,
            z=base_position.z * self._position_sensitivity,
        )


    def _make_object_state(self, packet: GesturePacket, object_id: str, interaction_state: str) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-state", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_object_state",
            object_id=object_id,
            payload={"interaction_state": interaction_state},
        )

    def _make_heartbeat(self, packet: GesturePacket) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("heartbeat", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="heartbeat",
            object_id=BRIDGE_SCENE_OBJECT_ID,
            payload={"interaction_state": self._interaction_state},
        )

    def _make_reset_interaction(self, packet: GesturePacket, *, reason: str) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("reset", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="reset_interaction",
            object_id=BRIDGE_SCENE_OBJECT_ID,
            payload={"reason": reason},
        )

    def _ensure_object_state(
        self,
        object_id: str,
        *,
        world_position: Vec3,
        interaction_radius: float,
        half_height: float,
        world_scale: tuple[float, float, float],
        interaction_state: str,
        initialized: bool,
    ) -> ObjectInteractionState:
        object_state = self._object_states.get(object_id)
        if object_state is None:
            object_state = ObjectInteractionState(
                object_id=object_id,
                world_position=world_position,
                interaction_radius=interaction_radius,
                half_height=half_height,
                world_scale=world_scale,
                initial_world_scale=world_scale,
                absolute_scale_ratio=1.0,
                interaction_state=interaction_state,
                initialized=initialized,
            )
            self._object_states[object_id] = object_state
            return object_state

        object_state.world_position = world_position
        object_state.interaction_radius = interaction_radius
        object_state.half_height = half_height
        object_state.world_scale = world_scale
        if not object_state.initialized and initialized:
            object_state.initial_world_scale = world_scale
            object_state.absolute_scale_ratio = 1.0
        else:
            object_state.absolute_scale_ratio = self._absolute_scale_ratio(object_state)
        object_state.interaction_state = interaction_state
        object_state.initialized = initialized
        return object_state

    def _object_state(self, object_id: str) -> ObjectInteractionState:
        object_state = self._object_states.get(object_id)
        if object_state is not None:
            return object_state
        raise KeyError(f"Unknown scene object: {object_id}")

    def _rotation_target_object(self, hovered_object_id: str | None) -> ObjectInteractionState | None:
        if self._rotation_object_id is not None:
            return self._object_state(self._rotation_object_id)

        if self._grabbed_object_id is not None:
            self._rotation_object_id = self._grabbed_object_id
            return self._object_state(self._grabbed_object_id)

        if self._hovered_object_id is not None:
            self._rotation_object_id = self._hovered_object_id
            return self._object_state(self._hovered_object_id)

        if hovered_object_id is not None:
            self._hovered_object_id = hovered_object_id
            self._rotation_object_id = hovered_object_id
            return self._object_state(hovered_object_id)

        return None

    def _sync_hover_state(self, packet: GesturePacket, hovered_object_id: str | None) -> list[SceneCommand]:
        if hovered_object_id == self._hovered_object_id:
            return []

        commands: list[SceneCommand] = []
        previous_hover_id = self._hovered_object_id
        self._hovered_object_id = hovered_object_id

        if previous_hover_id is not None and previous_hover_id != self._grabbed_object_id:
            commands.extend(
                self._set_object_interaction_state(
                    packet,
                    self._object_state(previous_hover_id),
                    BRIDGE_STATE_IDLE,
                )
            )

        if hovered_object_id is not None and hovered_object_id != self._grabbed_object_id:
            commands.extend(
                self._set_object_interaction_state(
                    packet,
                    self._object_state(hovered_object_id),
                    BRIDGE_STATE_PENDING_GRAB,
                )
            )

        return commands

    def _set_object_interaction_state(
        self,
        packet: GesturePacket,
        object_state: ObjectInteractionState,
        bridge_state: str,
    ) -> list[SceneCommand]:
        if bridge_state == object_state.interaction_state:
            return []

        object_state.interaction_state = bridge_state
        if bridge_state in {BRIDGE_STATE_GRABBING, BRIDGE_STATE_ROTATING}:
            self._last_interacted_object_id = object_state.object_id
        if bridge_state != BRIDGE_STATE_GRABBING:
            object_state.grab_offset_world = None
        if bridge_state != BRIDGE_STATE_ROTATING:
            object_state.rotation_reference_hpr = None
            object_state.rotation_reference_input = None
        self._interaction_state = bridge_state
        return [self._make_object_state(packet, object_state.object_id, self._render_state(bridge_state))]

    @staticmethod
    def _render_state(bridge_state: str) -> str:
        if bridge_state == BRIDGE_STATE_PENDING_GRAB:
            return INTERACTION_PENDING_GRAB
        if bridge_state == BRIDGE_STATE_GRABBING:
            return INTERACTION_GRABBED
        if bridge_state == BRIDGE_STATE_ROTATING:
            return INTERACTION_ROTATING
        return INTERACTION_IDLE

    @staticmethod
    def _distance(a: Vec3, b: Vec3) -> float:
        return math.sqrt(((a.x - b.x) ** 2) + ((a.y - b.y) ** 2) + ((a.z - b.z) ** 2))

    def _is_hovering_object(self, hand_anchor_world: Vec3, object_state: ObjectInteractionState) -> bool:
        return self._distance(hand_anchor_world, object_state.world_position) <= object_state.interaction_radius

    def _select_hovered_object(self, hand_anchor_world: Vec3) -> ObjectInteractionState | None:
        best_match: ObjectInteractionState | None = None
        best_distance: float | None = None
        for object_state in self._object_states.values():
            distance = self._distance(hand_anchor_world, object_state.world_position)
            if distance > object_state.interaction_radius:
                continue
            if best_distance is None or distance < best_distance:
                best_match = object_state
                best_distance = distance
        return best_match

    @staticmethod
    def _add_vec3(a: Vec3, b: Vec3) -> Vec3:
        return Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

    @staticmethod
    def _subtract_vec3(a: Vec3, b: Vec3) -> Vec3:
        return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)

    def _translate_world_position(self, position: Vec3, offset: Vec3) -> Vec3:
        return Vec3(
            x=position.x + offset.x,
            y=position.y + offset.y,
            z=position.z + offset.z,
        )

    @staticmethod
    def _scale_vec3(value: Vec3, factor: float) -> Vec3:
        return Vec3(value.x * factor, value.y * factor, value.z * factor)

    @staticmethod
    def _normalized_vec3(value: Vec3) -> Vec3:
        length = math.sqrt((value.x ** 2) + (value.y ** 2) + (value.z ** 2))
        if length <= 1e-6:
            return Vec3(0.0, 0.0, 0.0)
        inv_length = 1.0 / length
        return Vec3(value.x * inv_length, value.y * inv_length, value.z * inv_length)

    def _record_error(self, error: dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]
