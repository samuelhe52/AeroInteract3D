from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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

import math
import logging

# Create a dedicated logger for bridge service
logger = logging.getLogger("bridge.service")

# Create a dedicated logger for coordinate transformation
coordinate_logger = logging.getLogger("bridge.coordinate_transformation")


BRIDGE_STATE_IDLE = "idle"
BRIDGE_STATE_PINCH_CANDIDATE = "pinch_candidate"
BRIDGE_STATE_GRABBING = "grabbing"
BRIDGE_STATE_RELEASE_CANDIDATE = "release_candidate"

OBJECT_ID = "primary_cube"
INTERACTION_IDLE = "idle"
INTERACTION_GRABBED = "grabbed"
PINCH_STABILITY_FRAMES = 2
RELEASE_STABILITY_FRAMES = 2
MIN_TRACKING_CONFIDENCE = 0.6


@dataclass(slots=True)
class BridgeMetrics:
    packets_seen: int = 0
    commands_emitted: int = 0
    duplicate_packets: int = 0
    stale_packets: int = 0
    rejected_packets: int = 0
    resets_emitted: int = 0
    pose_updates: int = 0


class BridgeServiceImpl(BridgeService):
    def __init__(self) -> None:
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._expected_contract_version = EXPECTED_CONTRACT_VERSION
        self._interaction_state = BRIDGE_STATE_IDLE
        self._last_frame_id: int | None = None
        self._last_timestamp_ms: int | None = None
        self._errors: list[dict[str, Any]] = []
        self._metrics = BridgeMetrics()
        self._pending_init = False
        self._pinch_streak = 0
        self._release_streak = 0

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
        self._pinch_streak = 0
        self._release_streak = 0
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

        commands.extend(self._step_state_machine(packet))
        if packet.frame_id % 30 == 0:
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
        self._pinch_streak = 0
        self._release_streak = 0
        self._interaction_state = BRIDGE_STATE_IDLE
        self.lifecycle_state = LIFECYCLE_STOPPED
        return None

    def _step_state_machine(self, packet: GesturePacket) -> list[SceneCommand]:
        commands: list[SceneCommand] = []

        if packet.tracking_state != "tracked" or packet.confidence < MIN_TRACKING_CONFIDENCE:
            if self._interaction_state == BRIDGE_STATE_GRABBING:
                return self._reset_interaction(packet, reason="tracking_lost")
            if self._interaction_state == BRIDGE_STATE_RELEASE_CANDIDATE:
                self._interaction_state = BRIDGE_STATE_IDLE
                self._release_streak = 0
            self._pinch_streak = 0
            return commands

        if self._interaction_state == BRIDGE_STATE_IDLE:
            if packet.pinch_state in {"pinch_candidate", "pinched"}:
                self._pinch_streak = 1
                self._interaction_state = BRIDGE_STATE_PINCH_CANDIDATE
            else:
                self._pinch_streak = 0
            return commands

        if self._interaction_state == BRIDGE_STATE_PINCH_CANDIDATE:
            if packet.pinch_state == "pinched":
                self._pinch_streak += 1
                if self._pinch_streak >= PINCH_STABILITY_FRAMES:
                    self._interaction_state = BRIDGE_STATE_GRABBING
                    self._release_streak = 0
                    commands.append(self._make_object_state(packet, INTERACTION_GRABBED))
                    commands.append(self._make_object_pose(packet))
                return commands

            if packet.pinch_state == "pinch_candidate":
                return commands

            self._interaction_state = BRIDGE_STATE_IDLE
            self._pinch_streak = 0
            return commands

        if self._interaction_state == BRIDGE_STATE_GRABBING:
            if packet.pinch_state in {"release_candidate", "open"}:
                self._interaction_state = BRIDGE_STATE_RELEASE_CANDIDATE
                self._release_streak = 1
                return commands

            commands.append(self._make_object_pose(packet))
            return commands

        if self._interaction_state == BRIDGE_STATE_RELEASE_CANDIDATE:
            if packet.pinch_state in {"release_candidate", "open"}:
                self._release_streak += 1
                if self._release_streak >= RELEASE_STABILITY_FRAMES:
                    self._interaction_state = BRIDGE_STATE_IDLE
                    self._pinch_streak = 0
                    self._release_streak = 0
                    commands.append(self._make_object_state(packet, INTERACTION_IDLE))
                return commands

            self._interaction_state = BRIDGE_STATE_GRABBING
            self._release_streak = 0
            commands.append(self._make_object_pose(packet))
            return commands

        return commands

    def _reset_interaction(self, packet: GesturePacket, *, reason: str) -> list[SceneCommand]:
        self._interaction_state = BRIDGE_STATE_IDLE
        self._pinch_streak = 0
        self._release_streak = 0
        self._metrics.resets_emitted += 1
        return [
            self._make_reset_interaction(packet, reason=reason),
            self._make_object_state(packet, INTERACTION_IDLE),
        ]

    def _make_init_scene(self, packet: GesturePacket) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("init-scene", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="init_scene",
            object_id=OBJECT_ID,
            payload={
                "objects": [
                    {
                        "object_id": OBJECT_ID,
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "coordinate_space": "world_norm",
                        "interaction_state": INTERACTION_IDLE,
                    }
                ]
            },
        )

    def _make_object_pose(self, packet: GesturePacket) -> SceneCommand:
        self._metrics.pose_updates += 1
        world_position = self._camera_to_world_position(packet.palm_center)
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-pose", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_object_pose",
            object_id=OBJECT_ID,
            payload={
                "position": vec3_payload(world_position),
                "coordinate_space": "world_norm",
            },
        )

    logger = logging.getLogger("bridge_service")
    def _camera_to_world_position(self, position: Optional[Vec3]) -> Vec3:
        '''
        Complete camera_norm → world_norm coordinate transformation with full fault tolerance.
        
        camera_norm definition (gesture input space relative to camera frame):
        - +x: right (camera horizontal)
        - +y: up (camera vertical)
        - +z: toward the user/camera (camera depth)
        
        world_norm definition (renderer-facing scene space after bridge mapping):
        - +x: right (scene horizontal)
        - +y: up (scene vertical)
        - +z: away from the camera (scene depth)
        
        :param position: Original coordinates in camera_norm (Vec3), None is allowed
        :return: Transformed coordinates in world_norm (Vec3), guaranteed to be within [-1.0, 1.0]
        '''
        # 1. Null/illegal input fault tolerance
        if position is None:
            self._log_error("Coordinate transformation failed: input position is None")
            return Vec3(0.0, 0.0, 0.5)  # Return default safe position
        
        # 2. Invalid value (NaN/Inf) validation
        def is_valid_num(v: float) -> bool:
            return not (math.isnan(v) or math.isinf(v))
        
        x = position.x if is_valid_num(position.x) else 0.0
        y = position.y if is_valid_num(position.y) else 0.0
        z = position.z if is_valid_num(position.z) else 0.0
        
        # 3. Core transformation logic (camera_norm → world_norm)
        # - Scale by 0.8 to reserve 20% margin for world_norm range [-1.0, 1.0]
        # - Invert z-axis: camera +z (toward user) → world +z (away from camera)
        scaled_x = x * 0.8
        scaled_y = y * 0.8
        scaled_z = -z * 0.8  # Invert z-axis for world space alignment
        
        # 4. Final range clipping (guarantee no out-of-bounds in world_norm)
        def clip(v: float) -> float:
            return max(-1.0, min(1.0, v))
        
        final_x = clip(scaled_x)
        final_y = clip(scaled_y)
        final_z = clip(scaled_z + 0.5)  # Add offset to position interaction area in front of camera
        
        # 5. Warning log for clipped coordinates (aids debugging)
        if abs(scaled_x) > 1.0 or abs(scaled_y) > 1.0 or abs(scaled_z + 0.5) > 1.0:
            self._log_warning(
                f"Coordinate clipped: original({x:.2f},{y:.2f},{z:.2f}) → "
                f"scaled({scaled_x:.2f},{scaled_y:.2f},{scaled_z+0.5:.2f}) → "
                f"final({final_x:.2f},{final_y:.2f},{final_z:.2f})"
            )
        
        return Vec3(final_x, final_y, final_z)



    def _make_object_state(self, packet: GesturePacket, interaction_state: str) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("set-state", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="set_object_state",
            object_id=OBJECT_ID,
            payload={"interaction_state": interaction_state},
        )

    def _make_heartbeat(self, packet: GesturePacket) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("heartbeat", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="heartbeat",
            object_id=OBJECT_ID,
            payload={"interaction_state": self._interaction_state},
        )

    def _make_reset_interaction(self, packet: GesturePacket, *, reason: str) -> SceneCommand:
        return SceneCommand(
            contract_version=self._expected_contract_version,
            command_id=make_command_id("reset", packet.frame_id),
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            command_type="reset_interaction",
            object_id=OBJECT_ID,
            payload={"reason": reason},
        )

    def _record_error(self, error: dict[str, Any]) -> None:
        self._errors.append(error)
        self._errors = self._errors[-10:]