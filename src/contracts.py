from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TrackingState = Literal["tracked", "temporarily_lost", "not_detected"]
PinchState = Literal["open", "pinch_candidate", "pinched", "release_candidate"]
CoordinateSpace = Literal["camera_norm", "world_norm"]
SceneCommandType = Literal[
    "init_scene",
    "set_object_pose",
    "set_object_state",
    "heartbeat",
    "reset_interaction",
]


@dataclass(slots=True)
class Vec3:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class GesturePacket:
    contract_version: str
    frame_id: int
    timestamp_ms: int
    hand_id: str
    tracking_state: TrackingState
    confidence: float
    pinch_state: PinchState
    index_tip: Vec3
    thumb_tip: Vec3
    palm_center: Vec3
    coordinate_space: CoordinateSpace
    pinch_distance: float | None = None
    velocity: Vec3 | None = None
    smoothing_hint: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


@dataclass(slots=True)
class SceneCommand:
    contract_version: str
    command_id: str
    frame_id: int
    timestamp_ms: int
    command_type: SceneCommandType
    object_id: str
    payload: dict[str, Any]


__all__ = [
    "CoordinateSpace",
    "GesturePacket",
    "PinchState",
    "SceneCommand",
    "SceneCommandType",
    "TrackingState",
    "Vec3",
]
