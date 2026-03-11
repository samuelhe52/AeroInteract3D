from __future__ import annotations

from typing import Any

from src.contracts import GesturePacket, SceneCommand, Vec3
from src.utils.runtime import error_entry


EXPECTED_CONTRACT_VERSION = "0.1.0"

SCENE_COMMAND_TYPES = {
    "init_scene",
    "set_object_pose",
    "set_object_state",
    "heartbeat",
    "reset_interaction",
}


def validate_contract_version(
    contract_version: str,
    *,
    expected_version: str = EXPECTED_CONTRACT_VERSION,
    field_name: str = "contract_version",
) -> list[dict[str, Any]]:
    if contract_version == expected_version:
        return []
    return [
        error_entry(
            "contract.version_mismatch",
            f"Unsupported {field_name}: {contract_version}",
            recoverable=True,
            hint=f"Send schema version {expected_version} until a coordinated version bump lands.",
            details={"expected": expected_version, "actual": contract_version},
        )
    ]


def validate_gesture_packet(
    packet: GesturePacket,
    *,
    expected_version: str = EXPECTED_CONTRACT_VERSION,
) -> list[dict[str, Any]]:
    errors = validate_contract_version(packet.contract_version, expected_version=expected_version)

    if packet.frame_id < 0:
        errors.append(
            error_entry(
                "gesture.frame_id.invalid",
                "frame_id must be non-negative",
                recoverable=True,
                hint="Emit monotonically increasing non-negative frame identifiers.",
                details={"frame_id": packet.frame_id},
            )
        )

    if packet.timestamp_ms < 0:
        errors.append(
            error_entry(
                "gesture.timestamp.invalid",
                "timestamp_ms must be non-negative",
                recoverable=True,
                hint="Emit a monotonic timestamp in milliseconds.",
                details={"timestamp_ms": packet.timestamp_ms},
            )
        )

    if not 0.0 <= packet.confidence <= 1.0:
        errors.append(
            error_entry(
                "gesture.confidence.invalid",
                "confidence must be within [0.0, 1.0]",
                recoverable=True,
                hint="Clamp or normalize confidence before emitting packets.",
                details={"confidence": packet.confidence},
            )
        )

    if not packet.hand_id:
        errors.append(
            error_entry(
                "gesture.hand_id.missing",
                "hand_id must be a non-empty string",
                recoverable=True,
                hint="Provide a stable hand identifier while a hand remains tracked.",
            )
        )

    if packet.coordinate_space != "camera_norm":
        errors.append(
            error_entry(
                "gesture.coordinate_space.invalid",
                "Gesture packets must currently be emitted in camera_norm space",
                recoverable=True,
                hint="Normalize upstream coordinates into camera_norm before bridge ingestion.",
                details={"coordinate_space": packet.coordinate_space},
            )
        )

    for name, vec in (
        ("index_tip", packet.index_tip),
        ("thumb_tip", packet.thumb_tip),
        ("palm_center", packet.palm_center),
    ):
        errors.extend(validate_vec3(name, vec))

    return errors


def validate_scene_command(
    command: SceneCommand,
    *,
    expected_version: str = EXPECTED_CONTRACT_VERSION,
) -> list[dict[str, Any]]:
    errors = validate_contract_version(command.contract_version, expected_version=expected_version)

    if not isinstance(command.command_id, str) or not command.command_id:
        errors.append(
            error_entry(
                "scene.command_id.invalid",
                "command_id must be a non-empty string",
                recoverable=True,
                hint="Emit a unique non-empty command identifier for every scene command.",
                details={"command_id": command.command_id},
            )
        )

    if not isinstance(command.frame_id, int) or command.frame_id < 0:
        errors.append(
            error_entry(
                "scene.frame_id.invalid",
                "frame_id must be a non-negative integer",
                recoverable=True,
                hint="Emit monotonically increasing non-negative frame identifiers.",
                details={"frame_id": command.frame_id},
            )
        )

    if not isinstance(command.timestamp_ms, int) or command.timestamp_ms < 0:
        errors.append(
            error_entry(
                "scene.timestamp.invalid",
                "timestamp_ms must be a non-negative integer",
                recoverable=True,
                hint="Emit a monotonic timestamp in milliseconds for every scene command.",
                details={"timestamp_ms": command.timestamp_ms},
            )
        )

    if not isinstance(command.command_type, str) or command.command_type not in SCENE_COMMAND_TYPES:
        errors.append(
            error_entry(
                "scene.command_type.invalid",
                "command_type is not supported",
                recoverable=True,
                hint="Use one of the defined SceneCommandType values.",
                details={"command_type": command.command_type},
            )
        )

    if not isinstance(command.object_id, str) or not command.object_id:
        errors.append(
            error_entry(
                "scene.object_id.invalid",
                "object_id must be a non-empty string",
                recoverable=True,
                hint="Provide a target object identifier for every scene command.",
                details={"object_id": command.object_id},
            )
        )

    if not isinstance(command.payload, dict):
        errors.append(
            error_entry(
                "scene.payload.invalid",
                "payload must be a dictionary",
                recoverable=True,
                hint="Encode scene command payloads as JSON-style dictionaries.",
                details={"payload_type": type(command.payload).__name__},
            )
        )

    return errors


def validate_vec3(name: str, value: Vec3) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for axis in ("x", "y", "z"):
        component = getattr(value, axis)
        if not isinstance(component, int | float):
            errors.append(
                error_entry(
                    f"gesture.{name}.{axis}.invalid",
                    f"{name}.{axis} must be numeric",
                    recoverable=True,
                    hint="Emit floating-point vector components.",
                    details={"value": component},
                )
            )
    return errors


def vec3_payload(value: Vec3) -> dict[str, float]:
    return {
        "x": float(value.x),
        "y": float(value.y),
        "z": float(value.z),
    }