from __future__ import annotations

from typing import Any

from src.contracts import GesturePacket, Vec3
from src.utils.runtime import error_entry


EXPECTED_CONTRACT_VERSION = "0.1.0"


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