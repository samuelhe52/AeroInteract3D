from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal
from uuid import uuid4


LifecycleState = Literal["INITIALIZING", "RUNNING", "DEGRADED", "STOPPED"]
FrameStatus = Literal["fresh", "duplicate", "stale"]

LIFECYCLE_INITIALIZING: LifecycleState = "INITIALIZING"
LIFECYCLE_RUNNING: LifecycleState = "RUNNING"
LIFECYCLE_DEGRADED: LifecycleState = "DEGRADED"
LIFECYCLE_STOPPED: LifecycleState = "STOPPED"


def error_entry(
    code: str,
    message: str,
    *,
    recoverable: bool,
    hint: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "message": message,
        "recoverable": recoverable,
        "hint": hint,
    }
    if details:
        payload["details"] = _normalize(details)
    return payload


def build_health(
    *,
    component: str,
    lifecycle_state: LifecycleState,
    errors: list[dict[str, Any]],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "component": component,
        "lifecycle_state": lifecycle_state,
        "status": "ok" if not errors else "degraded",
        "errors": [_normalize(error) for error in errors],
        "stats": _normalize(stats or {}),
    }


def classify_frame(last_frame_id: int | None, frame_id: int) -> FrameStatus:
    if last_frame_id is None:
        return "fresh"
    if frame_id == last_frame_id:
        return "duplicate"
    if frame_id < last_frame_id:
        return "stale"
    return "fresh"


def make_command_id(prefix: str, frame_id: int) -> str:
    return f"{prefix}-{frame_id}-{uuid4().hex[:8]}"


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value