from __future__ import annotations

from typing import Any


def clamp_display_scale(value: float | int | None) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    if scale <= 0.0:
        return 1.0
    return max(1.0, min(scale, 4.0))


def logical_size_from_physical(size: tuple[int, int], display_scale: float | int | None) -> tuple[int, int]:
    scale = clamp_display_scale(display_scale)
    return (
        max(int(round(int(size[0]) / scale)), 1),
        max(int(round(int(size[1]) / scale)), 1),
    )


def apply_root_display_scale(root: Any, display_scale: float | int | None) -> float:
    scale = clamp_display_scale(display_scale)
    set_scale = getattr(root, "setScale", None)
    if callable(set_scale):
        set_scale(scale, 1.0, scale)
    return scale
