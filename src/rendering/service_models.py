from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ObjectInitialState:
    pos: tuple[float, float, float]
    hpr: tuple[float, float, float]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    state: str = "idle"


@dataclass(slots=True)
class SceneObjectDescriptor:
    object_id: str
    init_pos: tuple[float, float, float]
    init_hpr: tuple[float, float, float]
    interaction_state: str
    shape: str
    scale: tuple[float, float, float]
    color: tuple[float, float, float, float]
    interactable: bool


@dataclass(slots=True)
class ObjectVisualProfile:
    base_color: tuple[float, float, float, float]
    use_builtin_materials: bool


@dataclass(slots=True)
class RenderingMetrics:
    commands_seen: int = 0
    commands_applied: int = 0
    duplicate_commands: int = 0
    stale_commands: int = 0
    rejected_commands: int = 0
    resets_processed: int = 0
    pose_updates: int = 0
    state_updates: int = 0
    hand_pose_updates: int = 0
    init_scene_commands: int = 0
    heartbeats_received: int = 0
    render_steps: int = 0