# Shared Contract Specification (Bridge-First)

This file is the single source of truth for module integration contracts.

All modules MUST implement this contract. Any incompatible change MUST increment `contract_version` and be reviewed by all module owners.

## 1) Contract Versioning

- `contract_version`: string, semantic version format (`MAJOR.MINOR.PATCH`).
- Current baseline: `0.1.0`.
- Compatibility rule:
  - `PATCH`: docs clarifications only, no schema change.
  - `MINOR`: backward-compatible field additions (new optional fields only).
  - `MAJOR`: backward-incompatible schema or semantic changes.

## 2) Coordinate System and Units

All coordinate-bearing payloads MUST include explicit frame metadata.

- World units: normalized floats in `[-1.0, 1.0]` for MVP.
- Axes convention (default):
  - `+x`: right
  - `+y`: up
  - `+z`: toward user/camera
- `coordinate_space` values:
  - `camera_norm`: camera-normalized coordinates.
  - `world_norm`: bridge-mapped normalized world coordinates.

## 3) `GesturePacket` (Gesture -> Bridge)

MUST be emitted as an ordered stream.

### Required fields (`GesturePacket`)

- `contract_version: str`
- `frame_id: int` (monotonic, starts from 0 or 1)
- `timestamp_ms: int` (monotonic source from producer clock)
- `hand_id: str` (stable while tracked)
- `tracking_state: str` in `{ "tracked", "temporarily_lost", "not_detected" }`
- `confidence: float` in `[0.0, 1.0]`
- `pinch_state: str` in `{ "open", "pinch_candidate", "pinched", "release_candidate" }`
- `index_tip: {"x": float, "y": float, "z": float}`
- `thumb_tip: {"x": float, "y": float, "z": float}`
- `palm_center: {"x": float, "y": float, "z": float}`
- `coordinate_space: str` (typically `camera_norm` from gesture module)

### Optional fields (recommended)

- `pinch_distance: float`
- `velocity: {"x": float, "y": float, "z": float}`
- `smoothing_hint: {"method": str, "window": int}`
- `debug: dict`

## 4) `SceneCommand` (Bridge -> Rendering)

MUST be consumed as an ordered stream and applied idempotently where possible.

### Required fields (`SceneCommand`)

- `contract_version: str`
- `command_id: str` (unique)
- `frame_id: int` (source-aligned when available)
- `timestamp_ms: int`
- `command_type: str` in
  - `{ "init_scene", "set_object_pose", "set_object_state", "heartbeat", "reset_interaction" }`
- `object_id: str` (for object-scoped commands)
- `payload: dict` (schema depends on `command_type`)

### Payload requirements by `command_type`

- `init_scene`
  - MUST include target object identifiers and default transforms.
- `set_object_pose`
  - MUST include `position: {x,y,z}`.
  - MAY include `rotation: {x,y,z,w}` and `scale: {x,y,z}`.
  - MUST include `coordinate_space` (`world_norm`).
- `set_object_state`
  - MUST include `interaction_state` in `{ "idle", "hover", "grabbed" }`.
- `heartbeat`
  - MUST include minimal liveness payload.
- `reset_interaction`
  - MUST include reason in `{ "tracking_lost", "manual", "reinitialize" }`.

## 5) Ordering and Delivery Semantics

- Producers MUST emit non-decreasing `frame_id` and monotonic `timestamp_ms`.
- Consumers MUST tolerate duplicates and ignore already-applied `command_id`.
- Consumers SHOULD ignore stale messages when `frame_id` goes backward.
- Bridge MUST be the only module that translates between gesture semantics and rendering semantics.

## 6) Lifecycle and Health Signaling

Each module SHOULD expose lifecycle state:

- `INITIALIZING`
- `RUNNING`
- `DEGRADED`
- `STOPPED`

Modules MUST fail explicitly with structured error messages instead of silent failure.

## 7) Non-Goals (MVP)

- No multi-hand interaction contract yet.
- No persistence/replay format standardization yet.
- No network transport requirements (in-process integration only for MVP).
