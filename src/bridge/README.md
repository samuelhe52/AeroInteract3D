# Bridge Module Specification

Purpose: translate and synchronize upstream `GesturePacket` stream into downstream `SceneCommand` stream while preserving deterministic interaction behavior.

## Scope

- Bridge is the only translation layer between gesture semantics and rendering semantics.
- Bridge MUST not depend on rendering backend details.
- Bridge MUST not require gesture module internal model details.

## In-Process Interfaces (MVP)

- `GestureInputPort`
  - `poll() -> GesturePacket | None`
  - `health() -> dict`
- `RenderOutputPort`
  - `push(command: SceneCommand) -> None`
  - `health() -> dict`

Bridge MUST integrate through these abstract ports (or equivalent interfaces), not by importing concrete teammate internals.

## Core Responsibilities

- Validate incoming `GesturePacket` against shared contract.
- Maintain interaction state machine.
- Map coordinates from `camera_norm` to `world_norm`.
- Emit ordered `SceneCommand` messages.
- Handle packet anomalies (duplicates, stale frames, tracking loss) safely.

## Interaction State Machine

Required states:

- `idle`
- `pinch_candidate`
- `grabbing`
- `release_candidate`

State transitions MUST be driven only by `pinch_state`, `tracking_state`, and confidence gating from `GesturePacket`.

Required behavior:

- `idle -> pinch_candidate` when pinch intent appears.
- `pinch_candidate -> grabbing` only after stability criteria is met.
- `grabbing -> release_candidate` when release intent appears.
- `release_candidate -> idle` when release is confirmed.
- Any state -> `idle` on prolonged `not_detected` or explicit reset signal.

## Command Emission Rules

- MUST emit `init_scene` at startup or reinitialize.
- MUST emit `set_object_state` on grab/release boundaries.
- MUST emit `set_object_pose` only when object movement update is valid.
- SHOULD coalesce pose updates when upstream bursts exceed render consumer pace.
- MUST emit `reset_interaction` when tracking is lost during active grab.

## Error and Lifecycle Requirements

- Bridge lifecycle states: `INITIALIZING`, `RUNNING`, `DEGRADED`, `STOPPED`.
- MUST return structured error entries with code, message, and recoverability hint.
- MUST continue best-effort operation in `DEGRADED` state when possible.
- MUST support idempotent start/stop.

## Out of Scope (Bridge)

- Gesture model tuning.
- Renderer backend optimization.
- Multi-hand arbitration.
