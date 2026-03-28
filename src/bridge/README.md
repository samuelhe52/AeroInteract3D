# Bridge Module Specification

Purpose: translate and synchronize upstream `GesturePacket` stream into downstream `SceneCommand` stream while preserving deterministic interaction behavior.

## Scope

- Bridge is the only translation layer between gesture semantics and rendering semantics.
- Bridge MUST not depend on rendering backend details.
- Bridge MUST not require gesture module internal model details.

## In-Process Interfaces (MVP)

- `GestureInputPort`
  - `start() -> None`
  - `poll() -> GesturePacket | None`
  - `health() -> dict`
  - `stop() -> None`
- `RenderOutputPort`
  - `start() -> None`
  - `push(command: SceneCommand) -> None`
  - `health() -> dict`
  - `stop() -> None`
- `BridgeService`
  - `start() -> None`
  - `process(packet: GesturePacket) -> list[SceneCommand]`
  - `health() -> dict`
  - `stop() -> None`

Bridge MUST integrate through these abstract ports (or equivalent interfaces), not by importing concrete teammate internals.

Bridge MUST import `GesturePacket` and `SceneCommand` from `src/contracts.py`.
Bridge MUST NOT define local dataclass copies of these contract types.

## Implementation Ownership

- Bridge maintainers MUST implement a concrete service class inheriting `BridgeService` from `src/ports.py`.
- The current concrete implementation is `BridgeServiceImpl` in `src/bridge/service.py`.
- Application wiring in `main.py` imports this service today so integration can proceed before full implementation.
- Current coordination note: the bridge remains the call site for camera-to-world conversion before emitting `world_norm` pose commands, but the rendering maintainer is expected to implement the concrete conversion logic used by that hook.

## Core Responsibilities

- Validate incoming `GesturePacket` against shared contract.
- Maintain interaction state machine.
- Map coordinates from `camera_norm` to `world_norm`.
- Emit ordered `SceneCommand` messages.
- Handle packet anomalies (duplicates, stale frames, tracking loss) safely.

## Interaction State Machine

Required states:

- `idle`
- `pending_grab`
- `grabbing`

Bridge MUST own object interaction decisions. Rendering is a pure consumer of emitted `SceneCommand` updates.

Bridge MAY use `pinch_state` from gesture as a stabilized upstream hint, but object selection, hover gating, grab begin, drag offset, and release decisions MUST happen inside Bridge using the current object interaction table and gesture coordinates.

Required behavior:

- `idle -> pending_grab` when the hand interaction anchor is within the grab-init radius of an object.
- `pending_grab -> idle` when the hand leaves the grab-init radius.
- `pending_grab -> grabbing` when the candidate object receives stable grab intent.
- `grabbing -> pending_grab|idle` on release, depending on whether the hand remains near the object.
- `grabbing -> idle` with `reset_interaction` on tracking loss or insufficient confidence.

Bridge SHOULD preserve a relative grab offset captured at grab begin so dragging updates move the object relative to the hand rather than snapping the object to the hand center.
Bridge SHOULD emit a world-space hand overlay command so rendering can display the hand in the same coordinate space as interactive objects.

## Command Emission Rules

- MUST emit `init_scene` at startup or reinitialize.
- MUST emit `set_object_state` on pending_grab/grab/release boundaries.
- MUST emit `set_object_pose` only when object movement update is valid.
- MUST emit `set_hand_pose` when hand visibility or world-space hand points change.
- SHOULD coalesce pose updates when upstream bursts exceed render consumer pace.
- MUST emit `reset_interaction` when tracking is lost during active grab.
- MUST NOT use `reset_interaction` for a normal release.

## Error and Lifecycle Requirements

- Bridge lifecycle states: `INITIALIZING`, `RUNNING`, `DEGRADED`, `STOPPED`.
- MUST return structured error entries with code, message, and recoverability hint.
- MUST continue best-effort operation in `DEGRADED` state when possible.
- MUST support idempotent start/stop.

## Out of Scope (Bridge)

- Gesture model tuning.
- Renderer backend optimization.
- Multi-hand arbitration.
