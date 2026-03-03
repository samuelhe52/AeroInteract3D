# Rendering Module Specification

Purpose: consume bridge `SceneCommand` stream and render deterministic scene and object interaction feedback.

Rendering stack is fixed for current milestones: **Python + Panda3D**.

Contract behavior remains fixed regardless of implementation details.

## Technical Baseline (Python + Panda3D)

- MUST implement renderer logic in Python.
- MUST use Panda3D as the rendering engine/runtime.
- SHOULD rely on Panda3D scene graph, camera, and task loop primitives rather than raw graphics API calls.
- SHOULD keep windowing/context management behind an adapter boundary so the command consumer remains testable.
- SHOULD keep rendering loop timing decoupled from command ingestion (consume latest valid command state each frame).
- SHOULD treat GPU/resource initialization failures as structured startup errors.

## Deliverable Contract

Rendering module MUST consume `SceneCommand` exactly as defined in `src/contract_stub.md`.
Rendering module MUST import `SceneCommand` from `src/contracts.py`.
Rendering module MUST NOT define a local `SceneCommand` dataclass copy.

## Implementation Ownership

- Rendering maintainers MUST implement a concrete service class inheriting `RenderOutputPort` from `src/ports.py`.
- Use `RenderOutputServiceStub` in `src/rendering/service_stub.py` as the initial scaffold and replace no-op behavior incrementally.
- Application wiring in `main.py` imports this stub today so integration can proceed before full implementation.

## Functional Requirements

- MUST initialize scene via `init_scene` command before object updates.
- MUST apply `set_object_pose` updates to the target object identified by `object_id`.
- MUST apply `set_object_state` updates to visual interaction cues (`idle`, `hover`, `grabbed`).
- MUST accept and safely ignore duplicate `command_id` commands.
- MUST ignore or safely handle stale out-of-order command frames.
- MUST handle `reset_interaction` by restoring object interaction state to safe defaults.

## Contract and Data Handling

- MUST treat `world_norm` as canonical pose input space from bridge.
- MUST not reinterpret gesture-side semantics (pinch logic belongs to bridge).
- SHOULD tolerate unknown optional payload fields for forward compatibility.
- SHOULD produce lightweight runtime warnings for malformed commands without crashing.

## Error and Lifecycle

- MUST expose lifecycle states: `INITIALIZING`, `RUNNING`, `DEGRADED`, `STOPPED`.
- MUST continue rendering last valid scene state when transient command gaps occur.
- MUST fail explicitly with structured errors if scene cannot initialize.

## Integration Constraints

- MUST not import gesture internals.
- MUST not depend on bridge private implementation details.
- MUST integrate through a command-consumer boundary (queue, callback, or equivalent interface).

## Out of Scope (Rendering)

- Gesture recognition logic.
- Contract translation logic.
- Multi-object physics simulation beyond MVP interaction needs.
