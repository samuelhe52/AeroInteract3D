# Gesture Module Specification

Purpose: detect hand landmarks and emit an ordered stream of contract-compliant `GesturePacket` messages for the bridge.

Implementation workflow notes live in [workflow.md](workflow.md). The dedicated live preview debug entrypoint lives in src/gesture/debug/live_preview.py.

## Deliverable Contract

Gesture module MUST produce `GesturePacket` exactly as defined in `src/contract_stub.md`.
Gesture module MUST import `GesturePacket` from `src/contracts.py`.
Gesture module MUST NOT define a local `GesturePacket` dataclass copy.

## Implementation Ownership

- Gesture maintainers MUST implement a concrete service class inheriting `GestureInputPort` from `src/ports.py`.
- Use `GestureInputServiceImpl` in `src/gesture/service_impl.py` as the concrete gesture implementation.

## Functional Requirements

- MUST support single-hand tracking for MVP.
- MUST detect pinch intent and expose `pinch_state` values: `open`, `pinch_candidate`, `pinched`, `release_candidate`.
- MUST emit `tracking_state` values: `tracked`, `temporarily_lost`, `not_detected`.
- MUST emit `frame_id` and `timestamp_ms` monotonically.
- MUST emit coordinates in `camera_norm` space.

## Stability and Signal Quality

- MUST use hysteresis or equivalent logic to reduce pinch flicker.
- SHOULD smooth noisy coordinate updates while preserving responsiveness.
- SHOULD expose `confidence` with a meaningful 0.0-1.0 scale.
- SHOULD keep `hand_id` stable while a hand remains continuously tracked.

## Error and Lifecycle

- MUST expose lifecycle states: `INITIALIZING`, `RUNNING`, `DEGRADED`, `STOPPED`.
- MUST produce explicit degraded behavior when camera input is unavailable.
- MUST not crash on temporary landmark detection failures.

## Integration Constraints

- MUST not import bridge internals.
- MUST not import rendering internals.
- MUST keep output schema backward-compatible unless `contract_version` is upgraded.

## Out of Scope (Gesture)

- Multi-hand arbitration.
- Gesture vocabulary beyond pinch needed for one-cube grab/drag.
