# AeroInteract3D

## Project Intent

Build a real-time 3D mid-air interaction prototype where a user controls a virtual cube using hand gestures from a webcam.

The system should:

- detect hand landmarks in real time,
- recognize stable pinch events (thumb-index) with hysteresis,
- map hand coordinates from camera space into virtual 3D space,
- support grabbing and dragging one cube,
- render immediate visual feedback in a 3D scene.

## Baseline Scope (MVP)

Single-hand tracking, pinch-to-grab, and smooth cube translation in real time. Focus on robustness, coordinate consistency, and low-latency integration rather than training new ML models.

## Basic Tech Stack

- MediaPipe for gesture recognition.
- OpenGL with Python or Three.js for web-based rendering.

## Architecture Direction (Concise)

The architecture is intentionally defined at a high level with three modules:

1. Gesture + Coordinate Output
   - Responsible for gesture recognition and coordinate stream output.

2. Bridge Layer
   - Responsible for translating and synchronizing upstream coordinate streams with downstream renderer expectations.

3. Rendering + Scene + Interaction
   - Responsible for rendering engine integration, scene declaration, and interaction control.

Architecture principle:

- The upstream and downstream modules must align on a shared interface contract so the bridge can remain stable and communication-safe.
- This overview intentionally avoids implementation-level details at this stage.

## Bare-bone Module Skeleton

Current minimal module skeleton:

- `src/gesture/`
- `src/bridge/`
- `src/contract_stub.md`
- `src/rendering/`

Rendering backend choice remains open and will be decided later without changing this high-level split.

## Team Collaboration Context

This is a 3-member group project:

- Member A: responsible for overall design direction and module inspiration.
- Member B: responsible for practical implementation.
- Member C: responsible for practical implementation.

Implementation decisions should stay aligned with the design direction while prioritizing feasible, testable, and maintainable execution.

## Language Policy

- Group communication may be in either Chinese or English.
- Assistant responses should follow the user's chosen language.
- Developer-facing documentation (for example, this file and other development docs) should use English as the primary language.
- User-facing documentation (for example, README) should be bilingual (English + Chinese).
