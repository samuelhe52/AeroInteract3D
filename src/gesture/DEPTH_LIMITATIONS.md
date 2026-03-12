# MediaPipe Hand Depth Limitations

This note records an important limitation in the current gesture pipeline.

## Summary

MediaPipe Hand Landmarker does **not** provide a native camera-relative depth measurement for a hand landmark.

For forward/back interaction, the current normalized landmark `z` value is **not** sufficient as a stable distance-to-camera metric.

## What MediaPipe Actually Provides

### 1. Normalized landmarks

MediaPipe exposes `x`, `y`, and `z` for each of the 21 hand landmarks.

- `x` and `y` are normalized image coordinates.
- `z` is a model-estimated depth-like value.
- `z` is defined relative to the wrist landmark.
- The magnitude of `z` is only roughly on the same scale as `x`.

Implication:

- This is useful for hand pose geometry.
- This is useful for comparing which landmarks are farther forward or backward relative to the hand.
- This is **not** a native hand-to-camera distance signal.

### 2. World landmarks

MediaPipe also exposes hand world landmarks.

- These are expressed in meters.
- Their origin is the hand's approximate geometric center.
- They represent hand-local 3D structure rather than camera-space translation.

Implication:

- These landmarks are better for reasoning about 3D hand pose.
- They are not a direct measurement of how far the hand is from the camera.

## Limitation for This Project

If the goal is to move an object forward/backward based on how far the user's hand is from the camera, MediaPipe does not provide a native API that directly answers that question.

So:

- normalized landmark `z` is a weak proxy only
- world landmarks are not camera-relative depth
- monocular webcam input alone does not give true depth without additional modeling or hardware

## Practical Options

The following options are worth trying if better forward/back interaction is needed.

### 1. Heuristic monocular distance from hand scale

Estimate hand distance from apparent image size.

Examples:

- wrist to middle-finger MCP span in pixels
- index MCP to pinky MCP span in pixels
- bounding box area of the hand landmarks

Tradeoffs:

- simple to implement
- often more stable than normalized `z` for whole-hand distance
- still approximate and sensitive to hand orientation

### 2. Calibrated pose estimation with assumed hand size

Treat the hand as a roughly known-scale 3D structure and estimate pose relative to the camera.

Possible approach:

- calibrate the camera intrinsics
- choose a small set of stable landmarks
- assume approximate real-world hand dimensions
- solve for camera-relative pose with a PnP-style method

Tradeoffs:

- potentially better than pure heuristics
- still approximate because real hand sizes vary
- more complex and calibration-dependent

### 3. Real depth hardware

Use a depth-producing sensor instead of relying on monocular inference.

Examples:

- stereo camera
- depth camera
- LiDAR-backed device

Tradeoffs:

- best option for actual forward/back interaction
- requires different hardware and integration work

## Recommendation

For this repo, the practical experiment order is:

1. try heuristic monocular distance from hand scale
2. try calibrated pose estimation with assumed hand size if heuristics are not stable enough
3. use real depth hardware if accurate depth interaction becomes a hard requirement

## Current Contract Guidance

The gesture contract now uses `wrist` directly because MediaPipe exposes `wrist` as a real landmark, while it does not expose a dedicated palm-center landmark.

That resolves the naming mismatch, but it does **not** solve the depth limitation described above.
