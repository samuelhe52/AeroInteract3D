from __future__ import annotations

import time
from typing import Any

import cv2

from src.constants import DEBUG_FPS_SAMPLE_WINDOW
from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import (
    GestureRuntimeConfig,
    create_hand_detector,
    open_capture,
)
from src.gesture.temporal import GestureFrameAnalysis, GestureTemporalReducer
from src.utils.contracts import EXPECTED_CONTRACT_VERSION


LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_DEGRADED = "DEGRADED"
LIFECYCLE_STOPPED = "STOPPED"

DebugVideoConfig = GestureRuntimeConfig


class GestureDebugAnalyzer:
    def __init__(self, config: DebugVideoConfig) -> None:
        self.config = config
        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._reducer = GestureTemporalReducer()
        self._frame_id = 0
        self._hand_id = "hand-0"

    def start(self) -> None:
        self.lifecycle_state = LIFECYCLE_RUNNING
        self._reducer.reset()

    def stop(self) -> None:
        self.lifecycle_state = LIFECYCLE_STOPPED

    def process_landmarks(
        self,
        landmarks: dict[str, Vec3] | None,
        timestamp_ms: int,
        raw_confidence: float,
    ) -> GesturePacket:
        analysis = self._reducer.process(
            timestamp_ms=timestamp_ms,
            index_tip=None if landmarks is None else landmarks["index_finger_tip"],
            thumb_tip=None if landmarks is None else landmarks["thumb_tip"],
            palm_center=None if landmarks is None else landmarks["wrist"],
            raw_confidence=raw_confidence,
        )

        debug = {"source": "mediapipe", "tracking_loss_streak": analysis.tracking_loss_streak}
        if landmarks is None:
            debug = {"reason": "no_hand_detected", "tracking_loss_streak": analysis.tracking_loss_streak}

        return self._build_packet(analysis=analysis, debug=debug)

    def _build_packet(
        self,
        analysis: GestureFrameAnalysis,
        debug: dict[str, Any],
    ) -> GesturePacket:
        self._frame_id += 1
        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=self._frame_id,
            timestamp_ms=analysis.timestamp_ms,
            hand_id=self._hand_id,
            tracking_state=analysis.tracking_state,
            confidence=analysis.confidence,
            pinch_state=analysis.pinch_state,
            index_tip=analysis.index_tip,
            thumb_tip=analysis.thumb_tip,
            palm_center=analysis.palm_center,
            coordinate_space="camera_norm",
            pinch_distance=analysis.pinch_distance,
            velocity=analysis.velocity,
            smoothing_hint=analysis.smoothing_hint,
            debug=debug,
        )


def overlay_packet(frame: Any, packet: GesturePacket) -> Any:
    pinch_color = (0, 220, 255) if packet.pinch_state in {"pinched", "pinch_candidate"} else (0, 255, 0)
    lines = [
        f"frame_id={packet.frame_id}",
        f"tracking={packet.tracking_state}",
        f"pinch={packet.pinch_state}",
        f"confidence={packet.confidence:.2f}",
        f"pinch_distance={packet.pinch_distance or 0.0:.3f}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pinch_color, 2, cv2.LINE_AA)
        y += 28
    return frame


def overlay_fps(frame: Any, fps: float) -> Any:
    label = f"fps={fps:.1f}"
    cv2.putText(frame, label, (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1, cv2.LINE_AA)
    return frame


def _camera_norm_to_pixel(vec: Vec3, frame_width: int, frame_height: int, mirror: bool) -> tuple[int, int]:
    display_x = -vec.x if mirror else vec.x
    px = int((display_x * 0.5 + 0.5) * frame_width)
    py = int((0.5 - vec.y * 0.5) * frame_height)
    return px, py


def overlay_anchor_points(frame: Any, packet: GesturePacket, mirror: bool, draw_coordinates: bool) -> Any:
    frame_height, frame_width = frame.shape[:2]
    anchor_specs = [
        ("index", packet.index_tip, (0, 255, 255)),
        ("thumb", packet.thumb_tip, (255, 200, 0)),
        ("palm", packet.palm_center, (0, 255, 0)),
    ]

    for label, vec, color in anchor_specs:
        px, py = _camera_norm_to_pixel(vec, frame_width, frame_height, mirror)
        cv2.circle(frame, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if draw_coordinates:
            text = f"({vec.x:.2f}, {vec.y:.2f}, {vec.z:.2f})"
            cv2.putText(frame, text, (px + 10, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


def run_live_preview(config: DebugVideoConfig) -> None:
    capture = open_capture(config)
    if not capture.isOpened():
        raise RuntimeError("Unable to open video source")

    analyzer = GestureDebugAnalyzer(config)
    analyzer.start()

    wait_ms = 1
    if config.target_fps and config.target_fps > 0:
        wait_ms = max(1, int(1000.0 / config.target_fps))

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)

    try:
        with create_hand_detector(config) as detector:
            frame_index = 0
            previous_frame_started_at = time.perf_counter()
            smoothed_fps = 0.0
            while True:
                frame_started_at = time.perf_counter()
                frame_elapsed = max(frame_started_at - previous_frame_started_at, 1e-6)
                instantaneous_fps = 1.0 / frame_elapsed
                if smoothed_fps == 0.0:
                    smoothed_fps = instantaneous_fps
                else:
                    smoothed_fps += (instantaneous_fps - smoothed_fps) / DEBUG_FPS_SAMPLE_WINDOW
                previous_frame_started_at = frame_started_at

                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if config.max_frames > 0 and frame_index > config.max_frames:
                    break

                timestamp_ms = int(time.time() * 1000)
                landmarks, raw_confidence, raw_landmarks = detector.detect(frame, timestamp_ms)

                display_frame = frame.copy()
                detector.draw(display_frame, raw_landmarks)
                packet = analyzer.process_landmarks(landmarks, timestamp_ms, raw_confidence)

                if config.mirror:
                    display_frame = cv2.flip(display_frame, 1)

                overlay_packet(display_frame, packet)
                overlay_fps(display_frame, smoothed_fps)
                overlay_anchor_points(display_frame, packet, config.mirror, config.draw_coordinates)

                cv2.imshow(config.window_name, display_frame)
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in {27, ord("q")}:
                    break
    finally:
        analyzer.stop()
        capture.release()
        cv2.destroyAllWindows()


__all__ = ["DebugVideoConfig", "run_live_preview"]