from __future__ import annotations

import time
from typing import Any

import cv2

from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import (
    DEFAULT_PINCH_ENTER_THRESHOLD,
    DEFAULT_PINCH_HOLD_THRESHOLD,
    DEFAULT_PINCH_RELEASE_THRESHOLD,
    GestureRuntimeConfig,
    create_hand_detector,
    distance,
    open_capture,
)
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
        self._frame_id = 0
        self._hand_id = "hand-0"
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._last_timestamp_ms = 0
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)

    def start(self) -> None:
        self.lifecycle_state = LIFECYCLE_RUNNING

    def stop(self) -> None:
        self.lifecycle_state = LIFECYCLE_STOPPED

    def process_landmarks(
        self,
        landmarks: dict[str, Vec3] | None,
        timestamp_ms: int,
        raw_confidence: float,
    ) -> GesturePacket:
        timestamp_ms = self._normalize_timestamp(timestamp_ms)

        if landmarks is None:
            self._tracking_loss_streak += 1
            tracking_state = self._compute_tracking_state(None)
            pinch_state = self._compute_pinch_state(None, None)
            confidence = self._compute_confidence(None, tracking_state, pinch_state, raw_confidence)
            velocity = Vec3(0.0, 0.0, 0.0)
            packet = self._build_packet(
                timestamp_ms=timestamp_ms,
                tracking_state=tracking_state,
                pinch_state=pinch_state,
                confidence=confidence,
                index_tip=self._last_index_tip,
                thumb_tip=self._last_thumb_tip,
                palm_center=self._last_palm_center,
                velocity=velocity,
                debug={"reason": "no_hand_detected", "tracking_loss_streak": self._tracking_loss_streak},
            )
            self._tracking_state = tracking_state
            self._pinch_state = pinch_state
            return packet

        self._tracking_loss_streak = 0

        index_tip = landmarks["index_finger_tip"]
        thumb_tip = landmarks["thumb_tip"]
        palm_center = landmarks["wrist"]

        tracking_state = self._compute_tracking_state(landmarks)
        pinch_state = self._compute_pinch_state(index_tip, thumb_tip)
        confidence = self._compute_confidence(landmarks, tracking_state, pinch_state, raw_confidence)
        velocity = self._compute_velocity(palm_center, self._last_palm_center)

        self._last_index_tip = index_tip
        self._last_thumb_tip = thumb_tip
        self._last_palm_center = palm_center
        self._tracking_state = tracking_state
        self._pinch_state = pinch_state

        return self._build_packet(
            timestamp_ms=timestamp_ms,
            tracking_state=tracking_state,
            pinch_state=pinch_state,
            confidence=confidence,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            velocity=velocity,
            debug={"source": "mediapipe", "tracking_loss_streak": self._tracking_loss_streak},
        )

    def _normalize_timestamp(self, timestamp_ms: int) -> int:
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _compute_tracking_state(self, landmarks: dict[str, Vec3] | None) -> str:
        if landmarks is None:
            if self._tracking_loss_streak <= 2:
                return "temporarily_lost"
            return "not_detected"
        return "tracked"

    def _compute_pinch_state(self, index_tip: Vec3 | None, thumb_tip: Vec3 | None) -> str:
        if index_tip is None or thumb_tip is None:
            if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"}:
                self._release_candidate_streak += 1
                if self._release_candidate_streak <= 2:
                    return "release_candidate"
            self._pinch_candidate_streak = 0
            self._release_candidate_streak = 0
            return "open"

        pinch_distance = distance(index_tip, thumb_tip)
        if pinch_distance <= self.config.pinch_hold_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_candidate_streak >= 2:
                return "pinched"
            return "pinch_candidate"

        if pinch_distance <= self.config.pinch_enter_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_state == "pinched":
                return "pinched"
            return "pinch_candidate"

        self._pinch_candidate_streak = 0
        if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"} and pinch_distance >= self.config.release_threshold:
            self._release_candidate_streak += 1
            if self._release_candidate_streak <= 2:
                return "release_candidate"
            self._release_candidate_streak = 0
            return "open"

        self._release_candidate_streak = 0
        return "open"

    def _compute_confidence(
        self,
        landmarks: dict[str, Vec3] | None,
        tracking_state: str,
        pinch_state: str,
        raw_confidence: float,
    ) -> float:
        if tracking_state != "tracked":
            if tracking_state == "temporarily_lost":
                return 0.25
            return 0.0

        confidence_bonus = 0.05 if pinch_state == "pinched" else 0.0
        return max(0.0, min(1.0, raw_confidence + confidence_bonus))

    def _compute_velocity(self, current: Vec3, previous: Vec3) -> Vec3:
        return Vec3(
            x=current.x - previous.x,
            y=current.y - previous.y,
            z=current.z - previous.z,
        )

    def _build_packet(
        self,
        timestamp_ms: int,
        tracking_state: str,
        pinch_state: str,
        confidence: float,
        index_tip: Vec3,
        thumb_tip: Vec3,
        palm_center: Vec3,
        velocity: Vec3,
        debug: dict[str, Any],
    ) -> GesturePacket:
        self._frame_id += 1
        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self._hand_id,
            tracking_state=tracking_state,
            confidence=confidence,
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            coordinate_space="camera_norm",
            pinch_distance=distance(index_tip, thumb_tip),
            velocity=velocity,
            smoothing_hint={"method": "none", "alpha": 1.0},
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
            while True:
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