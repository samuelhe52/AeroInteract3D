from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.contracts import GesturePacket, Vec3

try:
    from mediapipe.tasks.python.core.base_options import BaseOptions as MpBaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode as MpVisionTaskRunningMode,
    )
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker as MpHandLandmarker,
        HandLandmarkerOptions as MpHandLandmarkerOptions,
        HandLandmarksConnections as MpHandLandmarksConnections,
    )
except ImportError:
    MpBaseOptions = None
    MpVisionTaskRunningMode = None
    MpHandLandmarker = None
    MpHandLandmarkerOptions = None
    MpHandLandmarksConnections = None


LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_DEGRADED = "DEGRADED"
LIFECYCLE_STOPPED = "STOPPED"

DEFAULT_PINCH_ENTER_THRESHOLD = 0.10
DEFAULT_PINCH_HOLD_THRESHOLD = 0.06
DEFAULT_PINCH_RELEASE_THRESHOLD = 0.12

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


@dataclass(slots=True)
class DebugVideoConfig:
    input_video: str | None
    camera_index: int | None
    hand_model: str | None
    output_dir: Path | None
    target_fps: float | None
    max_frames: int
    min_detection_confidence: float
    min_tracking_confidence: float
    model_complexity: int
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True
    pinch_enter_threshold: float = DEFAULT_PINCH_ENTER_THRESHOLD
    pinch_hold_threshold: float = DEFAULT_PINCH_HOLD_THRESHOLD
    release_threshold: float = DEFAULT_PINCH_RELEASE_THRESHOLD


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live gesture preview with landmarks and GesturePacket overlay")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-video", help="path to an input video file")
    source_group.add_argument("--camera-index", type=int, help="camera index for live capture")
    parser.add_argument("--hand-model", help="path to a MediaPipe hand_landmarker.task model file")
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no frame limit")
    parser.add_argument("--window-name", default="Gesture Live Preview")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--hide-coordinates", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_config(args: argparse.Namespace) -> DebugVideoConfig:
    return DebugVideoConfig(
        input_video=args.input_video,
        camera_index=args.camera_index,
        hand_model=args.hand_model,
        output_dir=None,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        window_name=args.window_name,
        mirror=not args.no_mirror,
        draw_coordinates=not args.hide_coordinates,
    )


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
            contract_version="0.1.0",
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


class LegacySolutionsHandDetector:
    backend_name = "mediapipe_solutions"

    def __init__(self, config: DebugVideoConfig) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def __enter__(self) -> "LegacySolutionsHandDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._hands.close()

    def detect(self, frame: Any, timestamp_ms: int) -> tuple[dict[str, Vec3] | None, float, Any]:
        del timestamp_ms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, 0.0, None

        hand_landmarks = results.multi_hand_landmarks[0]
        raw_confidence = 0.0
        if results.multi_handedness:
            raw_confidence = float(results.multi_handedness[0].classification[0].score)
        return extract_landmarks(hand_landmarks.landmark), raw_confidence, hand_landmarks

    def draw(self, frame: Any, raw_landmarks: Any) -> None:
        if raw_landmarks is None:
            return
        self._drawing.draw_landmarks(frame, raw_landmarks, self._connections)


class TaskHandLandmarkerDetector:
    backend_name = "mediapipe_tasks"

    def __init__(self, config: DebugVideoConfig) -> None:
        if (
            MpBaseOptions is None
            or MpVisionTaskRunningMode is None
            or MpHandLandmarker is None
            or MpHandLandmarkerOptions is None
            or MpHandLandmarksConnections is None
        ):
            raise RuntimeError("MediaPipe tasks API is unavailable in this environment")
        if not config.hand_model:
            raise RuntimeError("A hand landmarker model is required for MediaPipe tasks backend")
        model_path = Path(config.hand_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")

        options = MpHandLandmarkerOptions(
            base_options=MpBaseOptions(model_asset_path=str(model_path)),
            running_mode=MpVisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = MpHandLandmarker.create_from_options(options)
        self._connections = list(MpHandLandmarksConnections.HAND_CONNECTIONS)

    def __enter__(self) -> "TaskHandLandmarkerDetector":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._landmarker.close()

    def detect(self, frame: Any, timestamp_ms: int) -> tuple[dict[str, Vec3] | None, float, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None, 0.0, None

        raw_landmarks = result.hand_landmarks[0]
        raw_confidence = 0.0
        if result.handedness:
            raw_confidence = float(result.handedness[0][0].score)
        return extract_landmarks(raw_landmarks), raw_confidence, raw_landmarks

    def draw(self, frame: Any, raw_landmarks: Any) -> None:
        if raw_landmarks is None:
            return

        frame_height, frame_width = frame.shape[:2]
        for connection in self._connections:
            start = raw_landmarks[connection.start]
            end = raw_landmarks[connection.end]
            start_point = (int(start.x * frame_width), int(start.y * frame_height))
            end_point = (int(end.x * frame_width), int(end.y * frame_height))
            cv2.line(frame, start_point, end_point, (255, 180, 0), 2, cv2.LINE_AA)

        for landmark in raw_landmarks:
            center = (int(landmark.x * frame_width), int(landmark.y * frame_height))
            cv2.circle(frame, center, 3, (0, 255, 255), -1, cv2.LINE_AA)


def create_hand_detector(config: DebugVideoConfig) -> Any:
    if config.hand_model:
        detector = TaskHandLandmarkerDetector(config)
        logging.info("Using hand detector backend: %s", detector.backend_name)
        return detector

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        detector = LegacySolutionsHandDetector(config)
        logging.info("Using hand detector backend: %s", detector.backend_name)
        return detector

    raise RuntimeError(
        "No hand detector backend is available. MediaPipe solutions.hands is unavailable in this environment, "
        "and no hand_landmarker.task model was provided. Put the model at "
        r"C:\Users\22500\Desktop\JAVA\skeleton-sp24\proj0\AeroInteract3D\hand_landmarker.task"
        " or pass --hand-model with a valid file."
    )


def landmark_to_vec3(landmark: Any) -> Vec3:
    return Vec3(
        x=max(-1.0, min(1.0, (float(landmark.x) - 0.5) * 2.0)),
        y=max(-1.0, min(1.0, (0.5 - float(landmark.y)) * 2.0)),
        z=max(-1.0, min(1.0, -float(landmark.z))),
    )


def extract_landmarks(hand_landmarks: Any) -> dict[str, Vec3]:
    if hasattr(hand_landmarks, "landmark"):
        source = hand_landmarks.landmark
    else:
        source = hand_landmarks
    points = [landmark_to_vec3(landmark) for landmark in source]
    return {name: points[index] for index, name in enumerate(LANDMARK_NAMES)}


def distance(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def open_capture(config: DebugVideoConfig) -> cv2.VideoCapture:
    if config.input_video is not None:
        return cv2.VideoCapture(config.input_video)
    assert config.camera_index is not None
    return cv2.VideoCapture(config.camera_index)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)
    try:
        run_live_preview(config)
        return 0
    except Exception:
        logging.exception("Gesture live preview failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))