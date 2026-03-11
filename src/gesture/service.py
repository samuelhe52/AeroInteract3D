from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import (
    GestureRuntimeConfig,
    create_hand_detector,
    default_hand_model_path,
    distance,
    resolve_hand_model_path,
)
from src.ports import GestureInputPort
from src.utils.contracts import EXPECTED_CONTRACT_VERSION
from src.utils.runtime import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_INITIALIZING,
    LIFECYCLE_RUNNING,
    LIFECYCLE_STOPPED,
    build_health,
    error_entry,
)

TRACKING_TEMPORARY_LOSS_FRAMES = 2
PINCH_ENTER_THRESHOLD = 0.10
PINCH_HOLD_THRESHOLD = 0.06
PINCH_RELEASE_THRESHOLD = 0.12
PINCH_CONFIRM_FRAMES = 2
RELEASE_CONFIRM_FRAMES = 2
SMOOTHING_ALPHA = 0.65
FRAME_SUMMARY_INTERVAL = 30


logger = logging.getLogger("gesture.service")


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@dataclass(slots=True)
class GestureMetrics:
    polls_attempted: int = 0
    packets_emitted: int = 0
    tracked_packets: int = 0
    empty_packets: int = 0
    backend_failures: int = 0


def _close_detector_resource(detector_owner: Any | None, detector: Any | None = None) -> None:
    if detector_owner is not None and hasattr(detector_owner, "__exit__"):
        detector_owner.__exit__(None, None, None)
        return

    if detector is not None and hasattr(detector, "close"):
        detector.close()
        return

    if detector_owner is not None and hasattr(detector_owner, "close"):
        detector_owner.close()


class GestureServiceImpl(GestureInputPort):
    DEFAULT_HAND_MODEL_PATH = default_hand_model_path()

    def __init__(
        self,
        camera_index: int = 0,
        hand_model: str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._repo_root = Path(__file__).resolve().parents[2]
        self._camera_index = camera_index
        self._hand_model = self._resolve_hand_model(hand_model)
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._backend: dict[str, Any] | None = None
        self._detector_backend = "uninitialized"

        self._started = False
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._hand_id = "hand-0"
        self._errors: list[dict[str, Any]] = []
        self._metrics = GestureMetrics()

        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._clock_timestamp_ms = 0
        self._capture_tick = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._last_pinch_distance = 0.0
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)

    def start(self) -> None:
        if self._started:
            logger.info("Gesture module already running, skipping duplicate start")
            return None

        logger.info(
            f"Starting gesture module: camera_index={self._camera_index}, "
            f"hand_model={self._hand_model}, "
            f"min_detection_confidence={self._min_detection_confidence:.2f}, "
            f"min_tracking_confidence={self._min_tracking_confidence:.2f}, "
            f"model_complexity={self._model_complexity}"
        )
        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._errors = []
        self._metrics = GestureMetrics()
        self._reset_runtime_state()

        try:
            self._setup_backend()
        except Exception as exc:
            self.lifecycle_state = LIFECYCLE_DEGRADED
            logger.exception(
                f"Gesture module startup failed: camera_index={self._camera_index}, "
                f"detector_backend={self._detector_backend}"
            )
            self._record_error(
                error_entry(
                    "gesture.backend.startup_failure",
                    "Failed to initialize gesture input backend",
                    recoverable=False,
                    hint="Verify camera access and the hand_landmarker.task path before restarting the service.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
            raise

        self._started = True
        self.lifecycle_state = LIFECYCLE_RUNNING
        logger.info(
            f"Gesture module started successfully, state switched to RUNNING: "
            f"camera_index={self._camera_index}, detector_backend={self._detector_backend}"
        )
        return None

    def poll(self) -> GesturePacket | None:
        if not self._started:
            logger.debug("Gesture service poll skipped because the service is not running")
            return None

        try:
            self._metrics.polls_attempted += 1
            raw_frame = self._read_frame()
            timestamp_ms = self._resolve_timestamp_ms(raw_frame)
            hand_data = self._detect_hand(raw_frame)
            previous_tracking_state = self._tracking_state
            previous_pinch_state = self._pinch_state

            if hand_data is None:
                self._tracking_loss_streak += 1
                tracking_state = self._compute_tracking_state(None)
                pinch_state = self._compute_pinch_state(None, None)
                confidence = self._compute_confidence(None, tracking_state, pinch_state)
                velocity = Vec3(0.0, 0.0, 0.0)

                self._tracking_state = tracking_state
                self._pinch_state = pinch_state
                self._confidence = confidence
                self._last_velocity = velocity

                packet = self._build_packet(
                    timestamp_ms=timestamp_ms,
                    tracking_state=tracking_state,
                    pinch_state=pinch_state,
                    confidence=confidence,
                    index_tip=self._last_index_tip,
                    thumb_tip=self._last_thumb_tip,
                    palm_center=self._last_palm_center,
                    velocity=velocity,
                    debug={
                        "backend": self._detector_backend,
                        "tick": raw_frame.get("tick"),
                        "has_hand": False,
                        "tracking_loss_streak": self._tracking_loss_streak,
                    },
                )
                self.lifecycle_state = LIFECYCLE_RUNNING
                self._metrics.packets_emitted += 1
                self._metrics.empty_packets += 1
                self._log_state_changes(
                    previous_tracking_state=previous_tracking_state,
                    tracking_state=tracking_state,
                    previous_pinch_state=previous_pinch_state,
                    pinch_state=pinch_state,
                    confidence=confidence,
                    timestamp_ms=timestamp_ms,
                    hand_detected=False,
                )
                self._log_frame_summary(packet, hand_detected=False)
                return packet

            self._tracking_loss_streak = 0

            index_tip = self._normalize_vec3(hand_data["index_tip"])
            thumb_tip = self._normalize_vec3(hand_data["thumb_tip"])
            palm_center = self._normalize_vec3(hand_data["palm_center"])

            tracking_state = self._compute_tracking_state(hand_data)
            pinch_state = self._compute_pinch_state(index_tip, thumb_tip)
            confidence = self._compute_confidence(hand_data, tracking_state, pinch_state)

            smoothed_index_tip = self._smooth_vec3(index_tip, self._last_index_tip)
            smoothed_thumb_tip = self._smooth_vec3(thumb_tip, self._last_thumb_tip)
            smoothed_palm_center = self._smooth_vec3(palm_center, self._last_palm_center)
            velocity = self._compute_velocity(smoothed_palm_center, self._last_palm_center)

            self._last_index_tip = smoothed_index_tip
            self._last_thumb_tip = smoothed_thumb_tip
            self._last_palm_center = smoothed_palm_center
            self._last_velocity = velocity
            self._tracking_state = tracking_state
            self._pinch_state = pinch_state
            self._confidence = confidence

            packet = self._build_packet(
                timestamp_ms=timestamp_ms,
                tracking_state=tracking_state,
                pinch_state=pinch_state,
                confidence=confidence,
                index_tip=smoothed_index_tip,
                thumb_tip=smoothed_thumb_tip,
                palm_center=smoothed_palm_center,
                velocity=velocity,
                debug={
                    "backend": self._detector_backend,
                    "tick": raw_frame.get("tick"),
                    "has_hand": True,
                    "raw_confidence": hand_data.get("raw_confidence"),
                },
            )
            self.lifecycle_state = LIFECYCLE_RUNNING
            self._metrics.packets_emitted += 1
            self._metrics.tracked_packets += 1
            self._log_state_changes(
                previous_tracking_state=previous_tracking_state,
                tracking_state=tracking_state,
                previous_pinch_state=previous_pinch_state,
                pinch_state=pinch_state,
                confidence=confidence,
                timestamp_ms=timestamp_ms,
                hand_detected=True,
            )
            self._log_frame_summary(packet, hand_detected=True)
            return packet
        except Exception as exc:
            self.lifecycle_state = LIFECYCLE_DEGRADED
            self._metrics.backend_failures += 1
            logger.exception(
                f"Gesture polling failed, module switched to DEGRADED: "
                f"camera_index={self._camera_index}, detector_backend={self._detector_backend}, "
                f"polls_attempted={self._metrics.polls_attempted}"
            )
            self._record_error(
                error_entry(
                    "gesture.backend.failure",
                    "Gesture input backend failure",
                    recoverable=False,
                    hint="Inspect camera availability and detector health before resuming polling.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
            return None

    def health(self) -> dict[str, Any]:
        last_error = self._errors[-1]["message"] if self._errors else None
        return build_health(
            component="gesture",
            lifecycle_state=self.lifecycle_state,
            errors=self._errors,
            stats={
                "started": self._started,
                "frame_id": self._frame_id,
                "hand_id": self._hand_id,
                "tracking_state": self._tracking_state,
                "pinch_state": self._pinch_state,
                "confidence": self._confidence,
                "tracking_loss_streak": self._tracking_loss_streak,
                "detector_backend": self._detector_backend,
                "last_error": last_error,
                "polls_attempted": self._metrics.polls_attempted,
                "packets_emitted": self._metrics.packets_emitted,
                "tracked_packets": self._metrics.tracked_packets,
                "empty_packets": self._metrics.empty_packets,
                "backend_failures": self._metrics.backend_failures,
            },
        )

    def stop(self) -> None:
        if not self._started and self.lifecycle_state == LIFECYCLE_STOPPED:
            logger.info("Gesture module already stopped, no need for repeated operation")
            return None

        logger.info(
            f"Stopping gesture module: camera_index={self._camera_index}, "
            f"detector_backend={self._detector_backend}, frame_id={self._frame_id}"
        )
        try:
            self._teardown_backend()
        except Exception as exc:
            logger.warning(
                f"Gesture backend teardown failed: camera_index={self._camera_index}, "
                f"detector_backend={self._detector_backend}, error={exc}"
            )
            self._record_error(
                error_entry(
                    "gesture.backend.teardown_failure",
                    "Failed to release gesture backend resources cleanly",
                    recoverable=True,
                    hint="Release the capture device manually if the process still holds the camera.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
        finally:
            self._backend = None
            self._started = False
            self.lifecycle_state = LIFECYCLE_STOPPED
            logger.info("Gesture module stopped, all resources released")
        return None

    def _reset_runtime_state(self) -> None:
        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._clock_timestamp_ms = 0
        self._capture_tick = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._last_pinch_distance = 0.0
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)

    def _setup_backend(self) -> None:
        if self._hand_model is None:
            logger.error("Gesture module startup failed: no hand model could be resolved")
            raise RuntimeError(
                "Missing hand_landmarker.task. Put the model at "
                f"{self.DEFAULT_HAND_MODEL_PATH} or pass --hand-model with a valid file path."
            )

        logger.info(
            f"Opening gesture backend resources: camera_index={self._camera_index}, "
            f"hand_model={self._hand_model}"
        )
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            logger.error(f"Unable to open camera source: {self._camera_index}")
            raise RuntimeError(f"Unable to open camera source: {self._camera_index}")

        detector_config = GestureRuntimeConfig(
            input_video=None,
            camera_index=self._camera_index,
            hand_model=self._hand_model,
            output_dir=self._repo_root / "report" / "live",
            target_fps=None,
            max_frames=0,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            model_complexity=self._model_complexity,
        )
        detector_owner: Any | None = None
        detector: Any | None = None
        try:
            detector_owner = create_hand_detector(detector_config)
            detector = detector_owner.__enter__() if hasattr(detector_owner, "__enter__") else detector_owner
            backend_name = getattr(detector, "backend_name", type(detector).__name__)
        except Exception:
            _close_detector_resource(detector_owner, detector)
            capture.release()
            raise
        if backend_name == "passthrough":
            _close_detector_resource(detector_owner, detector)
            capture.release()
            logger.error("Gesture detector resolved to passthrough backend; refusing to start")
            raise RuntimeError(
                "No MediaPipe hand detector backend is available. "
                f"Put the model at {self.DEFAULT_HAND_MODEL_PATH} or pass --hand-model with a valid hand_landmarker.task file."
            )

        self._backend = {
            "capture": capture,
            "detector": detector,
            "detector_owner": detector_owner,
        }
        self._detector_backend = backend_name
        logger.info(
            f"Gesture backend ready: camera_index={self._camera_index}, "
            f"detector_backend={self._detector_backend}"
        )

    def _teardown_backend(self) -> None:
        if self._backend is None:
            self._detector_backend = "uninitialized"
            return

        logger.debug("Releasing gesture backend resources")
        detector_owner = self._backend.get("detector_owner")
        capture = self._backend.get("capture")
        if detector_owner is not None and hasattr(detector_owner, "__exit__"):
            detector_owner.__exit__(None, None, None)
        if capture is not None:
            capture.release()
        self._backend = None
        self._detector_backend = "uninitialized"

    def _read_frame(self) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("Gesture backend is not initialized")

        capture = self._backend["capture"]
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")

        self._capture_tick += 1
        return {
            "timestamp_ms": self._next_timestamp_ms(),
            "frame": frame,
            "tick": self._capture_tick,
            "source": f"camera:{self._camera_index}",
        }

    def _detect_hand(self, raw_frame: dict[str, Any]) -> dict[str, Any] | None:
        if self._backend is None:
            raise RuntimeError("Gesture backend is not initialized")

        frame = raw_frame["frame"]
        detector = self._backend["detector"]
        timestamp_ms = int(raw_frame["timestamp_ms"])
        landmarks, raw_confidence, _raw_landmarks = detector.detect(frame, timestamp_ms)
        if landmarks is None:
            return None

        return {
            "index_tip": landmarks["index_finger_tip"],
            "thumb_tip": landmarks["thumb_tip"],
            "palm_center": landmarks["wrist"],
            "raw_confidence": raw_confidence,
            "source": self._detector_backend,
        }

    def _normalize_vec3(self, value: Vec3) -> Vec3:
        return Vec3(
            x=max(-1.0, min(1.0, float(value.x))),
            y=max(-1.0, min(1.0, float(value.y))),
            z=max(-1.0, min(1.0, float(value.z))),
        )

    def _compute_tracking_state(self, hand_data: dict[str, Any] | None) -> str:
        if hand_data is not None:
            return "tracked"
        if self._tracking_loss_streak <= TRACKING_TEMPORARY_LOSS_FRAMES:
            return "temporarily_lost"
        return "not_detected"

    def _compute_pinch_state(self, index_tip: Vec3 | None, thumb_tip: Vec3 | None) -> str:
        if index_tip is None or thumb_tip is None:
            self._pinch_candidate_streak = 0
            if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"}:
                self._release_candidate_streak += 1
                if self._release_candidate_streak <= RELEASE_CONFIRM_FRAMES:
                    return "release_candidate"
            self._release_candidate_streak = 0
            self._last_pinch_distance = 0.0
            return "open"

        pinch_distance = distance(index_tip, thumb_tip)
        self._last_pinch_distance = pinch_distance

        if pinch_distance <= PINCH_HOLD_THRESHOLD:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_candidate_streak >= PINCH_CONFIRM_FRAMES:
                return "pinched"
            return "pinch_candidate"

        if pinch_distance <= PINCH_ENTER_THRESHOLD:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_state == "pinched":
                return "pinched"
            return "pinch_candidate"

        self._pinch_candidate_streak = 0
        if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"} and pinch_distance >= PINCH_RELEASE_THRESHOLD:
            self._release_candidate_streak += 1
            if self._release_candidate_streak <= RELEASE_CONFIRM_FRAMES:
                return "release_candidate"
            self._release_candidate_streak = 0
            return "open"

        self._release_candidate_streak = 0
        return "open"

    def _compute_confidence(
        self,
        hand_data: dict[str, Any] | None,
        tracking_state: str,
        pinch_state: str,
    ) -> float:
        if tracking_state == "temporarily_lost":
            return max(0.05, 0.3 - 0.05 * max(self._tracking_loss_streak - 1, 0))
        if tracking_state == "not_detected":
            return 0.0

        assert hand_data is not None
        raw_confidence = float(hand_data.get("raw_confidence", 0.8))
        if pinch_state == "pinched":
            raw_confidence += 0.05
        elif pinch_state in {"pinch_candidate", "release_candidate"}:
            raw_confidence -= 0.05
        return max(0.0, min(1.0, raw_confidence))

    def _smooth_vec3(self, current: Vec3, previous: Vec3) -> Vec3:
        if self._frame_id == 0:
            return current
        alpha = SMOOTHING_ALPHA
        return Vec3(
            x=current.x * alpha + previous.x * (1.0 - alpha),
            y=current.y * alpha + previous.y * (1.0 - alpha),
            z=current.z * alpha + previous.z * (1.0 - alpha),
        )

    def _compute_velocity(self, current: Vec3, previous: Vec3) -> Vec3:
        return self._normalize_vec3(
            Vec3(
                x=current.x - previous.x,
                y=current.y - previous.y,
                z=current.z - previous.z,
            )
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
        debug: dict[str, Any] | None,
    ) -> GesturePacket:
        self._frame_id += 1
        self._last_timestamp_ms = timestamp_ms
        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self._hand_id,
            tracking_state=tracking_state,
            confidence=max(0.0, min(1.0, confidence)),
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            coordinate_space="camera_norm",
            pinch_distance=self._last_pinch_distance,
            velocity=velocity,
            smoothing_hint={"method": "linear", "alpha": SMOOTHING_ALPHA},
            debug=debug,
        )

    def _resolve_timestamp_ms(self, raw_frame: dict[str, Any]) -> int:
        raw_timestamp = raw_frame.get("timestamp_ms")
        timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else self._next_timestamp_ms()
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        if timestamp_ms > self._clock_timestamp_ms:
            self._clock_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _next_timestamp_ms(self) -> int:
        now = int(time.time() * 1000)
        if now <= self._clock_timestamp_ms:
            now = self._clock_timestamp_ms + 1
        self._clock_timestamp_ms = now
        return now

    def _resolve_hand_model(self, hand_model: str | None) -> str | None:
        return resolve_hand_model_path(hand_model)

    def _record_error(self, error: dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-10:]

    def _log_state_changes(
        self,
        previous_tracking_state: str,
        tracking_state: str,
        previous_pinch_state: str,
        pinch_state: str,
        confidence: float,
        timestamp_ms: int,
        hand_detected: bool,
    ) -> None:
        if tracking_state != previous_tracking_state:
            logger.info(
                f"Tracking state changed: {previous_tracking_state} -> {tracking_state} "
                f"at frame_id={self._frame_id + 1}, timestamp_ms={timestamp_ms}, "
                f"hand_detected={hand_detected}, confidence={confidence:.3f}"
            )

        if pinch_state != previous_pinch_state:
            logger.info(
                f"Pinch state changed: {previous_pinch_state} -> {pinch_state} "
                f"at frame_id={self._frame_id + 1}, timestamp_ms={timestamp_ms}, "
                f"pinch_distance={self._last_pinch_distance:.4f}, confidence={confidence:.3f}"
            )

    def _log_frame_summary(self, packet: GesturePacket, hand_detected: bool) -> None:
        if packet.frame_id % FRAME_SUMMARY_INTERVAL != 0:
            return

        logger.debug(
            f"Gesture frame summary: frame_id={packet.frame_id}, "
            f"timestamp_ms={packet.timestamp_ms}, tracking_state={packet.tracking_state}, "
            f"pinch_state={packet.pinch_state}, confidence={packet.confidence:.3f}, "
            f"hand_detected={hand_detected}, backend={self._detector_backend}"
        )
