from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.constants import (
    DEBUG_FPS_SAMPLE_WINDOW,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_TARGET_FPS,
    GESTURE_FRAME_SUMMARY_INTERVAL,
    MAX_ERROR_HISTORY,
)
from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import (
    GestureRuntimeConfig,
    create_capture,
    create_hand_detector,
    default_hand_model_path,
    estimate_camera_depth_from_hand_scale,
    estimate_hand_scale,
    estimate_palm_anchor,
    resolve_hand_model_path,
)
from src.gesture.debug.live_preview_runtime import (
    overlay_anchor_points,
    overlay_fps,
    overlay_packet,
)
from src.gesture.temporal import (
    GestureFrameAnalysis,
    GestureTemporalReducer,
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
    preview_frames_rendered: int = 0
    preview_failures: int = 0


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
        camera_index: int = DEFAULT_CAMERA_INDEX,
        target_fps: float | None = float(DEFAULT_TARGET_FPS),
        frame_width: int | None = DEFAULT_FRAME_WIDTH,
        frame_height: int | None = DEFAULT_FRAME_HEIGHT,
        hand_model: str | None = None,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
        model_complexity: int = DEFAULT_MODEL_COMPLEXITY,
        preview_enabled: bool = False,
        preview_window_name: str = "Gesture Live Preview",
        preview_mirror: bool = True,
        preview_draw_coordinates: bool = True,
    ) -> None:
        self._repo_root = Path(__file__).resolve().parents[2]
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._hand_model = self._resolve_hand_model(hand_model)
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._preview_requested = preview_enabled
        self._preview_active = preview_enabled
        self._preview_window_name = preview_window_name
        self._preview_mirror = preview_mirror
        self._preview_draw_coordinates = preview_draw_coordinates
        self._backend: dict[str, Any] | None = None
        self._detector_backend = "uninitialized"

        self._started = False
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._hand_id = "hand-0"
        self._errors: list[dict[str, Any]] = []
        self._metrics = GestureMetrics()
        self._analyzer = GestureTemporalReducer()

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
        self._last_wrist = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)
        self._preview_window_open = False
        self._preview_closed_by_user = False
        self._preview_smoothed_fps = 0.0
        self._preview_last_frame_started_at: float | None = None
        self._preview_window_property_check_interval = 15

    def start(self) -> None:
        if self._started:
            logger.info("Gesture module already running, skipping duplicate start")
            return None

        logger.info(
            f"Starting gesture module: camera_index={self._camera_index}, "
            f"target_fps={self._target_fps}, "
            f"frame_width={self._frame_width}, frame_height={self._frame_height}, "
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

        self._ensure_preview_window()

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
            raw_confidence = 0.0 if hand_data is None else float(hand_data.get("raw_confidence", 0.0))

            analysis = self._analyzer.process(
                timestamp_ms=timestamp_ms,
                index_tip=None if hand_data is None else hand_data["index_tip"],
                thumb_tip=None if hand_data is None else hand_data["thumb_tip"],
                wrist=None if hand_data is None else hand_data["wrist"],
                raw_confidence=raw_confidence,
            )
            self._sync_analysis_state(analysis)

            if hand_data is None:
                packet = self._build_packet(
                    timestamp_ms=analysis.timestamp_ms,
                    tracking_state=analysis.tracking_state,
                    pinch_state=analysis.pinch_state,
                    confidence=analysis.confidence,
                    index_tip=analysis.index_tip,
                    thumb_tip=analysis.thumb_tip,
                    wrist=analysis.wrist,
                    velocity=analysis.velocity,
                    smoothing_hint=analysis.smoothing_hint,
                    debug={
                        "backend": self._detector_backend,
                        "tick": raw_frame.get("tick"),
                        "has_hand": False,
                        "tracking_loss_streak": analysis.tracking_loss_streak,
                    },
                )
                self.lifecycle_state = LIFECYCLE_RUNNING
                self._metrics.packets_emitted += 1
                self._metrics.empty_packets += 1
                self._log_state_changes(
                    previous_tracking_state=previous_tracking_state,
                    tracking_state=analysis.tracking_state,
                    previous_pinch_state=previous_pinch_state,
                    pinch_state=analysis.pinch_state,
                    confidence=analysis.confidence,
                    timestamp_ms=analysis.timestamp_ms,
                    hand_detected=False,
                )
                self._maybe_render_live_preview(raw_frame, hand_data, packet)
                self._log_frame_summary(packet, hand_detected=False)
                return packet

            packet = self._build_packet(
                timestamp_ms=analysis.timestamp_ms,
                tracking_state=analysis.tracking_state,
                pinch_state=analysis.pinch_state,
                confidence=analysis.confidence,
                index_tip=analysis.index_tip,
                thumb_tip=analysis.thumb_tip,
                wrist=analysis.wrist,
                velocity=analysis.velocity,
                smoothing_hint=analysis.smoothing_hint,
                debug={
                    "backend": self._detector_backend,
                    "tick": raw_frame.get("tick"),
                    "has_hand": True,
                    "raw_confidence": hand_data.get("raw_confidence"),
                    "camera_depth": hand_data.get("camera_depth"),
                    "hand_scale": hand_data.get("hand_scale"),
                },
            )
            self.lifecycle_state = LIFECYCLE_RUNNING
            self._metrics.packets_emitted += 1
            self._metrics.tracked_packets += 1
            self._log_state_changes(
                previous_tracking_state=previous_tracking_state,
                tracking_state=analysis.tracking_state,
                previous_pinch_state=previous_pinch_state,
                pinch_state=analysis.pinch_state,
                confidence=analysis.confidence,
                timestamp_ms=analysis.timestamp_ms,
                hand_detected=True,
            )
            self._maybe_render_live_preview(raw_frame, hand_data, packet)
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
                "target_fps": self._target_fps,
                "frame_width": self._frame_width,
                "frame_height": self._frame_height,
                "tracking_state": self._tracking_state,
                "pinch_state": self._pinch_state,
                "confidence": self._confidence,
                "tracking_loss_streak": self._tracking_loss_streak,
                "detector_backend": self._detector_backend,
                "preview_requested": self._preview_requested,
                "preview_active": self._preview_active,
                "preview_window_open": self._preview_window_open,
                "last_error": last_error,
                "polls_attempted": self._metrics.polls_attempted,
                "packets_emitted": self._metrics.packets_emitted,
                "tracked_packets": self._metrics.tracked_packets,
                "empty_packets": self._metrics.empty_packets,
                "backend_failures": self._metrics.backend_failures,
                "preview_frames_rendered": self._metrics.preview_frames_rendered,
                "preview_failures": self._metrics.preview_failures,
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
            self._close_preview_window()
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
        self._last_wrist = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)
        self._preview_active = self._preview_requested
        self._preview_window_open = False
        self._preview_closed_by_user = False
        self._preview_smoothed_fps = 0.0
        self._preview_last_frame_started_at = None
        self._analyzer.reset()

    def _setup_backend(self) -> None:
        if self._hand_model is None:
            logger.error("Gesture module startup failed: no hand model could be resolved")
            raise RuntimeError(
                "Missing hand_landmarker.task. Put the model at "
                f"{self.DEFAULT_HAND_MODEL_PATH} or pass --hand-model with a valid file path."
            )

        logger.info(
            f"Opening gesture backend resources: camera_index={self._camera_index}, "
            f"target_fps={self._target_fps}, "
            f"frame_width={self._frame_width}, frame_height={self._frame_height}, "
            f"hand_model={self._hand_model}"
        )
        capture = create_capture(
            camera_index=self._camera_index,
            target_fps=self._target_fps,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
        )
        if not capture.isOpened():
            capture.release()
            logger.error(f"Unable to open camera source: {self._camera_index}")
            raise RuntimeError(f"Unable to open camera source: {self._camera_index}")

        detector_config = GestureRuntimeConfig(
            input_video=None,
            camera_index=self._camera_index,
            hand_model=self._hand_model,
            output_dir=self._repo_root / "report" / "live",
            target_fps=self._target_fps,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
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

    def _ensure_preview_window(self) -> None:
        if not self._preview_active or self._preview_closed_by_user or self._preview_window_open:
            return

        try:
            cv2.namedWindow(self._preview_window_name, cv2.WINDOW_NORMAL)
            self._preview_window_open = True
        except Exception as exc:
            self._metrics.preview_failures += 1
            self._preview_active = False
            logger.warning("Live preview unavailable, disabling preview window: %s", exc)
            self._record_error(
                error_entry(
                    "gesture.preview.unavailable",
                    "Unable to open the live preview window",
                    recoverable=True,
                    hint="Disable --live-preview or verify that OpenCV GUI support is available.",
                    details={
                        "window_name": self._preview_window_name,
                        "error": str(exc),
                    },
                )
            )

    def _close_preview_window(self) -> None:
        if not self._preview_window_open:
            return

        try:
            cv2.destroyWindow(self._preview_window_name)
        except Exception:
            logger.debug("Preview window destroy skipped after OpenCV close failure", exc_info=True)
        finally:
            self._preview_window_open = False

    def _maybe_render_live_preview(
        self,
        raw_frame: dict[str, Any],
        hand_data: dict[str, Any] | None,
        packet: GesturePacket,
    ) -> None:
        if not self._preview_active:
            return

        try:
            self._render_live_preview(raw_frame, hand_data, packet)
        except Exception as exc:
            self._metrics.preview_failures += 1
            self._preview_active = False
            self._close_preview_window()
            logger.warning("Live preview disabled after render failure: %s", exc)
            self._record_error(
                error_entry(
                    "gesture.preview.failure",
                    "Live preview disabled after a rendering failure",
                    recoverable=True,
                    hint="Disable --live-preview or verify that OpenCV GUI support is working.",
                    details={
                        "window_name": self._preview_window_name,
                        "error": str(exc),
                    },
                )
            )

    def _render_live_preview(
        self,
        raw_frame: dict[str, Any],
        hand_data: dict[str, Any] | None,
        packet: GesturePacket,
    ) -> None:
        self._ensure_preview_window()
        if not self._preview_window_open or self._backend is None:
            return

        detector = self._backend.get("detector")
        display_frame = raw_frame["frame"].copy()
        raw_landmarks = None if hand_data is None else hand_data.get("raw_landmarks")

        if detector is not None and hasattr(detector, "draw"):
            detector.draw(display_frame, raw_landmarks)

        if self._preview_mirror:
            display_frame = cv2.flip(display_frame, 1)

        overlay_packet(display_frame, packet)
        overlay_fps(display_frame, self._compute_preview_fps())
        overlay_anchor_points(display_frame, packet, self._preview_mirror, self._preview_draw_coordinates)
        cv2.imshow(self._preview_window_name, display_frame)

        if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
            logger.info("Live preview window closed by user request")
            self._preview_active = False
            self._preview_closed_by_user = True
            self._close_preview_window()
            return

        self._metrics.preview_frames_rendered += 1

        if self._metrics.preview_frames_rendered % self._preview_window_property_check_interval != 0:
            return

        try:
            if cv2.getWindowProperty(self._preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Live preview window was closed by the window manager")
                self._preview_active = False
                self._preview_closed_by_user = True
                self._close_preview_window()
        except Exception:
            logger.debug("Preview visibility check skipped after OpenCV property lookup failure", exc_info=True)

    def _compute_preview_fps(self) -> float:
        now = time.perf_counter()
        if self._preview_last_frame_started_at is None:
            self._preview_last_frame_started_at = now
            return 0.0

        frame_elapsed = max(now - self._preview_last_frame_started_at, 1e-6)
        instantaneous_fps = 1.0 / frame_elapsed
        if self._preview_smoothed_fps == 0.0:
            self._preview_smoothed_fps = instantaneous_fps
        else:
            self._preview_smoothed_fps += (instantaneous_fps - self._preview_smoothed_fps) / DEBUG_FPS_SAMPLE_WINDOW
        self._preview_last_frame_started_at = now
        return self._preview_smoothed_fps

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
        landmarks, raw_confidence, raw_landmarks = detector.detect(frame, timestamp_ms)
        if landmarks is None:
            return None

        return {
            "index_tip": landmarks["index_finger_tip"],
            "thumb_tip": landmarks["thumb_tip"],
            "wrist": estimate_palm_anchor(landmarks),
            "raw_confidence": raw_confidence,
            "camera_depth": estimate_camera_depth_from_hand_scale(landmarks),
            "hand_scale": estimate_hand_scale(landmarks),
            "raw_landmarks": raw_landmarks,
            "source": self._detector_backend,
        }

    def _sync_analysis_state(self, analysis: GestureFrameAnalysis) -> None:
        self._tracking_state = analysis.tracking_state
        self._pinch_state = analysis.pinch_state
        self._confidence = analysis.confidence
        self._tracking_loss_streak = analysis.tracking_loss_streak
        self._last_pinch_distance = analysis.pinch_distance
        self._last_index_tip = analysis.index_tip
        self._last_thumb_tip = analysis.thumb_tip
        self._last_wrist = analysis.wrist
        self._last_velocity = analysis.velocity

    def _build_packet(
        self,
        timestamp_ms: int,
        tracking_state: str,
        pinch_state: str,
        confidence: float,
        index_tip: Vec3,
        thumb_tip: Vec3,
        wrist: Vec3,
        velocity: Vec3,
        smoothing_hint: dict[str, Any],
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
            wrist=wrist,
            coordinate_space="camera_norm",
            pinch_distance=self._last_pinch_distance,
            velocity=velocity,
            smoothing_hint=smoothing_hint,
            debug=debug,
        )

    def _resolve_timestamp_ms(self, raw_frame: dict[str, Any]) -> int:
        raw_timestamp = raw_frame.get("timestamp_ms")
        timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else self._next_timestamp_ms()
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
        self._errors = self._errors[-MAX_ERROR_HISTORY:]

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
        if packet.frame_id % GESTURE_FRAME_SUMMARY_INTERVAL != 0:
            return

        logger.debug(
            f"Gesture frame summary: frame_id={packet.frame_id}, "
            f"timestamp_ms={packet.timestamp_ms}, tracking_state={packet.tracking_state}, "
            f"pinch_state={packet.pinch_state}, confidence={packet.confidence:.3f}, "
            f"hand_detected={hand_detected}, backend={self._detector_backend}"
        )
