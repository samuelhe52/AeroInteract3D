from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

from src.constants import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_TARGET_FPS,
    GESTURE_FRAME_SUMMARY_INTERVAL,
    GESTURE_SMOOTHING_PRESET,
    MAX_ERROR_HISTORY,
)
from src.contracts import GesturePacket
from src.gesture.runtime import CaptureRuntime, HandLandmarkerRuntime, RawHandObservation
from src.gesture.temporal import SmoothingPreset, TemporalReducer, temporal_tuning_for_preset
from src.ports import GestureInputPort
from src.utils.contracts import validate_gesture_packet
from src.utils.runtime import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_INITIALIZING,
    LIFECYCLE_RUNNING,
    LIFECYCLE_STOPPED,
    build_health,
    error_entry,
)


logger = logging.getLogger("gesture.service")


@dataclass(slots=True)
class GestureMetrics:
    polls: int = 0
    packets_emitted: int = 0
    tracked_packets: int = 0
    temporary_loss_packets: int = 0
    not_detected_packets: int = 0
    capture_failures: int = 0
    detector_failures: int = 0
    validation_failures: int = 0


@dataclass(slots=True)
class GestureConfig:
    camera_index: int = 0
    target_fps: float = DEFAULT_TARGET_FPS
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    model_path: str | None = None
    preview_enabled: bool = False
    smoothing_preset: SmoothingPreset = GESTURE_SMOOTHING_PRESET


class GestureServiceImpl(GestureInputPort):
    def __init__(
        self,
        *,
        camera_index: int = 0,
        target_fps: float = DEFAULT_TARGET_FPS,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE,
        model_path: str | None = None,
        preview_enabled: bool = False,
        smoothing_preset: SmoothingPreset = GESTURE_SMOOTHING_PRESET,
        capture_factory: Callable[..., Any] = CaptureRuntime,
        detector_factory: Callable[..., Any] = HandLandmarkerRuntime,
        preview_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = GestureConfig(
            camera_index=camera_index,
            target_fps=target_fps,
            frame_width=frame_width,
            frame_height=frame_height,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            preview_enabled=preview_enabled,
            smoothing_preset=smoothing_preset,
        )
        self._capture_factory = capture_factory
        self._detector_factory = detector_factory
        self._preview_factory = preview_factory
        self._clock = clock or time.perf_counter
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._reducer = TemporalReducer(tuning=temporal_tuning_for_preset(smoothing_preset))
        self._capture: Any | None = None
        self._detector: Any | None = None
        self._preview: Any | None = None
        self._errors: list[dict[str, Any]] = []
        self._logged_error_codes: set[str] = set()
        self._metrics = GestureMetrics()
        self._last_packet: GesturePacket | None = None
        self._last_preview_frame = None
        self._frame_id = 0
        self._last_timestamp_ms = 0

    def start(self) -> None:
        if self.lifecycle_state == LIFECYCLE_RUNNING:
            return None

        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._errors = []
        self._logged_error_codes = set()
        self._metrics = GestureMetrics()
        self._reducer.reset()
        self._last_packet = None
        self._last_preview_frame = None
        self._frame_id = 0
        self._last_timestamp_ms = 0

        self._capture = self._build_capture()
        self._detector = self._build_detector()
        self._preview = self._build_preview() if self._config.preview_enabled else None

        if self._capture is None or self._detector is None:
            self.lifecycle_state = LIFECYCLE_DEGRADED
        else:
            self.lifecycle_state = LIFECYCLE_RUNNING
        return None

    def poll(self) -> GesturePacket | None:
        if self.lifecycle_state not in {LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED}:
            raise RuntimeError("Gesture service must be running before polling")

        self._metrics.polls += 1
        self._frame_id += 1
        timestamp_ms = self._next_timestamp_ms()
        frame = self._read_frame()
        observation: RawHandObservation | None = None

        if frame is not None and self._detector is not None:
            observation = self._detect(frame, timestamp_ms=timestamp_ms)

        packet = self._reducer.reduce(observation, frame_id=self._frame_id, timestamp_ms=timestamp_ms)
        validation_errors = validate_gesture_packet(packet)
        if validation_errors:
            self._metrics.validation_failures += len(validation_errors)
            for validation_error in validation_errors:
                self._record_error(validation_error)
            self.lifecycle_state = LIFECYCLE_DEGRADED

        self._last_packet = packet
        self._record_packet_metrics(packet)
        self._maybe_render_preview(frame, observation, packet)

        if packet.frame_id % GESTURE_FRAME_SUMMARY_INTERVAL == 0:
            logger.debug(
                "Gesture summary frame=%s tracking=%s pinch=%s confidence=%.3f",
                packet.frame_id,
                packet.tracking_state,
                packet.pinch_state,
                packet.confidence,
            )

        if self.lifecycle_state == LIFECYCLE_RUNNING and packet.tracking_state != "tracked" and self._capture is None:
            self.lifecycle_state = LIFECYCLE_DEGRADED

        self._metrics.packets_emitted += 1
        return packet

    def health(self) -> dict[str, Any]:
        return build_health(
            component="gesture",
            lifecycle_state=self.lifecycle_state,
            errors=self._errors,
            stats={
                "polls": self._metrics.polls,
                "packets_emitted": self._metrics.packets_emitted,
                "tracked_packets": self._metrics.tracked_packets,
                "temporary_loss_packets": self._metrics.temporary_loss_packets,
                "not_detected_packets": self._metrics.not_detected_packets,
                "capture_failures": self._metrics.capture_failures,
                "detector_failures": self._metrics.detector_failures,
                "validation_failures": self._metrics.validation_failures,
                "preview_enabled": self._config.preview_enabled,
                "smoothing_preset": self._config.smoothing_preset,
                "last_frame_id": None if self._last_packet is None else self._last_packet.frame_id,
                "last_tracking_state": None if self._last_packet is None else self._last_packet.tracking_state,
                "last_pinch_state": None if self._last_packet is None else self._last_packet.pinch_state,
            },
        )

    def stop(self) -> None:
        for component in (self._preview, self._detector, self._capture):
            if component is None:
                continue
            try:
                close = getattr(component, "close", None)
                if callable(close):
                    close()
            except Exception:
                logger.exception("Gesture component shutdown error")

        self._preview = None
        self._detector = None
        self._capture = None
        self.lifecycle_state = LIFECYCLE_STOPPED
        return None

    @property
    def preview_is_open(self) -> bool:
        if self._preview is None:
            return False
        return bool(getattr(self._preview, "is_open", False))

    def _build_capture(self) -> Any | None:
        try:
            return self._capture_factory(
                camera_index=self._config.camera_index,
                frame_width=self._config.frame_width,
                frame_height=self._config.frame_height,
                target_fps=self._config.target_fps,
            )
        except Exception as exc:
            self._metrics.capture_failures += 1
            self._record_error(
                error_entry(
                    "gesture.capture.start_failed",
                    "Unable to initialize camera capture",
                    recoverable=True,
                    hint="Check camera permissions, index, and whether another process owns the device.",
                    details={"error": str(exc), "camera_index": self._config.camera_index},
                )
            )
            logger.exception("Failed to initialize gesture capture")
            return None

    def _build_detector(self) -> Any | None:
        try:
            return self._detector_factory(
                model_path=self._config.model_path,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        except Exception as exc:
            self._metrics.detector_failures += 1
            self._record_error(
                error_entry(
                    "gesture.detector.start_failed",
                    "Unable to initialize hand detector",
                    recoverable=True,
                    hint="Verify the MediaPipe installation and model path before starting the app.",
                    details={"error": str(exc)},
                )
            )
            logger.exception("Failed to initialize gesture detector")
            return None

    def _build_preview(self) -> Any | None:
        preview_factory = self._preview_factory
        if preview_factory is None:
            from src.gesture.debug.live_preview_runtime import GesturePreviewWindow

            preview_factory = GesturePreviewWindow

        try:
            return preview_factory()
        except Exception as exc:
            self._record_error(
                error_entry(
                    "gesture.preview.start_failed",
                    "Unable to initialize gesture preview",
                    recoverable=True,
                    hint="Disable preview if the current environment cannot open an OpenCV window.",
                    details={"error": str(exc)},
                )
            )
            logger.exception("Failed to initialize gesture preview")
            return None

    def _read_frame(self):
        if self._capture is None:
            return None
        frame = self._capture.read()
        if frame is None:
            self._metrics.capture_failures += 1
            self.lifecycle_state = LIFECYCLE_DEGRADED
            self._record_error(
                error_entry(
                    "gesture.capture.read_failed",
                    "Unable to read a frame from the camera",
                    recoverable=True,
                    hint="The service will emit loss packets until camera frames become available again.",
                    details={"frame_id": self._frame_id},
                )
            )
        return frame

    def _detect(self, frame, *, timestamp_ms: int) -> RawHandObservation | None:
        try:
            observation = self._detector.detect(frame, timestamp_ms=timestamp_ms)
        except Exception as exc:
            self._metrics.detector_failures += 1
            self.lifecycle_state = LIFECYCLE_DEGRADED
            self._record_error(
                error_entry(
                    "gesture.detector.detect_failed",
                    "Hand detector failed while processing a frame",
                    recoverable=True,
                    hint="The gesture service will continue emitting degraded tracking packets.",
                    details={"error": str(exc), "frame_id": self._frame_id},
                )
            )
            self._log_recoverable_error_once(
                code="gesture.detector.detect_failed",
                message=(
                    "Gesture detector failed while processing frames; subsequent detector logs "
                    f"will be suppressed. frame_id={self._frame_id} error={exc}"
                ),
            )
            return None

        if observation is not None and self.lifecycle_state == LIFECYCLE_DEGRADED and self._capture is not None:
            self.lifecycle_state = LIFECYCLE_RUNNING
        return observation

    def _record_packet_metrics(self, packet: GesturePacket) -> None:
        if packet.tracking_state == "tracked":
            self._metrics.tracked_packets += 1
        elif packet.tracking_state == "temporarily_lost":
            self._metrics.temporary_loss_packets += 1
        else:
            self._metrics.not_detected_packets += 1

    def _maybe_render_preview(self, frame, observation: RawHandObservation | None, packet: GesturePacket) -> None:
        if self._preview is None or frame is None:
            return None
        try:
            self._preview.render(frame, observation=observation, packet=packet)
        except Exception as exc:
            self._record_error(
                error_entry(
                    "gesture.preview.render_failed",
                    "Gesture preview failed while rendering a frame",
                    recoverable=True,
                    hint="The service will continue without preview until the next restart.",
                    details={"error": str(exc), "frame_id": packet.frame_id},
                )
            )
            self._log_recoverable_error_once(
                code="gesture.preview.render_failed",
                message=(
                    "Gesture preview render failed and preview will be disabled. "
                    f"frame_id={packet.frame_id} error={exc}"
                ),
            )
            try:
                self._preview.close()
            except Exception:
                logger.exception("Gesture preview close failed")
            self._preview = None
        return None

    def _record_error(self, error: dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", time.time())
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]

    def _log_recoverable_error_once(self, *, code: str, message: str) -> None:
        if code in self._logged_error_codes:
            return None
        self._logged_error_codes.add(code)
        logger.warning(message)

    def _next_timestamp_ms(self) -> int:
        candidate = int(self._clock() * 1000)
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        self._last_timestamp_ms = candidate
        return candidate


__all__ = ["GestureConfig", "GestureMetrics", "GestureServiceImpl"]