from __future__ import annotations

from dataclasses import dataclass

from src.constants import (
    TEMPORAL_PINCH_CONFIRM_FRAMES as PINCH_CONFIRM_FRAMES,
    TEMPORAL_PINCH_ENTER_THRESHOLD as PINCH_ENTER_THRESHOLD,
    TEMPORAL_PINCH_HOLD_THRESHOLD as PINCH_HOLD_THRESHOLD,
    TEMPORAL_PINCH_RELEASE_THRESHOLD as PINCH_RELEASE_THRESHOLD,
    TEMPORAL_RELEASE_CONFIRM_FRAMES as RELEASE_CONFIRM_FRAMES,
    TEMPORAL_SMOOTHING_ALPHA as SMOOTHING_ALPHA,
    TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES as TRACKING_TEMPORARY_LOSS_FRAMES,
)
from src.contracts import Vec3
from src.gesture.runtime import distance


@dataclass(slots=True)
class GestureTemporalConfig:
    tracking_temporary_loss_frames: int = TRACKING_TEMPORARY_LOSS_FRAMES
    pinch_enter_threshold: float = PINCH_ENTER_THRESHOLD
    pinch_hold_threshold: float = PINCH_HOLD_THRESHOLD
    pinch_release_threshold: float = PINCH_RELEASE_THRESHOLD
    pinch_confirm_frames: int = PINCH_CONFIRM_FRAMES
    release_confirm_frames: int = RELEASE_CONFIRM_FRAMES
    smoothing_alpha: float = SMOOTHING_ALPHA
    smooth_coordinates: bool = True


@dataclass(slots=True)
class GestureFrameAnalysis:
    timestamp_ms: int
    tracking_state: str
    pinch_state: str
    confidence: float
    index_tip: Vec3
    thumb_tip: Vec3
    wrist: Vec3
    velocity: Vec3
    pinch_distance: float
    tracking_loss_streak: int
    smoothing_hint: dict[str, float | str]


class GestureTemporalReducer:
    def __init__(self, config: GestureTemporalConfig | None = None) -> None:
        self._config = config or GestureTemporalConfig()
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_ms = 0
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
        self._processed_frames = 0

    def process(
        self,
        *,
        timestamp_ms: int,
        index_tip: Vec3 | None = None,
        thumb_tip: Vec3 | None = None,
        wrist: Vec3 | None = None,
        raw_confidence: float = 0.0,
    ) -> GestureFrameAnalysis:
        timestamp_ms = self._normalize_timestamp(timestamp_ms)

        if index_tip is None or thumb_tip is None or wrist is None:
            self._tracking_loss_streak += 1
            tracking_state = self._compute_tracking_state(hand_detected=False)
            pinch_state = self._compute_pinch_state(None, None)
            confidence = self._compute_confidence(tracking_state, pinch_state, raw_confidence)
            velocity = Vec3(0.0, 0.0, 0.0)

            self._tracking_state = tracking_state
            self._pinch_state = pinch_state
            self._confidence = confidence
            self._last_velocity = velocity

            return GestureFrameAnalysis(
                timestamp_ms=timestamp_ms,
                tracking_state=tracking_state,
                pinch_state=pinch_state,
                confidence=confidence,
                index_tip=self._last_index_tip,
                thumb_tip=self._last_thumb_tip,
                wrist=self._last_wrist,
                velocity=velocity,
                pinch_distance=self._last_pinch_distance,
                tracking_loss_streak=self._tracking_loss_streak,
                smoothing_hint=self._smoothing_hint(),
            )

        self._tracking_loss_streak = 0

        normalized_index_tip = self._normalize_vec3(index_tip)
        normalized_thumb_tip = self._normalize_vec3(thumb_tip)
        normalized_wrist = self._normalize_vec3(wrist)

        tracking_state = self._compute_tracking_state(hand_detected=True)
        pinch_state = self._compute_pinch_state(normalized_index_tip, normalized_thumb_tip)
        confidence = self._compute_confidence(tracking_state, pinch_state, raw_confidence)

        smoothed_index_tip = self._smooth_vec3(normalized_index_tip, self._last_index_tip)
        smoothed_thumb_tip = self._smooth_vec3(normalized_thumb_tip, self._last_thumb_tip)
        smoothed_wrist = self._smooth_vec3(normalized_wrist, self._last_wrist)
        velocity = self._compute_velocity(smoothed_wrist, self._last_wrist)

        self._last_index_tip = smoothed_index_tip
        self._last_thumb_tip = smoothed_thumb_tip
        self._last_wrist = smoothed_wrist
        self._last_velocity = velocity
        self._tracking_state = tracking_state
        self._pinch_state = pinch_state
        self._confidence = confidence
        self._processed_frames += 1

        return GestureFrameAnalysis(
            timestamp_ms=timestamp_ms,
            tracking_state=tracking_state,
            pinch_state=pinch_state,
            confidence=confidence,
            index_tip=smoothed_index_tip,
            thumb_tip=smoothed_thumb_tip,
            wrist=smoothed_wrist,
            velocity=velocity,
            pinch_distance=self._last_pinch_distance,
            tracking_loss_streak=self._tracking_loss_streak,
            smoothing_hint=self._smoothing_hint(),
        )

    def _normalize_timestamp(self, timestamp_ms: int) -> int:
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _compute_tracking_state(self, *, hand_detected: bool) -> str:
        if hand_detected:
            return "tracked"
        if self._tracking_loss_streak <= self._config.tracking_temporary_loss_frames:
            return "temporarily_lost"
        return "not_detected"

    def _compute_pinch_state(self, index_tip: Vec3 | None, thumb_tip: Vec3 | None) -> str:
        if index_tip is None or thumb_tip is None:
            self._pinch_candidate_streak = 0
            if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"}:
                self._release_candidate_streak += 1
                if self._release_candidate_streak <= self._config.release_confirm_frames:
                    return "release_candidate"
            self._release_candidate_streak = 0
            self._last_pinch_distance = 0.0
            return "open"

        pinch_distance = distance(index_tip, thumb_tip)
        self._last_pinch_distance = pinch_distance

        if pinch_distance <= self._config.pinch_hold_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_candidate_streak >= self._config.pinch_confirm_frames:
                return "pinched"
            return "pinch_candidate"

        if pinch_distance <= self._config.pinch_enter_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_state == "pinched":
                return "pinched"
            return "pinch_candidate"

        self._pinch_candidate_streak = 0
        if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"} and pinch_distance >= self._config.pinch_release_threshold:
            self._release_candidate_streak += 1
            if self._release_candidate_streak <= self._config.release_confirm_frames:
                return "release_candidate"
            self._release_candidate_streak = 0
            return "open"

        self._release_candidate_streak = 0
        return "open"

    def _compute_confidence(self, tracking_state: str, pinch_state: str, raw_confidence: float) -> float:
        if tracking_state == "temporarily_lost":
            return max(0.05, 0.3 - 0.05 * max(self._tracking_loss_streak - 1, 0))
        if tracking_state == "not_detected":
            return 0.0

        if pinch_state == "pinched":
            raw_confidence += 0.05
        elif pinch_state in {"pinch_candidate", "release_candidate"}:
            raw_confidence -= 0.05
        return max(0.0, min(1.0, raw_confidence))

    def _normalize_vec3(self, value: Vec3) -> Vec3:
        return Vec3(
            x=max(-1.0, min(1.0, float(value.x))),
            y=max(-1.0, min(1.0, float(value.y))),
            z=max(-1.0, min(1.0, float(value.z))),
        )

    def _smooth_vec3(self, current: Vec3, previous: Vec3) -> Vec3:
        if not self._config.smooth_coordinates or self._processed_frames == 0:
            return current
        alpha = self._config.smoothing_alpha
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

    def _smoothing_hint(self) -> dict[str, float | str]:
        if self._config.smooth_coordinates:
            return {"method": "linear", "alpha": self._config.smoothing_alpha}
        return {"method": "none", "alpha": 1.0}
