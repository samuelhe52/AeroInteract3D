from __future__ import annotations

from dataclasses import dataclass, field

from src.constants import (
    GESTURE_DEFAULT_HAND_ID,
    TEMPORAL_LOST_TRACKING_MOTION_DAMPING,
    TEMPORAL_PINCH_CONFIRM_FRAMES,
    TEMPORAL_PINCH_ENTER_THRESHOLD,
    TEMPORAL_PINCH_HOLD_THRESHOLD,
    TEMPORAL_PINCH_RELEASE_THRESHOLD,
    TEMPORAL_POSITION_DEADZONE,
    TEMPORAL_PREDICTION_BLEND,
    TEMPORAL_PREDICTION_LEAD,
    TEMPORAL_RELEASE_CONFIRM_FRAMES,
    TEMPORAL_SMOOTHING_ALPHA,
    TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES,
    TEMPORAL_XY_SMOOTHING_ALPHA,
)
from src.contracts import GesturePacket, PinchState, TrackingState, Vec3
from src.gesture.runtime import RawHandObservation, distance_2d
from src.utils.contracts import EXPECTED_CONTRACT_VERSION


ZERO_VEC3 = Vec3(0.0, 0.0, 0.0)


@dataclass(slots=True)
class TemporalReducer:
    hand_id: str = GESTURE_DEFAULT_HAND_ID
    _last_index_tip: Vec3 = field(init=False, default_factory=lambda: ZERO_VEC3)
    _last_thumb_tip: Vec3 = field(init=False, default_factory=lambda: ZERO_VEC3)
    _last_wrist: Vec3 = field(init=False, default_factory=lambda: ZERO_VEC3)
    _last_velocity: Vec3 = field(init=False, default_factory=lambda: ZERO_VEC3)
    _last_timestamp_ms: int | None = field(init=False, default=None)
    _last_pinch_state: PinchState = field(init=False, default="open")
    _pinch_confirm_count: int = field(init=False, default=0)
    _release_confirm_count: int = field(init=False, default=0)
    _missing_frames: int = field(init=False, default=0)
    _last_hand_scale: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_index_tip = ZERO_VEC3
        self._last_thumb_tip = ZERO_VEC3
        self._last_wrist = ZERO_VEC3
        self._last_velocity = ZERO_VEC3
        self._last_timestamp_ms: int | None = None
        self._last_pinch_state: PinchState = "open"
        self._pinch_confirm_count = 0
        self._release_confirm_count = 0
        self._missing_frames = 0
        self._last_hand_scale = 1.0

    def reduce(
        self,
        observation: RawHandObservation | None,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> GesturePacket:
        if observation is None:
            return self._reduce_missing(frame_id=frame_id, timestamp_ms=timestamp_ms)
        return self._reduce_observation(observation, frame_id=frame_id, timestamp_ms=timestamp_ms)

    def _reduce_observation(
        self,
        observation: RawHandObservation,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> GesturePacket:
        self._missing_frames = 0
        previous_wrist = self._last_wrist
        index_tip = self._smooth(self._last_index_tip, observation.index_tip, xy_alpha=TEMPORAL_XY_SMOOTHING_ALPHA)
        thumb_tip = self._smooth(self._last_thumb_tip, observation.thumb_tip, xy_alpha=TEMPORAL_XY_SMOOTHING_ALPHA)
        wrist = self._smooth(self._last_wrist, observation.wrist, xy_alpha=TEMPORAL_XY_SMOOTHING_ALPHA)
        velocity = self._compute_velocity(previous_wrist, wrist, timestamp_ms=timestamp_ms)
        pinch_distance = self._normalized_camera_pinch_distance(index_tip, thumb_tip, hand_scale=observation.hand_scale)
        pinch_state = self._update_pinch_state(observation.raw_pinch_distance)

        self._last_index_tip = index_tip
        self._last_thumb_tip = thumb_tip
        self._last_wrist = wrist
        self._last_velocity = velocity
        self._last_timestamp_ms = timestamp_ms
        self._last_hand_scale = max(observation.hand_scale, 1e-6)

        confidence = self._tracked_confidence(observation.confidence, pinch_distance)
        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self.hand_id,
            tracking_state="tracked",
            confidence=confidence,
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            coordinate_space="camera_norm",
            pinch_distance=pinch_distance,
            velocity=velocity,
            smoothing_hint={
                "method": "ema_loss_prediction",
                "window": 1,
                "alpha_xy": TEMPORAL_XY_SMOOTHING_ALPHA,
                "alpha_z": TEMPORAL_SMOOTHING_ALPHA,
            },
            debug={
                "raw_pinch_distance": observation.raw_pinch_distance,
                "hand_scale": observation.hand_scale,
                "missing_frames": self._missing_frames,
                "pinch_confirm_count": self._pinch_confirm_count,
                "release_confirm_count": self._release_confirm_count,
                "detector_source": observation.detector_source,
                "handedness": observation.handedness,
            },
        )

    def _reduce_missing(self, *, frame_id: int, timestamp_ms: int) -> GesturePacket:
        self._missing_frames += 1
        tracking_state: TrackingState = (
            "temporarily_lost"
            if self._missing_frames <= TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES
            else "not_detected"
        )

        predicted_index_tip, predicted_thumb_tip, predicted_wrist = self._predict_positions()
        velocity = self._dampened_velocity()

        if tracking_state == "not_detected":
            self._pinch_confirm_count = 0
            self._release_confirm_count = 0
            self._last_pinch_state = "open"

        self._last_index_tip = predicted_index_tip
        self._last_thumb_tip = predicted_thumb_tip
        self._last_wrist = predicted_wrist
        self._last_velocity = velocity
        self._last_timestamp_ms = timestamp_ms

        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self.hand_id,
            tracking_state=tracking_state,
            confidence=self._missing_confidence(),
            pinch_state=self._last_pinch_state,
            index_tip=predicted_index_tip,
            thumb_tip=predicted_thumb_tip,
            wrist=predicted_wrist,
            coordinate_space="camera_norm",
            pinch_distance=self._normalized_camera_pinch_distance(
                predicted_index_tip,
                predicted_thumb_tip,
                hand_scale=self._last_hand_scale,
            ),
            velocity=velocity,
            smoothing_hint={
                "method": "loss_prediction",
                "window": self._missing_frames,
                "blend": TEMPORAL_PREDICTION_BLEND,
            },
            debug={
                "missing_frames": self._missing_frames,
                "pinch_confirm_count": self._pinch_confirm_count,
                "release_confirm_count": self._release_confirm_count,
            },
        )

    def _update_pinch_state(self, pinch_distance: float) -> PinchState:
        if self._last_pinch_state == "open":
            if pinch_distance <= TEMPORAL_PINCH_ENTER_THRESHOLD:
                self._pinch_confirm_count += 1
                self._release_confirm_count = 0
                if self._pinch_confirm_count >= TEMPORAL_PINCH_CONFIRM_FRAMES:
                    self._last_pinch_state = "pinched"
                    return self._last_pinch_state
                self._last_pinch_state = "pinch_candidate"
                return self._last_pinch_state

            self._pinch_confirm_count = 0
            self._release_confirm_count = 0
            self._last_pinch_state = "open"
            return self._last_pinch_state

        if self._last_pinch_state == "pinch_candidate":
            if pinch_distance <= TEMPORAL_PINCH_ENTER_THRESHOLD:
                self._pinch_confirm_count += 1
                if self._pinch_confirm_count >= TEMPORAL_PINCH_CONFIRM_FRAMES:
                    self._last_pinch_state = "pinched"
                    return self._last_pinch_state
                return self._last_pinch_state

            self._pinch_confirm_count = 0
            self._last_pinch_state = "open"
            return self._last_pinch_state

        if self._last_pinch_state == "pinched" and pinch_distance <= TEMPORAL_PINCH_HOLD_THRESHOLD:
            self._pinch_confirm_count = TEMPORAL_PINCH_CONFIRM_FRAMES
            self._release_confirm_count = 0
            self._last_pinch_state = "pinched"
            return self._last_pinch_state

        if self._last_pinch_state == "pinched" and pinch_distance >= TEMPORAL_PINCH_RELEASE_THRESHOLD:
            self._release_confirm_count += 1
            if self._release_confirm_count >= TEMPORAL_RELEASE_CONFIRM_FRAMES:
                self._pinch_confirm_count = 0
                self._last_pinch_state = "open"
                return self._last_pinch_state
            self._last_pinch_state = "release_candidate"
            return self._last_pinch_state

        if self._last_pinch_state == "release_candidate":
            if pinch_distance >= TEMPORAL_PINCH_RELEASE_THRESHOLD:
                self._release_confirm_count += 1
                if self._release_confirm_count >= TEMPORAL_RELEASE_CONFIRM_FRAMES:
                    self._pinch_confirm_count = 0
                    self._last_pinch_state = "open"
                    return self._last_pinch_state
                return self._last_pinch_state

            self._release_confirm_count = 0
            self._last_pinch_state = "pinched"
            return self._last_pinch_state

        self._pinch_confirm_count = TEMPORAL_PINCH_CONFIRM_FRAMES
        self._release_confirm_count = 0
        self._last_pinch_state = "pinched"
        return self._last_pinch_state

    def _smooth(self, previous: Vec3, current: Vec3, *, xy_alpha: float) -> Vec3:
        return Vec3(
            x=self._smooth_component(previous.x, current.x, alpha=xy_alpha),
            y=self._smooth_component(previous.y, current.y, alpha=xy_alpha),
            z=self._smooth_component(previous.z, current.z, alpha=TEMPORAL_SMOOTHING_ALPHA),
        )

    def _smooth_component(self, previous: float, current: float, *, alpha: float) -> float:
        delta = current - previous
        if abs(delta) <= TEMPORAL_POSITION_DEADZONE:
            return previous
        return previous + (alpha * delta)

    def _compute_velocity(self, previous: Vec3, current: Vec3, *, timestamp_ms: int) -> Vec3:
        if self._last_timestamp_ms is None:
            return ZERO_VEC3

        delta_ms = max(timestamp_ms - self._last_timestamp_ms, 1)
        delta_seconds = delta_ms / 1000.0
        return Vec3(
            x=(current.x - previous.x) / delta_seconds,
            y=(current.y - previous.y) / delta_seconds,
            z=(current.z - previous.z) / delta_seconds,
        )

    def _predict_positions(self) -> tuple[Vec3, Vec3, Vec3]:
        factor = TEMPORAL_PREDICTION_BLEND * (TEMPORAL_LOST_TRACKING_MOTION_DAMPING ** self._missing_frames)
        lead = TEMPORAL_PREDICTION_LEAD * self._missing_frames
        return (
            self._predict_vec(self._last_index_tip, factor=factor, lead=lead),
            self._predict_vec(self._last_thumb_tip, factor=factor, lead=lead),
            self._predict_vec(self._last_wrist, factor=factor, lead=lead),
        )

    def _predict_vec(self, base: Vec3, *, factor: float, lead: float) -> Vec3:
        return Vec3(
            x=self._clamp(base.x + (self._last_velocity.x * lead * factor)),
            y=self._clamp(base.y + (self._last_velocity.y * lead * factor)),
            z=self._clamp(base.z + (self._last_velocity.z * lead * factor)),
        )

    def _dampened_velocity(self) -> Vec3:
        factor = TEMPORAL_LOST_TRACKING_MOTION_DAMPING ** self._missing_frames
        return Vec3(
            x=self._last_velocity.x * factor,
            y=self._last_velocity.y * factor,
            z=self._last_velocity.z * factor,
        )

    def _tracked_confidence(self, observation_confidence: float, pinch_distance: float) -> float:
        pinch_signal = 1.0 - self._clamp(pinch_distance / TEMPORAL_PINCH_RELEASE_THRESHOLD, low=0.0, high=1.0)
        return self._clamp((0.75 * observation_confidence) + (0.25 * pinch_signal), low=0.0, high=1.0)

    def _normalized_camera_pinch_distance(self, index_tip: Vec3, thumb_tip: Vec3, *, hand_scale: float) -> float:
        return distance_2d(index_tip, thumb_tip) / max(2.0 * hand_scale, 1e-6)

    def _missing_confidence(self) -> float:
        if self._missing_frames > TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES:
            return 0.0
        remaining = TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES - self._missing_frames + 1
        return self._clamp(remaining / (TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES + 1), low=0.0, high=1.0)

    def _clamp(self, value: float, *, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))


__all__ = ["TemporalReducer"]