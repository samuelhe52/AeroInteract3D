from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.constants import GESTURE_DEFAULT_HAND_ID, TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES
from src.contracts import GesturePacket, PinchState, TrackingState, Vec3
from src.gesture.runtime import RawHandObservation, distance_2d
from src.utils.contracts import EXPECTED_CONTRACT_VERSION


ZERO_VEC3 = Vec3(0.0, 0.0, 0.0)
SmoothingPreset = Literal["high", "medium", "low"]

# Tunable defaults for blur/fallback stability while keeping low latency.
ENTER_THRESHOLD = 0.62
RELEASE_THRESHOLD = 0.42
PINCH_CONFIRM_FRAMES = 2
RELEASE_CONFIRM_FRAMES = 4
PINCH_MIN_HOLD_FRAMES = 4
GRACE_FRAMES = 6
FALLBACK_MAX_FRAMES = 8
REACQUIRE_BLEND_FRAMES = 4
REACQUIRE_GATE_DISTANCE = 0.16
REACQUIRE_MAX_DX = 0.06
REACQUIRE_MAX_DY = 0.05
OPEN_MARGIN_MIN = 0.10
LOW_QUALITY_CONFIDENCE = 0.60
HIGH_BLUR_LEVEL = 0.62


@dataclass(frozen=True, slots=True)
class TemporalTuning:
    smoothing_alpha: float
    xy_smoothing_alpha: float
    position_deadzone: float
    prediction_blend: float
    prediction_lead: float
    lost_tracking_motion_damping: float


PRESET_TUNINGS: dict[SmoothingPreset, TemporalTuning] = {
    "high": TemporalTuning(
        smoothing_alpha=0.72,
        xy_smoothing_alpha=0.64,
        position_deadzone=0.005,
        prediction_blend=0.28,
        prediction_lead=0.36,
        lost_tracking_motion_damping=0.60,
    ),
    "medium": TemporalTuning(
        smoothing_alpha=0.82,
        xy_smoothing_alpha=0.78,
        position_deadzone=0.004,
        prediction_blend=0.22,
        prediction_lead=0.30,
        lost_tracking_motion_damping=0.56,
    ),
    "low": TemporalTuning(
        smoothing_alpha=0.94,
        xy_smoothing_alpha=0.90,
        position_deadzone=0.0,
        prediction_blend=0.12,
        prediction_lead=0.16,
        lost_tracking_motion_damping=0.34,
    ),
}


def temporal_tuning_for_preset(preset: SmoothingPreset) -> TemporalTuning:
    return PRESET_TUNINGS[preset]


@dataclass(slots=True)
class TemporalReducer:
    hand_id: str = GESTURE_DEFAULT_HAND_ID
    tuning: TemporalTuning = field(default_factory=lambda: PRESET_TUNINGS["medium"])
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
    _pinched_hold_frames: int = field(init=False, default=0)
    _grace_frames_used: int = field(init=False, default=0)
    _reacquire_blend_remaining: int = field(init=False, default=0)
    _last_source: str = field(init=False, default="none")

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_index_tip = ZERO_VEC3
        self._last_thumb_tip = ZERO_VEC3
        self._last_wrist = ZERO_VEC3
        self._last_velocity = ZERO_VEC3
        self._last_timestamp_ms = None
        self._last_pinch_state = "open"
        self._pinch_confirm_count = 0
        self._release_confirm_count = 0
        self._missing_frames = 0
        self._last_hand_scale = 1.0
        self._pinched_hold_frames = 0
        self._grace_frames_used = 0
        self._reacquire_blend_remaining = 0
        self._last_source = "none"

    def reduce(
        self,
        observation: RawHandObservation | None,
        *,
        frame_id: int,
        timestamp_ms: int,
        runtime_hint: dict[str, Any] | None = None,
    ) -> GesturePacket:
        if observation is None:
            return self._reduce_missing(frame_id=frame_id, timestamp_ms=timestamp_ms)
        return self._reduce_observation(
            observation,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            runtime_hint=runtime_hint,
        )

    def _reduce_observation(
        self,
        observation: RawHandObservation,
        *,
        frame_id: int,
        timestamp_ms: int,
        runtime_hint: dict[str, Any] | None,
    ) -> GesturePacket:
        source = str(
            runtime_hint.get("observation_source") if runtime_hint and "observation_source" in runtime_hint else observation.observation_source
        )
        appearance_match_score = self._clamp01(
            float(runtime_hint.get("appearance_match_score") if runtime_hint and "appearance_match_score" in runtime_hint else observation.appearance_match_score)
        )
        blur_level = self._clamp01(
            float(runtime_hint.get("blur_level") if runtime_hint and "blur_level" in runtime_hint else observation.blur_level)
        )
        predicted_tracked = bool(
            runtime_hint.get("predicted_tracked") if runtime_hint and "predicted_tracked" in runtime_hint else observation.predicted_tracked
        )

        was_missing = self._missing_frames > 0
        if source == "detected" and (was_missing or self._last_source in {"fallback", "predicted"}):
            self._reacquire_blend_remaining = REACQUIRE_BLEND_FRAMES

        previous_wrist = self._last_wrist
        index_tip = observation.index_tip
        thumb_tip = observation.thumb_tip
        wrist = observation.wrist

        if self._reacquire_blend_remaining > 0 and source == "detected":
            index_tip, thumb_tip, wrist, source = self._blend_reacquire(
                index_tip=index_tip,
                thumb_tip=thumb_tip,
                wrist=wrist,
                source=source,
            )

        quality_score = self._observation_quality(
            confidence=observation.confidence,
            blur_level=blur_level,
            appearance_match_score=appearance_match_score,
            source=source,
            predicted_tracked=predicted_tracked,
            quality_hint=observation.quality_hint,
        )

        low_quality = quality_score < LOW_QUALITY_CONFIDENCE
        self._missing_frames = 0
        self._grace_frames_used = 0

        index_tip = self._smooth_vec(
            previous=self._last_index_tip,
            current=index_tip,
            low_quality=low_quality,
            blur_level=blur_level,
        )
        thumb_tip = self._smooth_vec(
            previous=self._last_thumb_tip,
            current=thumb_tip,
            low_quality=low_quality,
            blur_level=blur_level,
        )
        wrist = self._smooth_vec(previous=self._last_wrist, current=wrist, low_quality=low_quality, blur_level=blur_level)

        velocity = self._compute_velocity(previous_wrist, wrist, timestamp_ms=timestamp_ms)

        pinch_distance = self._normalized_camera_pinch_distance(index_tip, thumb_tip, hand_scale=observation.hand_scale)
        pinch_score, geometry_open_margin = self._pinch_score(
            raw_pinch_distance=observation.raw_pinch_distance,
            appearance_match_score=appearance_match_score,
            quality_score=quality_score,
        )

        pinch_state = self._update_pinch_state(
            pinch_score=pinch_score,
            quality_score=quality_score,
            geometry_open_margin=geometry_open_margin,
        )

        self._last_index_tip = index_tip
        self._last_thumb_tip = thumb_tip
        self._last_wrist = wrist
        self._last_velocity = velocity
        self._last_timestamp_ms = timestamp_ms
        self._last_hand_scale = max(observation.hand_scale, 1e-6)
        self._last_source = source

        tracking_state = "tracked"
        if source in {"fallback", "predicted"} and not self._is_pinch_stable():
            tracking_state = "temporarily_lost"

        confidence = self._tracked_confidence(
            observation_confidence=observation.confidence,
            pinch_score=pinch_score,
            quality_score=quality_score,
            source=source,
        )

        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self.hand_id,
            tracking_state=tracking_state,
            confidence=confidence,
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            coordinate_space="camera_norm",
            pinch_distance=pinch_distance,
            velocity=velocity,
            smoothing_hint={
                "method": "blur_aware_reacquire",
                "window": 1,
                "preset": self._preset_name(),
                "alpha_xy": self.tuning.xy_smoothing_alpha,
                "alpha_z": self.tuning.smoothing_alpha,
                "observation_source": source,
                "quality_score": quality_score,
            },
            debug={
                "raw_pinch_distance": observation.raw_pinch_distance,
                "pinch_score": pinch_score,
                "appearance_match_score": appearance_match_score,
                "feature_assisted_score": appearance_match_score,
                "predicted_tracked": predicted_tracked,
                "grace_frames_used": self._grace_frames_used,
                "blur_level": blur_level,
                "observation_source": source,
                "missing_frames": self._missing_frames,
                "pinch_confirm_count": self._pinch_confirm_count,
                "release_confirm_count": self._release_confirm_count,
                "reacquire_blend_progress": self._reacquire_blend_progress(),
                "hold_frames": self._pinched_hold_frames,
                "quality_score": quality_score,
                "open_margin": geometry_open_margin,
                "detector_source": observation.detector_source,
                "handedness": observation.handedness,
            },
        )

    def _reduce_missing(self, *, frame_id: int, timestamp_ms: int) -> GesturePacket:
        self._missing_frames += 1
        self._grace_frames_used = min(self._missing_frames, GRACE_FRAMES)
        predicted_index_tip, predicted_thumb_tip, predicted_wrist = self._predict_positions()
        velocity = self._dampened_velocity()

        pinch_distance = self._normalized_camera_pinch_distance(
            predicted_index_tip,
            predicted_thumb_tip,
            hand_scale=self._last_hand_scale,
        )
        pinch_score, geometry_open_margin = self._pinch_score(
            raw_pinch_distance=pinch_distance,
            appearance_match_score=0.0,
            quality_score=0.25,
        )
        pinch_state = self._update_pinch_state(
            pinch_score=pinch_score,
            quality_score=0.25,
            geometry_open_margin=geometry_open_margin,
        )

        if self._missing_frames > TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES:
            tracking_state: TrackingState = "not_detected"
        elif self._is_pinch_stable() and self._missing_frames <= GRACE_FRAMES:
            tracking_state = "tracked"
        else:
            tracking_state = "temporarily_lost"

        if tracking_state == "not_detected":
            self._pinch_confirm_count = 0
            self._release_confirm_count = 0
            self._pinched_hold_frames = 0
            self._last_pinch_state = "open"

        self._last_index_tip = predicted_index_tip
        self._last_thumb_tip = predicted_thumb_tip
        self._last_wrist = predicted_wrist
        self._last_velocity = velocity
        self._last_timestamp_ms = timestamp_ms
        self._last_source = "predicted"

        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self.hand_id,
            tracking_state=tracking_state,
            confidence=self._missing_confidence(),
            pinch_state=pinch_state,
            index_tip=predicted_index_tip,
            thumb_tip=predicted_thumb_tip,
            wrist=predicted_wrist,
            coordinate_space="camera_norm",
            pinch_distance=pinch_distance,
            velocity=velocity,
            smoothing_hint={
                "method": "loss_prediction",
                "window": self._missing_frames,
                "preset": self._preset_name(),
                "blend": self.tuning.prediction_blend,
                "observation_source": "predicted",
            },
            debug={
                "pinch_score": pinch_score,
                "appearance_match_score": 0.0,
                "feature_assisted_score": 0.0,
                "predicted_tracked": True,
                "grace_frames_used": self._grace_frames_used,
                "blur_level": 1.0,
                "observation_source": "predicted",
                "missing_frames": self._missing_frames,
                "pinch_confirm_count": self._pinch_confirm_count,
                "release_confirm_count": self._release_confirm_count,
                "reacquire_blend_progress": self._reacquire_blend_progress(),
                "open_margin": geometry_open_margin,
            },
        )

    def _blend_reacquire(
        self,
        *,
        index_tip: Vec3,
        thumb_tip: Vec3,
        wrist: Vec3,
        source: str,
    ) -> tuple[Vec3, Vec3, Vec3, str]:
        predicted_index, predicted_thumb, predicted_wrist = self._predict_positions()
        delta = distance_2d(predicted_wrist, wrist)
        progress = REACQUIRE_BLEND_FRAMES - self._reacquire_blend_remaining + 1
        blend_weight = progress / max(REACQUIRE_BLEND_FRAMES, 1)

        cautious = delta > REACQUIRE_GATE_DISTANCE
        if cautious:
            blend_weight = min(blend_weight * 0.45, 0.35)

        blended_index = self._blend_vec(predicted_index, index_tip, blend_weight)
        blended_thumb = self._blend_vec(predicted_thumb, thumb_tip, blend_weight)
        blended_wrist = self._blend_vec(predicted_wrist, wrist, blend_weight)

        blended_index = self._limit_reacquire_delta(self._last_index_tip, blended_index)
        blended_thumb = self._limit_reacquire_delta(self._last_thumb_tip, blended_thumb)
        blended_wrist = self._limit_reacquire_delta(self._last_wrist, blended_wrist)

        self._reacquire_blend_remaining = max(self._reacquire_blend_remaining - 1, 0)
        return blended_index, blended_thumb, blended_wrist, "reacquire_blend" if cautious else source

    def _update_pinch_state(
        self,
        *,
        pinch_score: float,
        quality_score: float,
        geometry_open_margin: float,
    ) -> PinchState:
        prior_pinch = self._is_pinch_stable()

        if prior_pinch:
            self._pinched_hold_frames += 1
        else:
            self._pinched_hold_frames = 0

        if not prior_pinch:
            if pinch_score > ENTER_THRESHOLD:
                self._pinch_confirm_count += 1
                self._release_confirm_count = 0
                if self._pinch_confirm_count >= PINCH_CONFIRM_FRAMES:
                    self._last_pinch_state = "pinched"
                    self._pinched_hold_frames = 0
                    return self._last_pinch_state
                self._last_pinch_state = "pinch_candidate"
                return self._last_pinch_state

            self._pinch_confirm_count = 0
            self._release_confirm_count = 0
            self._last_pinch_state = "open"
            return self._last_pinch_state

        self._pinch_confirm_count = PINCH_CONFIRM_FRAMES
        if self._pinched_hold_frames < PINCH_MIN_HOLD_FRAMES:
            self._release_confirm_count = 0
            self._last_pinch_state = "pinched"
            return self._last_pinch_state

        release_allowed = self._allow_release(
            pinch_score=pinch_score,
            quality_score=quality_score,
            geometry_open_margin=geometry_open_margin,
        )
        if release_allowed:
            self._release_confirm_count += 1
            if self._release_confirm_count >= RELEASE_CONFIRM_FRAMES:
                self._pinch_confirm_count = 0
                self._last_pinch_state = "open"
                self._pinched_hold_frames = 0
                return self._last_pinch_state
            self._last_pinch_state = "release_candidate"
            return self._last_pinch_state

        self._release_confirm_count = 0
        self._last_pinch_state = "pinched"
        return self._last_pinch_state

    def _allow_release(self, *, pinch_score: float, quality_score: float, geometry_open_margin: float) -> bool:
        return (
            pinch_score < RELEASE_THRESHOLD
            and quality_score >= LOW_QUALITY_CONFIDENCE
            and geometry_open_margin >= OPEN_MARGIN_MIN
            and self._last_source not in {"fallback", "predicted"}
        )

    def _pinch_score(
        self,
        *,
        raw_pinch_distance: float,
        appearance_match_score: float,
        quality_score: float,
    ) -> tuple[float, float]:
        pinched_likelihood = self._gaussian(raw_pinch_distance, mean=0.06, sigma=0.04)
        open_likelihood = self._gaussian(raw_pinch_distance, mean=0.18, sigma=0.08)
        likelihood_sum = max(pinched_likelihood + open_likelihood, 1e-6)
        geometry_score = pinched_likelihood / likelihood_sum
        geometry_open_margin = (open_likelihood / likelihood_sum) - geometry_score

        prior_score = 0.78 if self._is_pinch_stable() else 0.24
        if quality_score < LOW_QUALITY_CONFIDENCE:
            weight_geometry = 0.45
            weight_prior = 0.35
            weight_appearance = 0.20
        else:
            weight_geometry = 0.70
            weight_prior = 0.25
            weight_appearance = 0.05

        pinch_score = (
            weight_geometry * geometry_score
            + weight_prior * prior_score
            + weight_appearance * self._clamp01(appearance_match_score)
        )
        return self._clamp01(pinch_score), geometry_open_margin

    def _smooth_vec(self, *, previous: Vec3, current: Vec3, low_quality: bool, blur_level: float) -> Vec3:
        raw_dx = current.x - previous.x
        raw_dy = current.y - previous.y

        alpha_x = max(self.tuning.xy_smoothing_alpha * 0.70, 0.05)
        deadzone_x = self.tuning.position_deadzone * 1.8

        motion_y = abs(raw_dy)
        if low_quality or blur_level > HIGH_BLUR_LEVEL:
            alpha_y = 0.20
        elif motion_y > 0.04:
            alpha_y = 0.55
        else:
            alpha_y = self.tuning.xy_smoothing_alpha

        x = self._smooth_component(previous.x, current.x, alpha=alpha_x, deadzone=deadzone_x)

        # Keep Y independent when X jump is likely noise from lateral sweep blur.
        if abs(raw_dx) > (deadzone_x * 4.0):
            y_current = previous.y + (raw_dy * 0.4)
        else:
            y_current = current.y
        y = self._smooth_component(previous.y, y_current, alpha=alpha_y, deadzone=self.tuning.position_deadzone)

        z = self._smooth_component(
            previous.z,
            current.z,
            alpha=self.tuning.smoothing_alpha,
            deadzone=self.tuning.position_deadzone,
        )
        return Vec3(x=x, y=y, z=z)

    def _smooth_component(self, previous: float, current: float, *, alpha: float, deadzone: float) -> float:
        delta = current - previous
        if abs(delta) <= deadzone:
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
        factor = self.tuning.prediction_blend * (self.tuning.lost_tracking_motion_damping ** self._missing_frames)
        lead = self.tuning.prediction_lead * max(self._missing_frames, 1)
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
        factor = self.tuning.lost_tracking_motion_damping ** self._missing_frames
        return Vec3(
            x=self._last_velocity.x * factor,
            y=self._last_velocity.y * factor,
            z=self._last_velocity.z * factor,
        )

    def _blend_vec(self, left: Vec3, right: Vec3, weight: float) -> Vec3:
        clamped_weight = self._clamp01(weight)
        return Vec3(
            x=(left.x * (1.0 - clamped_weight)) + (right.x * clamped_weight),
            y=(left.y * (1.0 - clamped_weight)) + (right.y * clamped_weight),
            z=(left.z * (1.0 - clamped_weight)) + (right.z * clamped_weight),
        )

    def _limit_reacquire_delta(self, previous: Vec3, current: Vec3) -> Vec3:
        dx = current.x - previous.x
        dy = current.y - previous.y
        return Vec3(
            x=self._clamp(previous.x + max(-REACQUIRE_MAX_DX, min(REACQUIRE_MAX_DX, dx))),
            y=self._clamp(previous.y + max(-REACQUIRE_MAX_DY, min(REACQUIRE_MAX_DY, dy))),
            z=current.z,
        )

    def _reacquire_blend_progress(self) -> float:
        if REACQUIRE_BLEND_FRAMES <= 0:
            return 1.0
        done = REACQUIRE_BLEND_FRAMES - self._reacquire_blend_remaining
        return self._clamp01(done / REACQUIRE_BLEND_FRAMES)

    def _observation_quality(
        self,
        *,
        confidence: float,
        blur_level: float,
        appearance_match_score: float,
        source: str,
        predicted_tracked: bool,
        quality_hint: float,
    ) -> float:
        source_penalty = 0.0
        if source == "fallback":
            source_penalty = 0.16
        elif source == "predicted":
            source_penalty = 0.24
        if predicted_tracked:
            source_penalty += 0.08

        score = (
            0.58 * self._clamp01(confidence)
            + 0.18 * (1.0 - self._clamp01(blur_level))
            + 0.14 * self._clamp01(appearance_match_score)
            + 0.10 * self._clamp01(quality_hint)
            - source_penalty
        )
        return self._clamp01(score)

    def _tracked_confidence(self, *, observation_confidence: float, pinch_score: float, quality_score: float, source: str) -> float:
        base = (0.65 * self._clamp01(observation_confidence)) + (0.35 * self._clamp01(quality_score))
        if source in {"fallback", "predicted"} and self._is_pinch_stable():
            return self._clamp(max(base, 0.62), low=0.0, high=1.0)
        return self._clamp(base + (0.05 * pinch_score), low=0.0, high=1.0)

    def _normalized_camera_pinch_distance(self, index_tip: Vec3, thumb_tip: Vec3, *, hand_scale: float) -> float:
        return distance_2d(index_tip, thumb_tip) / max(2.0 * hand_scale, 1e-6)

    def _missing_confidence(self) -> float:
        if self._missing_frames > TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES:
            return 0.0
        if self._is_pinch_stable() and self._missing_frames <= GRACE_FRAMES:
            return self._clamp(0.66 - (self._missing_frames - 1) * 0.01, low=0.62, high=0.75)
        remaining = TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES - self._missing_frames + 1
        return self._clamp(remaining / (TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES + 1), low=0.0, high=1.0)

    def _is_pinch_stable(self) -> bool:
        return self._last_pinch_state in {"pinched", "release_candidate"}

    def _preset_name(self) -> str:
        for name, preset_tuning in PRESET_TUNINGS.items():
            if preset_tuning == self.tuning:
                return name
        return "custom"

    def _gaussian(self, value: float, *, mean: float, sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        delta = (value - mean) / sigma
        return 2.718281828 ** (-0.5 * delta * delta)

    def _clamp(self, value: float, *, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _clamp01(self, value: float) -> float:
        return self._clamp(value, low=0.0, high=1.0)


__all__ = ["PRESET_TUNINGS", "SmoothingPreset", "TemporalReducer", "TemporalTuning", "temporal_tuning_for_preset"]
