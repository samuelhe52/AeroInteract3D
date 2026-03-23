from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from src.contracts import GesturePacket, PinchState, TrackingState, Vec3
from src.gesture.constants import (
    ENTER_THRESHOLD,
    GESTURE_DEFAULT_HAND_ID,
    GRACE_FRAMES,
    HIGH_BLUR_LEVEL,
    HIGH_MOTION_ALPHA_Y_FLOOR,
    LATERAL_BLUR_X_GATE_MULTIPLIER,
    LATERAL_BLUR_Y_DELTA_BLEND,
    LOW_QUALITY_CONFIDENCE,
    LOW_QUALITY_MOTION_ALPHA_Y_FLOOR,
    LOW_QUALITY_MOTION_ALPHA_Y_MULTIPLIER,
    MOTION_ALPHA_X_FLOOR,
    MOTION_ALPHA_X_MULTIPLIER,
    MOTION_ALPHA_Y_MAX,
    MOTION_ALPHA_Y_OFFSET,
    OPEN_MARGIN_MIN,
    PINCH_CONFIRM_FRAMES,
    PINCH_MIN_HOLD_FRAMES,
    PINCH_RELEASE_BLOCK_WRIST_SPEED,
    PINCH_RELEASE_MIN_QUALITY_WHEN_STABLE,
    REACQUIRE_BLEND_FRAMES,
    REACQUIRE_GATE_DISTANCE,
    REACQUIRE_MAX_DX,
    REACQUIRE_MAX_DY,
    RELAXED_OPEN_MARGIN_MIN,
    RELAXED_QUALITY_CONFIDENCE,
    RELAXED_RELEASE_CONFIRM_FRAMES,
    ROT_EQ_ACTIVE_MIN_DEG,
    ROT_EQ_DEADZONE_X,
    ROT_EQ_DEADZONE_Y,
    ROT_EQ_DEADZONE_Z,
    ROT_EQ_GAIN_X,
    ROT_EQ_GAIN_Y,
    ROT_EQ_GAIN_Z,
    ROT_EQ_STEP_THRESHOLD_X,
    ROT_EQ_STEP_THRESHOLD_Y,
    ROT_EQ_STEP_THRESHOLD_Z,
    ROT_DELTA_CLAMP_DEG,
    ROT_DELTA_NOISE_DEG,
    ROT_GATE_FRAMES,
    ROT_GATE_RELEASE_FRAMES,
    ROT_HAND_GRAB_TIP_SPREAD_MAX,
    ROT_MODE_GRAB_MIN_FRAMES,
    ROT_MODE_OPEN_MIN_FRAMES,
    ROT_MODE_TOGGLE_COOLDOWN_MS,
    ROT_MODE_TIMEOUT_FRAMES,
    ROT_OPPOSITE_JITTER_SUPPRESS_DEG,
    ROT_SLOT_COUNT,
    ROT_SLOT_STEP_DEG,
    ROT_STEP_ACCUM_MIN_DEG,
    ROT_STACK_CLEAR_IDLE_FRAMES,
    ROT_TREND_MIN_DEG,
    RELEASE_CONFIRM_FRAMES,
    RELEASE_THRESHOLD,
    TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES,
)
from src.gesture.runtime import RawHandObservation, distance_2d
from src.utils.contracts import EXPECTED_CONTRACT_VERSION


ZERO_VEC3 = Vec3(0.0, 0.0, 0.0)
MotionPreset = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class TemporalTuning:
    smoothing_alpha: float
    xy_smoothing_alpha: float
    position_deadzone: float
    prediction_blend: float
    prediction_lead: float
    lost_tracking_motion_damping: float


MOTION_PRESET_TUNINGS: dict[MotionPreset, TemporalTuning] = {
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


def temporal_tuning_for_motion_preset(preset: MotionPreset) -> TemporalTuning:
    return MOTION_PRESET_TUNINGS[preset]


@dataclass(slots=True)
class TemporalReducer:
    hand_id: str = GESTURE_DEFAULT_HAND_ID
    tuning: TemporalTuning = field(default_factory=lambda: MOTION_PRESET_TUNINGS["medium"])
    aggressive_release_guard: bool = False
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
    _rotation_slot_x: int = field(init=False, default=0)
    _rotation_slot_y: int = field(init=False, default=0)
    _rotation_slot_z: int = field(init=False, default=0)
    _rotation_step_buffer_x_deg: float = field(init=False, default=0.0)
    _rotation_step_buffer_y_deg: float = field(init=False, default=0.0)
    _rotation_step_buffer_z_deg: float = field(init=False, default=0.0)
    _rotation_last_midpoint: Vec3 | None = field(init=False, default=None)
    _rotation_source: str = field(init=False, default="none")
    _rotation_gate_count: int = field(init=False, default=0)
    _rotation_last_delta_x_deg: float = field(init=False, default=0.0)
    _rotation_last_delta_y_deg: float = field(init=False, default=0.0)
    _rotation_last_delta_z_deg: float = field(init=False, default=0.0)
    _rotation_mode_progress: int = field(init=False, default=0)
    _rotation_mode_stage: str = field(init=False, default="wait_grab")
    _rotation_mode_stage_frames: int = field(init=False, default=0)
    _rotation_grab_frames: int = field(init=False, default=0)
    _rotation_open_frames: int = field(init=False, default=0)
    _rotation_mode_active: bool = field(init=False, default=False)
    _rotation_rotating_smoothed: bool = field(init=False, default=False)
    _rotation_stack_idle_frames: int = field(init=False, default=0)
    _rotation_last_is_grabbed: bool = field(init=False, default=False)
    _rotation_last_is_open_hand: bool = field(init=False, default=False)
    _rotation_last_tip_spread: float = field(init=False, default=0.0)
    _rotation_last_tip_min_dist: float = field(init=False, default=0.0)
    _rotation_last_tip_max_dist: float = field(init=False, default=0.0)
    _rotation_last_fist_curl: float = field(init=False, default=0.0)
    _rotation_mode_cooldown_until_ms: int = field(init=False, default=0)

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
        self._rotation_slot_x = 0
        self._rotation_slot_y = 0
        self._rotation_slot_z = 0
        self._rotation_step_buffer_x_deg = 0.0
        self._rotation_step_buffer_y_deg = 0.0
        self._rotation_step_buffer_z_deg = 0.0
        self._rotation_last_midpoint = None
        self._rotation_source = "none"
        self._rotation_gate_count = 0
        self._rotation_last_delta_x_deg = 0.0
        self._rotation_last_delta_y_deg = 0.0
        self._rotation_last_delta_z_deg = 0.0
        self._rotation_mode_progress = 0
        self._rotation_mode_stage = "wait_grab"
        self._rotation_mode_stage_frames = 0
        self._rotation_grab_frames = 0
        self._rotation_open_frames = 0
        self._rotation_mode_active = False
        self._rotation_rotating_smoothed = False
        self._rotation_stack_idle_frames = 0
        self._rotation_last_is_grabbed = False
        self._rotation_last_is_open_hand = False
        self._rotation_last_tip_spread = 0.0
        self._rotation_last_tip_min_dist = 0.0
        self._rotation_last_tip_max_dist = 0.0
        self._rotation_last_fist_curl = 0.0
        self._rotation_mode_cooldown_until_ms = 0

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
        wrist_speed = self._vec_magnitude(velocity)

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
            source=source,
            wrist_speed=wrist_speed,
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

        rotation_debug = self._update_rotation_channel(
            tracking_state=tracking_state,
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            wrist=wrist,
            landmarks=observation.landmarks,
            timestamp_ms=timestamp_ms,
        )

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
                "rotation": rotation_debug,
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
            source="predicted",
            wrist_speed=self._vec_magnitude(velocity),
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

        rotation_debug = self._rotation_debug_payload(
            enabled=False,
            rotating=False,
        )

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
                "rotation": rotation_debug,
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
        source: str,
        wrist_speed: float,
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
            source=source,
            wrist_speed=wrist_speed,
        )
        if release_allowed:
            self._release_confirm_count += 1
            if self._release_confirm_count >= self._release_confirm_frames():
                self._pinch_confirm_count = 0
                self._last_pinch_state = "open"
                self._pinched_hold_frames = 0
                return self._last_pinch_state
            self._last_pinch_state = "release_candidate"
            return self._last_pinch_state

        self._release_confirm_count = 0
        self._last_pinch_state = "pinched"
        return self._last_pinch_state

    def _allow_release(
        self,
        *,
        pinch_score: float,
        quality_score: float,
        geometry_open_margin: float,
        source: str,
        wrist_speed: float,
    ) -> bool:
        # Keep pinch sticky during fast transport or low-quality front-facing frames.
        if self._is_pinch_stable():
            if wrist_speed >= PINCH_RELEASE_BLOCK_WRIST_SPEED:
                return False
            if quality_score < PINCH_RELEASE_MIN_QUALITY_WHEN_STABLE:
                return False

        if self.aggressive_release_guard:
            return (
                pinch_score < RELEASE_THRESHOLD
                and quality_score >= LOW_QUALITY_CONFIDENCE
                and geometry_open_margin >= OPEN_MARGIN_MIN
                and source not in {"fallback", "predicted"}
            )

        return (
            pinch_score < 0.50
            and quality_score >= RELAXED_QUALITY_CONFIDENCE
            and geometry_open_margin >= RELAXED_OPEN_MARGIN_MIN
            and source != "predicted"
        )

    def _release_confirm_frames(self) -> int:
        if self.aggressive_release_guard:
            return RELEASE_CONFIRM_FRAMES
        return RELAXED_RELEASE_CONFIRM_FRAMES

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

        alpha_x = max(self.tuning.xy_smoothing_alpha * MOTION_ALPHA_X_MULTIPLIER, MOTION_ALPHA_X_FLOOR)
        base_alpha_y = min(max(self.tuning.xy_smoothing_alpha + MOTION_ALPHA_Y_OFFSET, alpha_x), MOTION_ALPHA_Y_MAX)
        deadzone_x = self.tuning.position_deadzone * 1.8

        motion_y = abs(raw_dy)
        if low_quality or blur_level > HIGH_BLUR_LEVEL:
            alpha_y = max(base_alpha_y * LOW_QUALITY_MOTION_ALPHA_Y_MULTIPLIER, LOW_QUALITY_MOTION_ALPHA_Y_FLOOR)
        elif motion_y > 0.04:
            alpha_y = max(base_alpha_y, HIGH_MOTION_ALPHA_Y_FLOOR)
        else:
            alpha_y = base_alpha_y

        x = self._smooth_component(previous.x, current.x, alpha=alpha_x, deadzone=deadzone_x)

        # Keep Y independent when X jump is likely noise from lateral sweep blur.
        if abs(raw_dx) > (deadzone_x * LATERAL_BLUR_X_GATE_MULTIPLIER):
            y_current = previous.y + (raw_dy * LATERAL_BLUR_Y_DELTA_BLEND)
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
        for name, preset_tuning in MOTION_PRESET_TUNINGS.items():
            if preset_tuning == self.tuning:
                return name
        return "custom"

    def _gaussian(self, value: float, *, mean: float, sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        delta = (value - mean) / sigma
        return 2.718281828 ** (-0.5 * delta * delta)

    def _update_rotation_channel(
        self,
        *,
        tracking_state: TrackingState,
        pinch_state: PinchState,
        index_tip: Vec3,
        thumb_tip: Vec3,
        wrist: Vec3,
        landmarks: list[Vec3],
        timestamp_ms: int,
    ) -> dict[str, Any]:
        is_grabbed = self._is_hand_grabbed(landmarks=landmarks, wrist=wrist)
        is_open_hand = self._is_hand_open(landmarks=landmarks, wrist=wrist)
        self._rotation_last_is_grabbed = is_grabbed
        self._rotation_last_is_open_hand = is_open_hand
        self._update_rotation_mode_gate(
            tracking_state=tracking_state,
            is_grabbed=is_grabbed,
            is_open_hand=is_open_hand,
            timestamp_ms=timestamp_ms,
        )

        enabled = tracking_state == "tracked" and self._rotation_mode_active
        if not enabled:
            self._rotation_last_midpoint = None
            self._rotation_source = "none"
            self._rotation_gate_count = 0
            self._rotation_rotating_smoothed = False
            self._rotation_last_delta_x_deg = 0.0
            self._rotation_last_delta_y_deg = 0.0
            self._rotation_last_delta_z_deg = 0.0
            self._rotation_step_buffer_x_deg = 0.0
            self._rotation_step_buffer_y_deg = 0.0
            self._rotation_step_buffer_z_deg = 0.0
            self._rotation_stack_idle_frames = 0
            return self._rotation_debug_payload(
                enabled=False,
                rotating=False,
            )

        pinch_mid = Vec3(
            (index_tip.x + thumb_tip.x) * 0.5,
            (index_tip.y + thumb_tip.y) * 0.5,
            (index_tip.z + thumb_tip.z) * 0.5,
        )

        if self._rotation_last_midpoint is None:
            mid_dx = 0.0
            mid_dy = 0.0
            mid_dz = 0.0
        else:
            mid_dx = pinch_mid.x - self._rotation_last_midpoint.x
            mid_dy = pinch_mid.y - self._rotation_last_midpoint.y
            mid_dz = pinch_mid.z - self._rotation_last_midpoint.z

        delta_x_deg = self._scaled_axis_delta(
            raw_delta=mid_dx,
            gain=ROT_EQ_GAIN_X,
            deadzone=ROT_EQ_DEADZONE_X,
            last_delta=self._rotation_last_delta_x_deg,
        )
        delta_y_deg = self._scaled_axis_delta(
            raw_delta=mid_dy,
            gain=ROT_EQ_GAIN_Y,
            deadzone=ROT_EQ_DEADZONE_Y,
            last_delta=self._rotation_last_delta_y_deg,
        )
        delta_z_deg = self._scaled_axis_delta(
            raw_delta=mid_dz,
            gain=ROT_EQ_GAIN_Z,
            deadzone=ROT_EQ_DEADZONE_Z,
            last_delta=self._rotation_last_delta_z_deg,
        )

        active_mag = max(abs(delta_x_deg), abs(delta_y_deg), abs(delta_z_deg))
        gate_hit = active_mag >= ROT_EQ_ACTIVE_MIN_DEG
        if gate_hit:
            self._rotation_gate_count = min(self._rotation_gate_count + 1, ROT_GATE_FRAMES)
        else:
            self._rotation_gate_count = max(self._rotation_gate_count - 1, 0)

        if (not self._rotation_rotating_smoothed) and self._rotation_gate_count >= ROT_GATE_FRAMES:
            self._rotation_rotating_smoothed = True
        elif self._rotation_rotating_smoothed and self._rotation_gate_count <= ROT_GATE_RELEASE_FRAMES:
            self._rotation_rotating_smoothed = False
        rotating = self._rotation_rotating_smoothed

        has_delta = self._rotation_last_midpoint is not None
        if has_delta:
            self._accumulate_axis_stack(delta_x_deg, axis="x")
            self._accumulate_axis_stack(delta_y_deg, axis="y")
            self._accumulate_axis_stack(delta_z_deg, axis="z")

            if rotating:
                self._emit_axis_slot_if_needed(axis="x", threshold=ROT_EQ_STEP_THRESHOLD_X)
                self._emit_axis_slot_if_needed(axis="y", threshold=ROT_EQ_STEP_THRESHOLD_Y)
                self._emit_axis_slot_if_needed(axis="z", threshold=ROT_EQ_STEP_THRESHOLD_Z)
            elif self._rotation_stack_idle_frames >= ROT_STACK_CLEAR_IDLE_FRAMES:
                self._rotation_step_buffer_x_deg = 0.0
                self._rotation_step_buffer_y_deg = 0.0
                self._rotation_step_buffer_z_deg = 0.0
                self._rotation_stack_idle_frames = 0

            self._rotation_last_delta_x_deg = delta_x_deg if abs(delta_x_deg) > 0.0 else (self._rotation_last_delta_x_deg * 0.65)
            self._rotation_last_delta_y_deg = delta_y_deg if abs(delta_y_deg) > 0.0 else (self._rotation_last_delta_y_deg * 0.65)
            self._rotation_last_delta_z_deg = delta_z_deg if abs(delta_z_deg) > 0.0 else (self._rotation_last_delta_z_deg * 0.65)
        else:
            self._rotation_stack_idle_frames += 1
            if (not rotating) and self._rotation_stack_idle_frames >= ROT_STACK_CLEAR_IDLE_FRAMES:
                self._rotation_step_buffer_x_deg = 0.0
                self._rotation_step_buffer_y_deg = 0.0
                self._rotation_step_buffer_z_deg = 0.0
                self._rotation_stack_idle_frames = 0

        self._rotation_last_midpoint = pinch_mid
        self._rotation_source = "equivalent_xyz"
        return self._rotation_debug_payload(
            enabled=True,
            rotating=rotating,
        )

    def _update_rotation_mode_gate(
        self,
        *,
        tracking_state: TrackingState,
        is_grabbed: bool,
        is_open_hand: bool,
        timestamp_ms: int,
    ) -> None:
        # Highest-priority active gate: one valid full-hand grab->open toggles mode.
        if tracking_state == "not_detected":
            self._reset_rotation_mode_sequence(clear_mode=True)
            return

        if tracking_state == "temporarily_lost":
            # Keep mode alive across brief pinch/track dropouts.
            return

        if timestamp_ms < self._rotation_mode_cooldown_until_ms:
            # Lock state changes for a fixed cooldown window after each successful toggle.
            self._reset_rotation_mode_sequence(clear_mode=False)
            return

        self._rotation_mode_stage_frames += 1
        if self._rotation_mode_stage_frames > ROT_MODE_TIMEOUT_FRAMES:
            self._reset_rotation_mode_sequence(clear_mode=False)

        if self._rotation_mode_stage == "wait_grab":
            if is_grabbed and (not is_open_hand):
                self._rotation_grab_frames += 1
                if self._rotation_grab_frames >= ROT_MODE_GRAB_MIN_FRAMES:
                    self._rotation_mode_stage = "wait_open"
                    self._rotation_mode_stage_frames = 0
                    self._rotation_open_frames = 0
            else:
                self._rotation_grab_frames = 0
            self._rotation_mode_progress = 0
            return

        if is_open_hand:
            self._rotation_open_frames += 1
            self._rotation_mode_progress = 1
            if self._rotation_open_frames >= ROT_MODE_OPEN_MIN_FRAMES:
                self._rotation_mode_active = not self._rotation_mode_active
                self._rotation_mode_cooldown_until_ms = timestamp_ms + ROT_MODE_TOGGLE_COOLDOWN_MS
                self._reset_rotation_mode_sequence(clear_mode=False)
        else:
            self._rotation_open_frames = 0

    def _reset_rotation_mode_sequence(self, *, clear_mode: bool) -> None:
        self._rotation_mode_progress = 0
        self._rotation_mode_stage = "wait_grab"
        self._rotation_mode_stage_frames = 0
        self._rotation_grab_frames = 0
        self._rotation_open_frames = 0
        if clear_mode:
            self._rotation_mode_active = False

    def _is_hand_grabbed(self, *, landmarks: list[Vec3], wrist: Vec3) -> bool:
        tip_indices = (4, 8, 12, 16, 20)
        if len(landmarks) <= 20:
            return False
        scale = max(self._last_hand_scale, 1e-6)
        spread = self._tip_spread(landmarks=landmarks, tip_indices=tip_indices) / scale
        min_dist, max_dist = self._tip_wrist_dist_range(landmarks=landmarks, wrist=wrist, tip_indices=tip_indices)
        min_dist /= scale
        max_dist /= scale

        self._rotation_last_tip_spread = spread
        self._rotation_last_tip_min_dist = min_dist
        self._rotation_last_tip_max_dist = max_dist
        self._rotation_last_fist_curl = 0.0

        # Simplified grab detector: strictly based on tip spread as requested.
        return spread <= ROT_HAND_GRAB_TIP_SPREAD_MAX

    def _is_hand_open(self, *, landmarks: list[Vec3], wrist: Vec3) -> bool:
        tip_indices = (4, 8, 12, 16, 20)
        if len(landmarks) <= max(tip_indices):
            return False
        scale = max(self._last_hand_scale, 1e-6)
        spread = self._tip_spread(landmarks=landmarks, tip_indices=tip_indices) / scale
        min_dist, max_dist = self._tip_wrist_dist_range(landmarks=landmarks, wrist=wrist, tip_indices=tip_indices)
        min_dist /= scale
        max_dist /= scale
        self._rotation_last_tip_spread = spread
        self._rotation_last_tip_min_dist = min_dist
        self._rotation_last_tip_max_dist = max_dist

        # Spread-only release detector. Keep it exclusive from grab.
        return spread > ROT_HAND_GRAB_TIP_SPREAD_MAX

    def _tip_wrist_dist_range(
        self,
        *,
        landmarks: list[Vec3],
        wrist: Vec3,
        tip_indices: tuple[int, ...],
    ) -> tuple[float, float]:
        min_dist = 1.0
        max_dist = 0.0
        for idx in tip_indices:
            tip = landmarks[idx]
            dist = distance_2d(tip, wrist)
            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)
        return min_dist, max_dist

    def _tip_spread(self, *, landmarks: list[Vec3], tip_indices: tuple[int, ...]) -> float:
        min_x = 1.0
        max_x = -1.0
        min_y = 1.0
        max_y = -1.0
        for idx in tip_indices:
            tip = landmarks[idx]
            min_x = min(min_x, tip.x)
            max_x = max(max_x, tip.x)
            min_y = min(min_y, tip.y)
            max_y = max(max_y, tip.y)
        return math.sqrt(((max_x - min_x) ** 2) + ((max_y - min_y) ** 2))

    def _scaled_axis_delta(self, *, raw_delta: float, gain: float, deadzone: float, last_delta: float) -> float:
        delta_deg = self._clamp(raw_delta * gain, low=-ROT_DELTA_CLAMP_DEG, high=ROT_DELTA_CLAMP_DEG)
        if abs(raw_delta) <= deadzone or abs(delta_deg) <= ROT_DELTA_NOISE_DEG:
            return 0.0
        if (
            abs(last_delta) >= ROT_TREND_MIN_DEG
            and (delta_deg * last_delta) < 0.0
            and abs(delta_deg) <= ROT_OPPOSITE_JITTER_SUPPRESS_DEG
        ):
            return 0.0
        return delta_deg

    def _accumulate_axis_stack(self, delta_deg: float, *, axis: str) -> None:
        if abs(delta_deg) < ROT_STEP_ACCUM_MIN_DEG:
            self._rotation_stack_idle_frames += 1
            return

        self._rotation_stack_idle_frames = 0
        if axis == "x":
            if self._rotation_step_buffer_x_deg * delta_deg < 0.0:
                self._rotation_step_buffer_x_deg = 0.0
            self._rotation_step_buffer_x_deg += delta_deg
            return
        if axis == "y":
            if self._rotation_step_buffer_y_deg * delta_deg < 0.0:
                self._rotation_step_buffer_y_deg = 0.0
            self._rotation_step_buffer_y_deg += delta_deg
            return
        if self._rotation_step_buffer_z_deg * delta_deg < 0.0:
            self._rotation_step_buffer_z_deg = 0.0
        self._rotation_step_buffer_z_deg += delta_deg

    def _emit_axis_slot_if_needed(self, *, axis: str, threshold: float) -> None:
        if axis == "x":
            if self._rotation_step_buffer_x_deg >= threshold:
                self._rotation_slot_x = (self._rotation_slot_x + 1) % ROT_SLOT_COUNT
                self._rotation_step_buffer_x_deg = 0.0
            elif self._rotation_step_buffer_x_deg <= -threshold:
                self._rotation_slot_x = (self._rotation_slot_x - 1) % ROT_SLOT_COUNT
                self._rotation_step_buffer_x_deg = 0.0
            return
        if axis == "y":
            if self._rotation_step_buffer_y_deg >= threshold:
                self._rotation_slot_y = (self._rotation_slot_y + 1) % ROT_SLOT_COUNT
                self._rotation_step_buffer_y_deg = 0.0
            elif self._rotation_step_buffer_y_deg <= -threshold:
                self._rotation_slot_y = (self._rotation_slot_y - 1) % ROT_SLOT_COUNT
                self._rotation_step_buffer_y_deg = 0.0
            return
        if self._rotation_step_buffer_z_deg >= threshold:
            self._rotation_slot_z = (self._rotation_slot_z + 1) % ROT_SLOT_COUNT
            self._rotation_step_buffer_z_deg = 0.0
        elif self._rotation_step_buffer_z_deg <= -threshold:
            self._rotation_slot_z = (self._rotation_slot_z - 1) % ROT_SLOT_COUNT
            self._rotation_step_buffer_z_deg = 0.0

    def _rotation_debug_payload(
        self,
        *,
        enabled: bool,
        rotating: bool,
    ) -> dict[str, Any]:
        deg_x = self._rotation_slot_x * ROT_SLOT_STEP_DEG
        deg_y = self._rotation_slot_y * ROT_SLOT_STEP_DEG
        deg_z = self._rotation_slot_z * ROT_SLOT_STEP_DEG
        return {
            "enabled": enabled,
            "rotating": rotating,
            # Keep legacy single-slot key for compatibility (mapped to Z axis).
            "slot": self._rotation_slot_z,
            "slot_count": ROT_SLOT_COUNT,
            "slot_x": self._rotation_slot_x,
            "slot_y": self._rotation_slot_y,
            "slot_z": self._rotation_slot_z,
            "deg_x": deg_x,
            "deg_y": deg_y,
            "deg_z": deg_z,
            "gate_count": self._rotation_gate_count,
            "source": self._rotation_source,
            "mode_progress": self._rotation_mode_progress,
            "mode_target": 1,
            "mode_stage": self._rotation_mode_stage,
            "mode_active": self._rotation_mode_active,
            "mode_cooldown_ms": max(self._rotation_mode_cooldown_until_ms - (self._last_timestamp_ms or 0), 0),
            "grab_detected": self._rotation_last_is_grabbed,
            "open_detected": self._rotation_last_is_open_hand,
            "tip_spread": self._rotation_last_tip_spread,
            "tip_min_dist": self._rotation_last_tip_min_dist,
            "tip_max_dist": self._rotation_last_tip_max_dist,
            "fist_curl": self._rotation_last_fist_curl,
            "stack_deg": self._rotation_step_buffer_z_deg,
            "stack_x_deg": self._rotation_step_buffer_x_deg,
            "stack_y_deg": self._rotation_step_buffer_y_deg,
            "stack_z_deg": self._rotation_step_buffer_z_deg,
            "mode_name": "ROTATE_ENABLED" if self._rotation_mode_active else "MOVE_ONLY",
        }

    def _lerp(self, start: float, end: float, alpha: float) -> float:
        t = self._clamp01(alpha)
        return start + ((end - start) * t)

    def _vec_magnitude(self, value: Vec3) -> float:
        return math.sqrt((value.x * value.x) + (value.y * value.y) + (value.z * value.z))

    def _clamp(self, value: float, *, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _clamp01(self, value: float) -> float:
        return self._clamp(value, low=0.0, high=1.0)


__all__ = ["MOTION_PRESET_TUNINGS", "MotionPreset", "TemporalReducer", "TemporalTuning", "temporal_tuning_for_motion_preset"]
