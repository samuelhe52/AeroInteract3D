from __future__ import annotations

import math

from src.gesture.constants import TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES
from src.contracts import Vec3
from src.gesture.runtime import RawHandObservation, normalized_pinch_distance
from src.gesture.temporal import MOTION_PRESET_TUNINGS, TemporalReducer, temporal_tuning_for_motion_preset


def make_observation(
    *,
    wrist_x: float = 0.0,
    pinch_gap: float = 0.04,
    confidence: float = 0.95,
) -> RawHandObservation:
    return RawHandObservation(
        index_tip=Vec3(wrist_x + pinch_gap, 0.20, 0.10),
        thumb_tip=Vec3(wrist_x, 0.20, 0.10),
        wrist=Vec3(wrist_x, 0.0, 0.0),
        confidence=confidence,
        raw_pinch_distance=pinch_gap,
        hand_scale=0.35,
        landmarks=[Vec3(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
    )


def make_rotation_observation(*, theta_rad: float, pinch_gap: float = 0.03) -> RawHandObservation:
    radius = 0.02
    cx = 0.0
    cy = 0.20
    dx = math.cos(theta_rad) * radius
    dy = math.sin(theta_rad) * radius
    return RawHandObservation(
        index_tip=Vec3(cx + dx, cy + dy, 0.10),
        thumb_tip=Vec3(cx - dx, cy - dy, 0.10),
        wrist=Vec3(0.0, 0.0, 0.0),
        confidence=0.95,
        raw_pinch_distance=pinch_gap,
        hand_scale=0.35,
        landmarks=[Vec3(0.5, 0.5, 0.0) for _ in range(21)],
        handedness="Right",
    )


def activate_rotation_mode(reducer: TemporalReducer, *, start_frame_id: int = 1, theta_rad: float = 0.0) -> int:
    frame_id = start_frame_id

    for _ in range(2):
        pinched_packet = None
        for _ in range(24):
            pinched_packet = reducer.reduce(
                make_rotation_observation(theta_rad=theta_rad),
                frame_id=frame_id,
                timestamp_ms=frame_id * 16,
            )
            frame_id += 1
            if pinched_packet.pinch_state == "pinched":
                break

        assert pinched_packet is not None
        assert pinched_packet.pinch_state == "pinched"

        open_packet = None
        for _ in range(24):
            open_packet = reducer.reduce(
                make_observation(pinch_gap=0.18),
                frame_id=frame_id,
                timestamp_ms=frame_id * 16,
            )
            frame_id += 1
            if open_packet.pinch_state == "open":
                break

        assert open_packet is not None
        assert open_packet.pinch_state == "open"

    check = reducer.reduce(
        make_rotation_observation(theta_rad=theta_rad),
        frame_id=frame_id,
        timestamp_ms=frame_id * 16,
    )
    frame_id += 1
    assert check.debug["rotation"]["mode_active"] is True

    # Ensure rotation channel is truly enabled before handing control to tests.
    for _ in range(24):
        check = reducer.reduce(
            make_rotation_observation(theta_rad=theta_rad),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1
        if check.pinch_state == "pinched" and check.debug["rotation"]["enabled"]:
            return frame_id

    raise AssertionError("failed to enter stable enabled rotation state")


def test_temporal_reducer_requires_multiple_frames_to_confirm_pinch_and_release() -> None:
    reducer = TemporalReducer()
    pinch_states: list[str] = []

    for frame_id in range(1, 5):
        packet = reducer.reduce(make_observation(pinch_gap=0.03), frame_id=frame_id, timestamp_ms=frame_id * 16)
        pinch_states.append(packet.pinch_state)

    assert pinch_states == ["pinch_candidate", "pinch_candidate", "pinch_candidate", "pinched"]

    release_states: list[str] = []
    for offset in range(5, 9):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=offset, timestamp_ms=offset * 16)
        release_states.append(packet.pinch_state)

    assert release_states == ["release_candidate", "open", "open", "open"]


def test_temporal_reducer_aggressive_release_guard_keeps_stricter_release_confirmation() -> None:
    reducer = TemporalReducer(aggressive_release_guard=True)

    for frame_id in range(1, 5):
        reducer.reduce(make_observation(pinch_gap=0.03), frame_id=frame_id, timestamp_ms=frame_id * 16)

    release_states: list[str] = []
    for offset in range(5, 9):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=offset, timestamp_ms=offset * 16)
        release_states.append(packet.pinch_state)

    assert release_states == ["release_candidate", "release_candidate", "release_candidate", "open"]


def test_temporal_reducer_predicts_motion_during_temporary_tracking_loss() -> None:
    reducer = TemporalReducer()
    reducer.reduce(make_observation(wrist_x=0.0, pinch_gap=0.12), frame_id=1, timestamp_ms=16)
    tracked_packet = reducer.reduce(make_observation(wrist_x=0.6, pinch_gap=0.12), frame_id=2, timestamp_ms=32)

    lost_packet = reducer.reduce(None, frame_id=3, timestamp_ms=48)

    assert tracked_packet.tracking_state == "tracked"
    assert lost_packet.tracking_state == "temporarily_lost"
    assert lost_packet.wrist.x > tracked_packet.wrist.x
    assert lost_packet.confidence < tracked_packet.confidence

    packet = lost_packet
    for frame_id in range(4, 4 + TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES):
        packet = reducer.reduce(None, frame_id=frame_id, timestamp_ms=frame_id * 16)

    assert packet.tracking_state == "not_detected"
    assert packet.confidence == 0.0


def test_normalized_pinch_distance_scales_with_hand_size() -> None:
    near_distance = normalized_pinch_distance(Vec3(0.52, 0.40, 0.0), Vec3(0.50, 0.40, 0.0), hand_scale=0.40)
    far_distance = normalized_pinch_distance(Vec3(0.70, 0.40, 0.0), Vec3(0.30, 0.40, 0.0), hand_scale=0.40)

    assert near_distance < 0.10
    assert far_distance > 0.90


def test_temporal_reducer_preserves_small_vertical_motion() -> None:
    reducer = TemporalReducer()

    first_packet = reducer.reduce(make_observation(), frame_id=1, timestamp_ms=16)
    second_observation = make_observation()
    second_observation.wrist = Vec3(0.0, 0.012, 0.0)
    second_observation.index_tip = Vec3(0.04, 0.212, 0.10)
    second_observation.thumb_tip = Vec3(0.0, 0.212, 0.10)

    second_packet = reducer.reduce(second_observation, frame_id=2, timestamp_ms=32)

    assert second_packet.wrist.y > first_packet.wrist.y


def test_temporal_reducer_keeps_vertical_motion_responsive_under_blur() -> None:
    reducer = TemporalReducer()

    reducer.reduce(make_observation(), frame_id=1, timestamp_ms=16)
    blurred_observation = make_observation()
    blurred_observation.wrist = Vec3(0.0, 0.06, 0.0)
    blurred_observation.index_tip = Vec3(0.04, 0.26, 0.10)
    blurred_observation.thumb_tip = Vec3(0.0, 0.26, 0.10)

    packet = reducer.reduce(
        blurred_observation,
        frame_id=2,
        timestamp_ms=32,
        runtime_hint={"blur_level": 0.85},
    )

    assert packet.wrist.y > 0.02


def test_temporal_motion_preset_affects_responsiveness() -> None:
    high_reducer = TemporalReducer(tuning=temporal_tuning_for_motion_preset("high"))
    low_reducer = TemporalReducer(tuning=temporal_tuning_for_motion_preset("low"))

    initial_observation = make_observation()
    moved_observation = make_observation(wrist_x=0.5)

    high_reducer.reduce(initial_observation, frame_id=1, timestamp_ms=16)
    low_reducer.reduce(initial_observation, frame_id=1, timestamp_ms=16)
    high_packet = high_reducer.reduce(moved_observation, frame_id=2, timestamp_ms=32)
    low_packet = low_reducer.reduce(moved_observation, frame_id=2, timestamp_ms=32)

    assert MOTION_PRESET_TUNINGS["low"].xy_smoothing_alpha > MOTION_PRESET_TUNINGS["high"].xy_smoothing_alpha
    assert low_packet.wrist.x > high_packet.wrist.x


def test_rotation_angle_updates_continuously_during_pinch_rotation() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    packet_a = reducer.reduce(make_rotation_observation(theta_rad=0.22), frame_id=frame_id, timestamp_ms=frame_id * 16)
    packet_b = reducer.reduce(make_rotation_observation(theta_rad=0.44), frame_id=frame_id + 1, timestamp_ms=(frame_id + 1) * 16)
    packet_c = reducer.reduce(make_rotation_observation(theta_rad=0.66), frame_id=frame_id + 2, timestamp_ms=(frame_id + 2) * 16)
    packet_d = reducer.reduce(make_rotation_observation(theta_rad=0.88), frame_id=frame_id + 3, timestamp_ms=(frame_id + 3) * 16)
    packet_e = reducer.reduce(make_rotation_observation(theta_rad=1.10), frame_id=frame_id + 4, timestamp_ms=(frame_id + 4) * 16)
    packet_f = reducer.reduce(make_rotation_observation(theta_rad=1.32), frame_id=frame_id + 5, timestamp_ms=(frame_id + 5) * 16)
    packet_g = reducer.reduce(make_rotation_observation(theta_rad=1.54), frame_id=frame_id + 6, timestamp_ms=(frame_id + 6) * 16)

    rot_a = packet_a.debug["rotation"]
    rot_b = packet_b.debug["rotation"]
    rot_c = packet_c.debug["rotation"]
    rot_d = packet_d.debug["rotation"]
    rot_g = packet_g.debug["rotation"]
    assert rot_a["enabled"] is True
    assert rot_b["enabled"] is True
    assert rot_d["rotating"] is True
    assert rot_a["slot_count"] > 0
    assert rot_g["slot"] != rot_a["slot"]


def test_rotation_angle_reverses_with_opposite_direction() -> None:
    reducer = TemporalReducer()

    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    reducer.reduce(make_rotation_observation(theta_rad=0.10), frame_id=frame_id, timestamp_ms=frame_id * 16)
    forward = reducer.reduce(make_rotation_observation(theta_rad=0.20), frame_id=frame_id + 1, timestamp_ms=(frame_id + 1) * 16)
    reducer.reduce(make_rotation_observation(theta_rad=0.30), frame_id=frame_id + 2, timestamp_ms=(frame_id + 2) * 16)
    reducer.reduce(make_rotation_observation(theta_rad=-0.20), frame_id=frame_id + 3, timestamp_ms=(frame_id + 3) * 16)
    reducer.reduce(make_rotation_observation(theta_rad=-0.40), frame_id=frame_id + 4, timestamp_ms=(frame_id + 4) * 16)
    reducer.reduce(make_rotation_observation(theta_rad=-0.60), frame_id=frame_id + 5, timestamp_ms=(frame_id + 5) * 16)
    reducer.reduce(make_rotation_observation(theta_rad=-0.80), frame_id=frame_id + 6, timestamp_ms=(frame_id + 6) * 16)
    backward_5 = reducer.reduce(make_rotation_observation(theta_rad=-1.00), frame_id=frame_id + 7, timestamp_ms=(frame_id + 7) * 16)
    backward_6 = reducer.reduce(make_rotation_observation(theta_rad=-1.20), frame_id=frame_id + 8, timestamp_ms=(frame_id + 8) * 16)

    rot_forward = forward.debug["rotation"]
    rot_backward_5 = backward_5.debug["rotation"]
    rot_backward_6 = backward_6.debug["rotation"]
    assert rot_forward["enabled"] is True
    assert rot_backward_5["slot"] != rot_forward["slot"] or rot_backward_6["slot"] != rot_forward["slot"]


def test_rotation_angle_stays_stable_on_one_missing_frame() -> None:
    reducer = TemporalReducer()

    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    tracked = reducer.reduce(make_rotation_observation(theta_rad=0.22), frame_id=frame_id, timestamp_ms=frame_id * 16)
    missing = reducer.reduce(None, frame_id=frame_id + 1, timestamp_ms=(frame_id + 1) * 16)

    tracked_slot = tracked.debug["rotation"]["slot"]
    missing_slot = missing.debug["rotation"]["slot"]
    assert missing_slot == tracked_slot


def test_rotation_slot_wraps_when_rotating_forward_continuously() -> None:
    reducer = TemporalReducer()

    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    first_slot = reducer.reduce(make_rotation_observation(theta_rad=0.24), frame_id=frame_id, timestamp_ms=frame_id * 16).debug["rotation"]["slot"]
    last = None
    theta = 0.24
    for frame_id in range(frame_id + 1, frame_id + 116):
        theta += 0.24
        last = reducer.reduce(make_rotation_observation(theta_rad=theta), frame_id=frame_id, timestamp_ms=frame_id * 16)

    assert last is not None
    final_slot = last.debug["rotation"]["slot"]
    assert final_slot != first_slot
    assert 0 <= final_slot < last.debug["rotation"]["slot_count"]


def test_rotation_gate_stays_off_when_wrist_and_pinch_translate_together() -> None:
    reducer = TemporalReducer()

    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    packet = None
    for frame_id in range(frame_id, frame_id + 8):
        # Keep pinch orientation fixed while translating wrist and pinch midpoint together.
        wrist_x = (frame_id + 1) * 0.025
        obs = make_observation(wrist_x=wrist_x, pinch_gap=0.03)
        packet = reducer.reduce(obs, frame_id=frame_id, timestamp_ms=frame_id * 16)

    assert packet is not None
    rotation = packet.debug["rotation"]
    assert rotation["translation_block"] is True
    assert rotation["rotating"] is False


def test_rotation_mode_requires_two_pinch_open_cycles() -> None:
    reducer = TemporalReducer()

    frame_id = 1
    # First cycle pinch->open: should still stay in MOVE_ONLY mode.
    for _ in range(20):
        packet = reducer.reduce(make_rotation_observation(theta_rad=0.0), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "pinched":
            break
    assert packet.debug["rotation"]["mode_active"] is False

    for _ in range(20):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "open":
            break
    assert packet.debug["rotation"]["mode_progress"] == 1
    assert packet.debug["rotation"]["mode_active"] is False

    # Second cycle completes activation.
    for _ in range(20):
        packet = reducer.reduce(make_rotation_observation(theta_rad=0.0), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "pinched":
            break
    for _ in range(20):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "open":
            break

    assert packet.debug["rotation"]["mode_progress"] == 0
    assert packet.debug["rotation"]["mode_active"] is True


def test_rotation_mode_does_not_accumulate_while_holding_pinch() -> None:
    reducer = TemporalReducer()
    frame_id = 1

    # Long continuous pinch should be rejected as a toggle cycle.
    for _ in range(40):
        packet = reducer.reduce(make_rotation_observation(theta_rad=0.0), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1

    for _ in range(20):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "open":
            break

    rotation = packet.debug["rotation"]
    assert rotation["mode_progress"] == 0
    assert rotation["mode_active"] is False


def test_rotation_mode_progress_times_out_without_second_cycle() -> None:
    reducer = TemporalReducer()
    frame_id = 1

    # Complete first pinch->open cycle.
    for _ in range(20):
        packet = reducer.reduce(make_rotation_observation(theta_rad=0.0), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "pinched":
            break
    for _ in range(20):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1
        if packet.pinch_state == "open":
            break

    assert packet.debug["rotation"]["mode_progress"] == 1

    # Wait without second cycle: progress must reset.
    for _ in range(60):
        packet = reducer.reduce(make_observation(pinch_gap=0.18), frame_id=frame_id, timestamp_ms=frame_id * 16)
        frame_id += 1

    assert packet.debug["rotation"]["mode_progress"] == 0
    assert packet.debug["rotation"]["mode_active"] is False


def test_rotation_stack_resets_after_single_slot_jump() -> None:
    reducer = TemporalReducer()

    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    # Drive until the first slot jump occurs, then stack should be cleared immediately.
    jump_packet = None
    jump_slot = 0
    theta = 0.0
    for frame_id in range(frame_id, frame_id + 18):
        theta += 0.22
        candidate = reducer.reduce(make_rotation_observation(theta_rad=theta), frame_id=frame_id, timestamp_ms=frame_id * 16)
        if candidate.debug["rotation"]["slot"] != 0:
            jump_packet = candidate
            jump_slot = candidate.debug["rotation"]["slot"]
            break

    assert jump_packet is not None
    assert jump_packet.debug["rotation"]["stack_deg"] == 0.0

    # Tiny follow-up motion should not produce delayed extra jumps.
    tiny_packet = reducer.reduce(make_rotation_observation(theta_rad=theta + 0.01), frame_id=frame_id + 20, timestamp_ms=(frame_id + 20) * 16)
    assert tiny_packet.debug["rotation"]["slot"] == jump_slot


def test_rotation_gate_hysteresis_keeps_state_stable_on_short_drop() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1, theta_rad=0.0)

    # Build rotating=True first.
    packet = None
    for step in range(6):
        packet = reducer.reduce(
            make_rotation_observation(theta_rad=0.30 + (0.25 * step)),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )
    assert packet is not None
    assert packet.debug["rotation"]["rotating"] is True

    # A brief weak-motion frame should not immediately flip rotating off.
    weak = reducer.reduce(
        make_rotation_observation(theta_rad=0.30 + (0.25 * 6) + 0.001),
        frame_id=frame_id + 6,
        timestamp_ms=(frame_id + 6) * 16,
    )
    assert weak.debug["rotation"]["rotating"] is True


def test_pinch_remains_stable_during_fast_motion() -> None:
    reducer = TemporalReducer()

    for frame_id in range(1, 5):
        reducer.reduce(make_observation(wrist_x=0.0, pinch_gap=0.03), frame_id=frame_id, timestamp_ms=frame_id * 16)

    moved = reducer.reduce(make_observation(wrist_x=0.35, pinch_gap=0.03), frame_id=5, timestamp_ms=80)
    assert moved.pinch_state in {"pinched", "release_candidate"}


def test_pinch_remains_stable_on_low_quality_front_facing_frame() -> None:
    reducer = TemporalReducer()

    for frame_id in range(1, 5):
        reducer.reduce(make_observation(wrist_x=0.0, pinch_gap=0.03), frame_id=frame_id, timestamp_ms=frame_id * 16)

    low_quality = reducer.reduce(
        make_observation(wrist_x=0.0, pinch_gap=0.03),
        frame_id=5,
        timestamp_ms=80,
        runtime_hint={"blur_level": 0.9, "appearance_match_score": 0.2},
    )
    assert low_quality.pinch_state in {"pinched", "release_candidate"}
