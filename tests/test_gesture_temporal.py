from __future__ import annotations

from src.gesture.constants import ROT_SLOT_COUNT, ROT_SLOT_STEP_DEG, TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES
from src.contracts import Vec3
from src.gesture.runtime import RawHandObservation, normalized_pinch_distance
from src.gesture.temporal import MOTION_PRESET_TUNINGS, TemporalReducer, temporal_tuning_for_motion_preset


def make_observation(
    *,
    wrist_x: float = 0.0,
    wrist_y: float = 0.0,
    wrist_z: float = 0.0,
    pinch_gap: float = 0.04,
    confidence: float = 0.95,
    hand_pose: str = "open",
) -> RawHandObservation:
    wrist = Vec3(wrist_x, wrist_y, wrist_z)
    return RawHandObservation(
        index_tip=Vec3(wrist_x + pinch_gap, 0.20 + wrist_y, 0.10 + wrist_z),
        thumb_tip=Vec3(wrist_x, 0.20 + wrist_y, 0.10 + wrist_z),
        wrist=wrist,
        confidence=confidence,
        raw_pinch_distance=pinch_gap,
        hand_scale=0.35,
        landmarks=make_hand_landmarks(wrist=wrist, pose=hand_pose),
        handedness="Right",
    )


def make_hand_landmarks(*, wrist: Vec3, pose: str) -> list[Vec3]:
    points = [Vec3(wrist.x, wrist.y, wrist.z) for _ in range(21)]
    tip_indices = [4, 8, 12, 16, 20]
    # Keep synthetic spread separated so spread-only grab/open logic remains testable.
    spread = 0.06 if pose == "grab" else 0.32
    offset_step = 0.005 if pose == "grab" else 0.080
    for i, tip_idx in enumerate(tip_indices):
        offset = (i - 2) * offset_step
        points[tip_idx] = Vec3(wrist.x + offset, wrist.y + spread, wrist.z)
    return points


def make_eq_rotation_observation(
    *,
    mid_x: float,
    mid_y: float,
    mid_z: float,
    pinch_gap: float = 0.03,
    hand_pose: str = "open",
) -> RawHandObservation:
    wrist = Vec3(0.0, 0.0, 0.0)
    return RawHandObservation(
        index_tip=Vec3(mid_x + (pinch_gap * 0.5), mid_y, mid_z),
        thumb_tip=Vec3(mid_x - (pinch_gap * 0.5), mid_y, mid_z),
        wrist=wrist,
        confidence=0.95,
        raw_pinch_distance=pinch_gap,
        hand_scale=0.35,
        landmarks=make_hand_landmarks(wrist=wrist, pose=hand_pose),
        handedness="Right",
    )


def make_curled_grab_landmarks(*, wrist: Vec3) -> list[Vec3]:
    points = [Vec3(wrist.x, wrist.y, wrist.z) for _ in range(21)]
    tip_indices = [4, 8, 12, 16, 20]
    offsets = [-0.05, -0.025, 0.0, 0.025, 0.05]
    for offset, tip_idx in zip(offsets, tip_indices):
        points[tip_idx] = Vec3(wrist.x + offset, wrist.y + 0.045, wrist.z)
    return points


def activate_rotation_mode(reducer: TemporalReducer, *, start_frame_id: int = 1) -> int:
    frame_id = start_frame_id

    # Grab phase.
    for _ in range(8):
        reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    # Open phase to complete one grab->open sequence.
    for _ in range(8):
        check = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    assert check.debug["rotation"]["mode_active"] is True

    # Ensure rotation channel is truly enabled before handing control to tests.
    for _ in range(24):
        check = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
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


def test_equivalent_rotation_x_axis_updates_continuously() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    mid_x = 0.0
    previous_deg_x = None
    step_deltas: list[float] = []
    for step in range(1, 28):
        mid_x += 0.008
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=mid_x, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )
        current_deg_x = float(packet.debug["rotation"]["deg_x"])
        if previous_deg_x is not None:
            step_deltas.append(current_deg_x - previous_deg_x)
        previous_deg_x = current_deg_x

    assert packet.debug["rotation"]["enabled"] is True
    assert packet.debug["rotation"]["deg_x"] > 0.0
    assert any(0.0 < delta < ROT_SLOT_STEP_DEG for delta in step_deltas)


def test_equivalent_rotation_y_axis_updates_slot_y() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    start_packet = reducer.reduce(
        make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
        frame_id=frame_id,
        timestamp_ms=frame_id * 16,
    )
    start_slot = start_packet.debug["rotation"]["slot_y"]

    mid_y = 0.20
    packet = start_packet
    for step in range(1, 28):
        mid_y += 0.008
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=mid_y, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )

    assert packet.debug["rotation"]["slot_y"] != start_slot
    assert packet.debug["rotation"]["deg_y"] > 0.0


def test_equivalent_rotation_z_axis_updates_slot_z() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    start_packet = reducer.reduce(
        make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
        frame_id=frame_id,
        timestamp_ms=frame_id * 16,
    )
    start_slot = start_packet.debug["rotation"]["slot_z"]

    mid_z = 0.10
    packet = start_packet
    for step in range(1, 28):
        mid_z += 0.007
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=mid_z, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )

    assert packet.debug["rotation"]["slot_z"] != start_slot
    assert packet.debug["rotation"]["slot"] == packet.debug["rotation"]["slot_z"]
    assert packet.debug["rotation"]["deg_z"] > 0.0


def test_equivalent_rotation_slots_wrap_on_continuous_motion() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    mid_x = 0.0
    packet = None
    for step in range(1, 220):
        mid_x += 0.009
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=mid_x, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )

    assert packet is not None
    slot_x = packet.debug["rotation"]["slot_x"]
    assert 0 <= slot_x < ROT_SLOT_COUNT


def test_rotation_mode_switches_with_single_grab_open_sequence() -> None:
    reducer = TemporalReducer()

    frame_id = 1
    packet = None
    for _ in range(6):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    for _ in range(6):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    assert packet is not None
    assert packet.debug["rotation"]["mode_active"] is True
    assert packet.debug["rotation"]["mode_name"] == "ROTATE_ENABLED"


def test_rotation_mode_switches_with_curled_fist_grab_sequence() -> None:
    reducer = TemporalReducer()

    frame_id = 1
    packet = None
    for _ in range(6):
        observation = make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab")
        observation.landmarks = make_curled_grab_landmarks(wrist=Vec3(0.0, 0.0, 0.0))
        packet = reducer.reduce(
            observation,
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    for _ in range(6):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    assert packet is not None
    assert packet.debug["rotation"]["mode_active"] is True
    assert packet.debug["rotation"]["grab_detected"] is False


def test_rotation_mode_jitter_does_not_switch() -> None:
    reducer = TemporalReducer()

    frame_id = 1
    packet = None
    for _ in range(6):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    for _ in range(2):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    for _ in range(2):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    assert packet is not None
    assert packet.debug["rotation"]["mode_active"] is False


def test_rotation_mode_progress_times_out() -> None:
    reducer = TemporalReducer()

    frame_id = 1
    packet = None
    for _ in range(6):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    for _ in range(50):
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="grab"),
            frame_id=frame_id,
            timestamp_ms=frame_id * 16,
        )
        frame_id += 1

    assert packet is not None
    assert packet.debug["rotation"]["mode_active"] is False


def test_rotation_updates_before_gate_fully_opens() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    start_packet = reducer.reduce(
        make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.10, hand_pose="open"),
        frame_id=frame_id,
        timestamp_ms=frame_id * 16,
    )
    packet = reducer.reduce(
        make_eq_rotation_observation(mid_x=0.0, mid_y=0.20, mid_z=0.106, hand_pose="open"),
        frame_id=frame_id + 1,
        timestamp_ms=(frame_id + 1) * 16,
    )

    assert packet.debug["rotation"]["rotating"] is False
    assert packet.debug["rotation"]["deg_z"] > start_packet.debug["rotation"]["deg_z"]
    assert packet.debug["rotation"]["stack_z_deg"] > 0.0


def test_rotation_dominant_axis_suppresses_minor_cross_axis_noise() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    mid_x = 0.0
    mid_y = 0.20
    packet = None
    for step in range(1, 18):
        mid_x += 0.012
        mid_y += 0.0062
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=mid_x, mid_y=mid_y, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )

    assert packet is not None
    assert packet.debug["rotation"]["deg_x"] > 0.0
    assert packet.debug["rotation"]["deg_y"] > 0.0
    assert packet.debug["rotation"]["deg_y"] < (packet.debug["rotation"]["deg_x"] * 0.35)


def test_rotation_gate_hysteresis_keeps_state_stable_on_short_drop() -> None:
    reducer = TemporalReducer()
    frame_id = activate_rotation_mode(reducer, start_frame_id=1)

    packet = None
    mid_x = 0.0
    for step in range(6):
        mid_x += 0.009
        packet = reducer.reduce(
            make_eq_rotation_observation(mid_x=mid_x, mid_y=0.20, mid_z=0.10, hand_pose="open"),
            frame_id=frame_id + step,
            timestamp_ms=(frame_id + step) * 16,
        )
    assert packet is not None
    assert packet.debug["rotation"]["rotating"] is True

    weak = reducer.reduce(
        make_eq_rotation_observation(mid_x=mid_x + 0.0002, mid_y=0.20, mid_z=0.10, hand_pose="open"),
        frame_id=frame_id + 7,
        timestamp_ms=(frame_id + 7) * 16,
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
