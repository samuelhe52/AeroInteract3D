from __future__ import annotations

from src.constants import TEMPORAL_TRACKING_TEMPORARY_LOSS_FRAMES
from src.contracts import Vec3
from src.gesture.runtime import RawHandObservation, normalized_pinch_distance
from src.gesture.temporal import TemporalReducer


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