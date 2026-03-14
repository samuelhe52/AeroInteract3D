from __future__ import annotations

import pytest

from src.contracts import Vec3
from src.gesture.runtime import landmark_to_camera_vec3


def test_landmark_to_camera_vec3_preserves_negative_xy_coordinates() -> None:
    point = landmark_to_camera_vec3(Vec3(0.25, 0.75, 0.0), depth_hint=0.0)

    assert point.x == pytest.approx(-0.5)
    assert point.y == pytest.approx(-0.5)


def test_landmark_to_camera_vec3_clips_only_at_camera_space_bounds() -> None:
    point = landmark_to_camera_vec3(Vec3(-0.2, 1.4, 0.0), depth_hint=0.0)

    assert point.x == -1.0
    assert point.y == -1.0