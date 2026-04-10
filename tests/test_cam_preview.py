from __future__ import annotations

import pytest

from src.rendering.debug.cam_preview import CameraPreviewManager


class FakePreviewNode:
    def __init__(self) -> None:
        self.hidden = False
        self.color_scale = None

    def hide(self) -> None:
        self.hidden = True

    def show(self) -> None:
        self.hidden = False

    def setColorScale(self, *values: float) -> None:
        self.color_scale = values


@pytest.fixture
def camera_preview_manager() -> CameraPreviewManager:
    manager = CameraPreviewManager.__new__(CameraPreviewManager)
    manager._camera_preview_node = FakePreviewNode()
    return manager


def test_visibility_uses_preview_node_without_debug_chrome_attributes(
    camera_preview_manager: CameraPreviewManager,
) -> None:
    camera_preview_manager.set_visible(False)
    assert camera_preview_manager._camera_preview_node.hidden is True

    camera_preview_manager.set_visible(True)
    assert camera_preview_manager._camera_preview_node.hidden is False


def test_brightness_uses_preview_node_without_debug_chrome_attributes(
    camera_preview_manager: CameraPreviewManager,
) -> None:
    camera_preview_manager.set_brightness(0.4)

    assert camera_preview_manager._camera_preview_node.color_scale == pytest.approx((0.4, 0.4, 0.4, 1.0))
