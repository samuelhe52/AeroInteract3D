from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RenderView(StrEnum):
    HOME = "home"
    TABLE = "table"
    SETTING = "setting"
    CALIBRATION = "calibration"


class TableOverlay(StrEnum):
    NONE = "none"
    MENU = "menu"
    OPTION = "option"


@dataclass(slots=True)
class RenderingViewState:
    active_view: RenderView = RenderView.HOME

    def set_active_view(self, view: RenderView | str) -> RenderView:
        self.active_view = RenderView(view)
        return self.active_view


@dataclass(slots=True)
class TableOverlayState:
    active_overlay: TableOverlay = TableOverlay.NONE
    opened_at_ms: int = 0
    trigger_cooldown_until_ms: int = 0

    def set_active_overlay(self, overlay: TableOverlay | str, *, opened_at_ms: int | None = None) -> TableOverlay:
        self.active_overlay = TableOverlay(overlay)
        self.opened_at_ms = 0 if self.active_overlay == TableOverlay.NONE else int(opened_at_ms or 0)
        return self.active_overlay


@dataclass(slots=True)
class UIInputState:
    cursor_norm: tuple[float, float] = (0.5, 0.5)
    cursor_pixels: tuple[float, float] = (0.0, 0.0)
    visible: bool = False


@dataclass(slots=True)
class UICalibrationPreviewState:
    camera_midpoint: tuple[float, float] = (0.0, 0.0)
    source_cursor_norm: tuple[float, float] = (0.5, 0.5)
    source_cursor_pixels: tuple[float, float] = (0.0, 0.0)
    mapped_cursor_norm: tuple[float, float] = (0.5, 0.5)
    mapped_cursor_pixels: tuple[float, float] = (0.0, 0.0)
    window_size: tuple[int, int] = (0, 0)
    pinch_state: str | None = None
    visible: bool = False
    source_clamped: bool = False
    mapped_clamped: bool = False


@dataclass(slots=True)
class UISettingsState:
    data_panel_enabled: bool = True
    cam_preview_enabled: bool = True
    cursor_scale: float = 1.0
    cursor_opacity: float = 0.92
    brightness: float = 100.0
    volume: float = 50.0
    ui_cursor_scale_x: float = 1.0
    ui_cursor_scale_y: float = 1.0
    ui_cursor_offset_x: float = 0.0
    ui_cursor_offset_y: float = 0.0
    calibration_preview_enabled: bool = True

    BRIGHTNESS_MIN = 0.0
    BRIGHTNESS_MAX = 100.0
    BRIGHTNESS_STEP = 1.0
    VOLUME_MIN = 0.0
    VOLUME_MAX = 100.0
    VOLUME_STEP = 1.0
    CURSOR_SCALE_MIN = 0.5
    CURSOR_SCALE_MAX = 2.0
    CURSOR_SCALE_STEP = 0.01
    CURSOR_OPACITY_MIN = 0.2
    CURSOR_OPACITY_MAX = 1.0
    CURSOR_OPACITY_STEP = 0.01
    UI_CURSOR_SCALE_MIN = 0.5
    UI_CURSOR_SCALE_MAX = 1.5
    UI_CURSOR_SCALE_STEP = 0.01
    UI_CURSOR_OFFSET_MIN = -0.25
    UI_CURSOR_OFFSET_MAX = 0.25
    UI_CURSOR_OFFSET_STEP = 0.01

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    def set_cursor_scale(self, value: float) -> float:
        self.cursor_scale = round(self._clamp(value, self.CURSOR_SCALE_MIN, self.CURSOR_SCALE_MAX), 2)
        return self.cursor_scale

    def set_cursor_opacity(self, value: float) -> float:
        self.cursor_opacity = round(self._clamp(value, self.CURSOR_OPACITY_MIN, self.CURSOR_OPACITY_MAX), 2)
        return self.cursor_opacity

    def set_brightness(self, value: float) -> float:
        self.brightness = round(self._clamp(value, self.BRIGHTNESS_MIN, self.BRIGHTNESS_MAX), 2)
        return self.brightness

    def set_volume(self, value: float) -> float:
        self.volume = round(self._clamp(value, self.VOLUME_MIN, self.VOLUME_MAX), 2)
        return self.volume

    def set_ui_cursor_scale_x(self, value: float) -> float:
        self.ui_cursor_scale_x = round(self._clamp(value, self.UI_CURSOR_SCALE_MIN, self.UI_CURSOR_SCALE_MAX), 2)
        return self.ui_cursor_scale_x

    def set_ui_cursor_scale_y(self, value: float) -> float:
        self.ui_cursor_scale_y = round(self._clamp(value, self.UI_CURSOR_SCALE_MIN, self.UI_CURSOR_SCALE_MAX), 2)
        return self.ui_cursor_scale_y

    def set_ui_cursor_offset_x(self, value: float) -> float:
        self.ui_cursor_offset_x = round(self._clamp(value, self.UI_CURSOR_OFFSET_MIN, self.UI_CURSOR_OFFSET_MAX), 2)
        return self.ui_cursor_offset_x

    def set_ui_cursor_offset_y(self, value: float) -> float:
        self.ui_cursor_offset_y = round(self._clamp(value, self.UI_CURSOR_OFFSET_MIN, self.UI_CURSOR_OFFSET_MAX), 2)
        return self.ui_cursor_offset_y

    @property
    def brightness_scale(self) -> float:
        return 0.2 + 0.8 * (self.brightness / 100.0)

    def adjust_cursor_scale(self, step_count: int) -> float:
        return self.set_cursor_scale(self.cursor_scale + self.CURSOR_SCALE_STEP * float(step_count))

    def adjust_cursor_opacity(self, step_count: int) -> float:
        return self.set_cursor_opacity(self.cursor_opacity + self.CURSOR_OPACITY_STEP * float(step_count))