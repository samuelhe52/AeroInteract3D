from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode

from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .display_metrics import apply_root_display_scale
from .state import UICalibrationPreviewState, UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.calibration_view")


class CalibrationUIView:
    TITLE_TEXT = "cursor calibration"
    SUBTITLE_TEXT = "f2 open  esc back  tab focus  arrows adjust  r reset"
    PARAMETER_KEYS = (
        "ui_cursor_scale_x",
        "ui_cursor_scale_y",
        "ui_cursor_offset_x",
        "ui_cursor_offset_y",
    )
    PARAMETER_LABELS = {
        "ui_cursor_scale_x": "scale x",
        "ui_cursor_scale_y": "scale y",
        "ui_cursor_offset_x": "offset x",
        "ui_cursor_offset_y": "offset y",
    }
    BUTTON_STYLES = {
        "idle": {
            "frameColor": (0.18, 0.22, 0.28, 1.0),
            "textColor": (0.96, 0.95, 0.92, 1.0),
        },
        "hover": {
            "frameColor": (0.26, 0.32, 0.40, 1.0),
            "textColor": (1.0, 0.98, 0.94, 1.0),
        },
        "pressed": {
            "frameColor": (0.78, 0.28, 0.24, 1.0),
            "textColor": (1.0, 0.96, 0.92, 1.0),
        },
    }
    SLIDER_STYLES = {
        "idle": {
            "trackColor": (0.71, 0.73, 0.70, 1.0),
            "fillColor": (0.82, 0.34, 0.28, 1.0),
            "knobColor": (0.20, 0.24, 0.30, 1.0),
            "labelColor": (0.12, 0.15, 0.19, 1.0),
            "valueColor": (0.12, 0.15, 0.19, 1.0),
            "knobHalfWidth": 12.0,
            "knobHalfHeight": 16.0,
        },
        "hover": {
            "trackColor": (0.62, 0.66, 0.69, 1.0),
            "fillColor": (0.88, 0.42, 0.30, 1.0),
            "knobColor": (0.24, 0.31, 0.39, 1.0),
            "labelColor": (0.10, 0.13, 0.18, 1.0),
            "valueColor": (0.10, 0.13, 0.18, 1.0),
            "knobHalfWidth": 13.0,
            "knobHalfHeight": 17.0,
        },
        "active": {
            "trackColor": (0.50, 0.55, 0.58, 1.0),
            "fillColor": (0.93, 0.28, 0.22, 1.0),
            "knobColor": (0.86, 0.25, 0.21, 1.0),
            "labelColor": (0.12, 0.15, 0.19, 1.0),
            "valueColor": (0.82, 0.24, 0.20, 1.0),
            "knobHalfWidth": 15.0,
            "knobHalfHeight": 19.0,
        },
    }

    def __init__(
        self,
        pixel2d,
        window_size_provider: Callable[[], tuple[int, int]],
        on_button_activated: Callable[[str], None] | None = None,
        *,
        display_scale_provider: Callable[[], float] | None = None,
    ) -> None:
        self._pixel2d = pixel2d
        self._window_size_provider = window_size_provider
        self._on_button_activated = on_button_activated
        self._display_scale_provider = display_scale_provider or (lambda: 1.0)
        self._root: Optional[DirectFrame] = None
        self._overlay_root: Optional[DirectFrame] = None
        self._title: Optional[NodePath] = None
        self._subtitle: Optional[NodePath] = None
        self._hint: Optional[NodePath] = None
        self._status: Optional[NodePath] = None
        self._notes: Optional[NodePath] = None
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_actions: list[str] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._button_index_by_action: dict[str, int] = {}
        self._row_labels: dict[str, NodePath] = {}
        self._value_nodes: dict[str, NodePath] = {}
        self._slider_tracks: dict[str, DirectFrame] = {}
        self._slider_fills: dict[str, DirectFrame] = {}
        self._slider_knobs: dict[str, DirectFrame] = {}
        self._slider_bounds: dict[str, UIButtonBounds] = {}
        self._slider_visual_states: dict[str, str] = {}
        self._source_frame: Optional[DirectFrame] = None
        self._source_title: Optional[NodePath] = None
        self._source_crosshair_h: Optional[DirectFrame] = None
        self._source_crosshair_v: Optional[DirectFrame] = None
        self._source_dot: Optional[DirectFrame] = None
        self._mapped_frame: Optional[DirectFrame] = None
        self._mapped_title: Optional[NodePath] = None
        self._mapped_crosshair_h: Optional[DirectFrame] = None
        self._mapped_crosshair_v: Optional[DirectFrame] = None
        self._mapped_dot: Optional[DirectFrame] = None
        self._cursor: Optional[DirectFrame] = None
        self._interaction_controller = UIButtonInteractionController()
        self._visible = False
        self._last_layout_size: tuple[int, int, float] | None = None
        self._last_cursor_state = UIInputState()
        self._calibration_preview = UICalibrationPreviewState()
        self._hover_slider_key: str | None = None
        self._active_slider_key: str | None = None
        self._selected_parameter_index = 0
        self._ui_settings = UISettingsState()
        self.init_view()

    @property
    def selected_parameter_key(self) -> str:
        return self.PARAMETER_KEYS[self._selected_parameter_index]

    @staticmethod
    def _create_text_node(
        parent,
        *,
        node_name: str,
        text: str,
        align: int,
        color: tuple[float, float, float, float],
    ) -> NodePath:
        text_node = TextNode(node_name)
        text_node.setText(text)
        text_node.setAlign(align)
        text_node.setTextColor(*color)
        return parent.attachNewNode(text_node)

    def _create_button(self, *, action: str, label: str) -> None:
        button = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=self.BUTTON_STYLES["idle"]["frameColor"],
            relief=1,
            borderWidth=(1, 1),
        )
        button_label = self._create_text_node(
            button,
            node_name=f"calibration_button_label_{len(self._buttons)}",
            text=label,
            align=TextNode.ACenter,
            color=self.BUTTON_STYLES["idle"]["textColor"],
        )
        button_index = len(self._buttons)
        self._buttons.append(button)
        self._button_labels.append(button_label)
        self._button_actions.append(action)
        self._button_visual_states.append("idle")
        self._button_index_by_action[action] = button_index

    def _create_slider_visuals(self, key: str) -> None:
        idle_style = self.SLIDER_STYLES["idle"]
        track = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=idle_style["trackColor"],
            relief=1,
            borderWidth=(1, 1),
        )
        fill = DirectFrame(
            parent=track,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=idle_style["fillColor"],
            relief=0,
        )
        knob = DirectFrame(
            parent=track,
            pos=(0, 0, 0),
            frameSize=(
                -idle_style["knobHalfWidth"],
                idle_style["knobHalfWidth"],
                -idle_style["knobHalfHeight"],
                idle_style["knobHalfHeight"],
            ),
            frameColor=idle_style["knobColor"],
            relief=1,
            borderWidth=(1, 1),
        )
        self._slider_tracks[key] = track
        self._slider_fills[key] = fill
        self._slider_knobs[key] = knob
        self._slider_visual_states[key] = "idle"

    def init_view(self) -> None:
        self._root = DirectFrame(
            parent=self._pixel2d,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.92, 0.93, 0.90, 1.0),
        )
        self._title = self._create_text_node(
            self._root,
            node_name="calibration_title",
            text=self.TITLE_TEXT,
            align=TextNode.ARight,
            color=(0.11, 0.14, 0.18, 1.0),
        )
        self._subtitle = self._create_text_node(
            self._root,
            node_name="calibration_subtitle",
            text=self.SUBTITLE_TEXT,
            align=TextNode.ARight,
            color=(0.33, 0.37, 0.41, 1.0),
        )
        self._hint = self._create_text_node(
            self._root,
            node_name="calibration_hint",
            text="selected: scale x",
            align=TextNode.ALeft,
            color=(0.16, 0.18, 0.22, 1.0),
        )
        self._status = self._create_text_node(
            self._root,
            node_name="calibration_status",
            text="",
            align=TextNode.ALeft,
            color=(0.34, 0.37, 0.40, 1.0),
        )
        self._notes = self._create_text_node(
            self._root,
            node_name="calibration_notes",
            text="scale changes how wide the usable cursor range is from the center\noffset shifts the whole mapped area without changing its size",
            align=TextNode.ALeft,
            color=(0.22, 0.25, 0.30, 1.0),
        )

        self._create_button(action="back_setting", label="back")
        self._create_button(action="reset_calibration", label="reset")

        for key in self.PARAMETER_KEYS:
            self._row_labels[key] = self._create_text_node(
                self._root,
                node_name=f"calibration_row_label_{key}",
                text=self.PARAMETER_LABELS[key],
                align=TextNode.ALeft,
                color=(0.12, 0.15, 0.19, 1.0),
            )
            self._value_nodes[key] = self._create_text_node(
                self._root,
                node_name=f"calibration_value_{key}",
                text="",
                align=TextNode.ARight,
                color=(0.12, 0.15, 0.19, 1.0),
            )
            self._create_slider_visuals(key)

        self._source_frame = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.95, 0.96, 0.93, 1.0),
            relief=1,
            borderWidth=(2, 2),
        )
        self._source_title = self._create_text_node(
            self._source_frame,
            node_name="calibration_source_title",
            text="source",
            align=TextNode.ALeft,
            color=(0.18, 0.18, 0.20, 1.0),
        )
        self._source_crosshair_h = DirectFrame(
            parent=self._source_frame,
            pos=(0, 0, 0),
            frameSize=(0, 1, -2, 2),
            frameColor=(0.86, 0.66, 0.14, 0.74),
            relief=0,
            sortOrder=20,
        )
        self._source_crosshair_v = DirectFrame(
            parent=self._source_frame,
            pos=(0, 0, 0),
            frameSize=(-2, 2, -1, 0),
            frameColor=(0.86, 0.66, 0.14, 0.74),
            relief=0,
            sortOrder=20,
        )
        self._source_dot = DirectFrame(
            parent=self._source_frame,
            pos=(0, 0, 0),
            frameSize=(-10, 10, -10, 10),
            frameColor=(0.90, 0.72, 0.22, 0.96),
            relief=1,
            borderWidth=(1, 1),
            sortOrder=21,
        )
        self._mapped_frame = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.95, 0.96, 0.93, 1.0),
            relief=1,
            borderWidth=(2, 2),
        )
        self._mapped_title = self._create_text_node(
            self._mapped_frame,
            node_name="calibration_mapped_title",
            text="mapped",
            align=TextNode.ALeft,
            color=(0.18, 0.18, 0.20, 1.0),
        )
        self._mapped_crosshair_h = DirectFrame(
            parent=self._mapped_frame,
            pos=(0, 0, 0),
            frameSize=(0, 1, -2, 2),
            frameColor=(0.88, 0.30, 0.26, 0.72),
            relief=0,
            sortOrder=20,
        )
        self._mapped_crosshair_v = DirectFrame(
            parent=self._mapped_frame,
            pos=(0, 0, 0),
            frameSize=(-2, 2, -1, 0),
            frameColor=(0.88, 0.30, 0.26, 0.72),
            relief=0,
            sortOrder=20,
        )
        self._mapped_dot = DirectFrame(
            parent=self._mapped_frame,
            pos=(0, 0, 0),
            frameSize=(-10, 10, -10, 10),
            frameColor=(0.90, 0.14, 0.12, 0.94),
            relief=1,
            borderWidth=(1, 1),
            sortOrder=21,
        )

        self._overlay_root = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.0, 0.0, 0.0, 0.0),
            relief=None,
            sortOrder=100,
        )
        self._cursor = DirectFrame(
            parent=self._overlay_root,
            pos=(0, 0, 0),
            frameSize=(-18, 18, -18, 18),
            frameColor=(0.94, 0.10, 0.10, 0.92),
            relief=1,
            borderWidth=(2, 2),
        )
        self._cursor.hide()
        self._refresh_setting_values()
        self._apply_cursor_style()
        self.update_layout(force=True)
        self._root.hide()
        logger.info("Calibration UI initialized successfully")

    def _apply_button_visual_state(self, index: int, state_name: str) -> None:
        style = self.BUTTON_STYLES[state_name]
        self._buttons[index]["frameColor"] = style["frameColor"]
        self._button_labels[index].node().setTextColor(*style["textColor"])
        self._button_visual_states[index] = state_name

    def _update_button_visuals(self, snapshot: UIButtonInteractionSnapshot) -> None:
        for index in range(len(self._buttons)):
            next_state = "idle"
            if snapshot.hovered_index == index:
                next_state = "hover"
            if snapshot.hovered_index == index and snapshot.pressed_index == index:
                next_state = "pressed"
            if self._button_visual_states[index] != next_state:
                self._apply_button_visual_state(index, next_state)

    def _slider_progress(self, key: str) -> float:
        settings = self._ui_settings
        if key == "ui_cursor_scale_x":
            return (settings.ui_cursor_scale_x - settings.UI_CURSOR_SCALE_MIN) / (settings.UI_CURSOR_SCALE_MAX - settings.UI_CURSOR_SCALE_MIN)
        if key == "ui_cursor_scale_y":
            return (settings.ui_cursor_scale_y - settings.UI_CURSOR_SCALE_MIN) / (settings.UI_CURSOR_SCALE_MAX - settings.UI_CURSOR_SCALE_MIN)
        if key == "ui_cursor_offset_x":
            return (settings.ui_cursor_offset_x - settings.UI_CURSOR_OFFSET_MIN) / (settings.UI_CURSOR_OFFSET_MAX - settings.UI_CURSOR_OFFSET_MIN)
        if key == "ui_cursor_offset_y":
            return (settings.ui_cursor_offset_y - settings.UI_CURSOR_OFFSET_MIN) / (settings.UI_CURSOR_OFFSET_MAX - settings.UI_CURSOR_OFFSET_MIN)
        return 0.0

    def _current_parameter_value(self, key: str) -> float:
        return float(getattr(self._ui_settings, key))

    def _refresh_hint_text(self) -> None:
        if self._hint is None or self._status is None:
            return
        focus_key = self.selected_parameter_key
        self._hint.node().setText(
            f"selected: {self.PARAMETER_LABELS[focus_key]}   tab switch   shift+arrows fast"
        )
        source_status = "clamped" if self._calibration_preview.source_clamped else "in-bounds"
        mapped_status = "clamped" if self._calibration_preview.mapped_clamped else "in-bounds"
        if not self._calibration_preview.visible:
            source_status = "not tracked"
            mapped_status = "not tracked"
        self._status.node().setText(
            "\n".join(
                (
                    f"cam: {self._calibration_preview.camera_midpoint[0]:+.2f} {self._calibration_preview.camera_midpoint[1]:+.2f}",
                    f"src: {self._calibration_preview.source_cursor_norm[0]:.2f} {self._calibration_preview.source_cursor_norm[1]:.2f} ({source_status})",
                    f"ui : {self._calibration_preview.mapped_cursor_norm[0]:.2f} {self._calibration_preview.mapped_cursor_norm[1]:.2f} ({mapped_status})",
                    f"px : {int(round(self._calibration_preview.mapped_cursor_pixels[0]))} {int(round(self._calibration_preview.mapped_cursor_pixels[1]))}",
                    f"pinch: {self._calibration_preview.pinch_state or 'none'}",
                )
            )
        )

    def _refresh_setting_values(self) -> None:
        self._value_nodes["ui_cursor_scale_x"].node().setText(f"{self._ui_settings.ui_cursor_scale_x:.2f}")
        self._value_nodes["ui_cursor_scale_y"].node().setText(f"{self._ui_settings.ui_cursor_scale_y:.2f}")
        self._value_nodes["ui_cursor_offset_x"].node().setText(f"{self._ui_settings.ui_cursor_offset_x:+.2f}")
        self._value_nodes["ui_cursor_offset_y"].node().setText(f"{self._ui_settings.ui_cursor_offset_y:+.2f}")
        for key, fill in self._slider_fills.items():
            track = self._slider_tracks[key]
            knob = self._slider_knobs[key]
            frame_size = track["frameSize"]
            width = max(float(frame_size[1]) - float(frame_size[0]), 1.0)
            progress = self._slider_progress(key)
            fill["frameSize"] = (0, width * progress, float(frame_size[2]), float(frame_size[3]))
            knob.setPos(width * progress, 0, (float(frame_size[2]) + float(frame_size[3])) * 0.5)
        self._refresh_slider_visuals()
        self._refresh_hint_text()

    def _apply_slider_visual_state(self, key: str, state_name: str) -> None:
        if self._slider_visual_states.get(key) == state_name:
            return
        style = self.SLIDER_STYLES[state_name]
        track = self._slider_tracks[key]
        fill = self._slider_fills[key]
        knob = self._slider_knobs[key]
        label = self._row_labels[key]
        value = self._value_nodes[key]
        track["frameColor"] = style["trackColor"]
        fill["frameColor"] = style["fillColor"]
        knob["frameColor"] = style["knobColor"]
        knob["frameSize"] = (
            -style["knobHalfWidth"],
            style["knobHalfWidth"],
            -style["knobHalfHeight"],
            style["knobHalfHeight"],
        )
        label.node().setTextColor(*style["labelColor"])
        value.node().setTextColor(*style["valueColor"])
        self._slider_visual_states[key] = state_name

    def _refresh_slider_visuals(self) -> None:
        for key in self.PARAMETER_KEYS:
            next_state = "idle"
            if self._hover_slider_key == key:
                next_state = "hover"
            if self.selected_parameter_key == key or self._active_slider_key == key:
                next_state = "active"
            self._apply_slider_visual_state(key, next_state)

    def _apply_cursor_style(self) -> None:
        if self._cursor is None:
            return
        extent = 18.0 * self._ui_settings.cursor_scale
        self._cursor["frameSize"] = (-extent, extent, -extent, extent)
        self._cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)

    def _update_preview_zone(
        self,
        frame: DirectFrame,
        crosshair_h: DirectFrame,
        crosshair_v: DirectFrame,
        dot: DirectFrame,
        cursor_norm: tuple[float, float],
    ) -> None:
        frame_size = frame["frameSize"]
        zone_width = float(frame_size[1]) - float(frame_size[0])
        zone_height = abs(float(frame_size[2]) - float(frame_size[3]))
        inset_left = 18.0
        inset_top = 34.0
        inset_right = 18.0
        inset_bottom = 18.0
        preview_width = max(zone_width - inset_left - inset_right, 24.0)
        preview_height = max(zone_height - inset_top - inset_bottom, 24.0)
        center_x = inset_left + preview_width * max(0.0, min(1.0, cursor_norm[0]))
        center_y = -(inset_top + preview_height * max(0.0, min(1.0, cursor_norm[1])))
        crosshair_h["frameSize"] = (0, preview_width, -2, 2)
        crosshair_v["frameSize"] = (-2, 2, -preview_height, 0)
        crosshair_h.setPos(inset_left, 0, center_y)
        crosshair_v.setPos(center_x, 0, -inset_top)
        dot.setPos(center_x, 0, center_y)

    def _update_mapping_preview(self) -> None:
        if (
            self._source_frame is None
            or self._source_crosshair_h is None
            or self._source_crosshair_v is None
            or self._source_dot is None
            or self._mapped_frame is None
            or self._mapped_crosshair_h is None
            or self._mapped_crosshair_v is None
            or self._mapped_dot is None
        ):
            return
        self._update_preview_zone(
            self._source_frame,
            self._source_crosshair_h,
            self._source_crosshair_v,
            self._source_dot,
            self._calibration_preview.source_cursor_norm,
        )
        self._update_preview_zone(
            self._mapped_frame,
            self._mapped_crosshair_h,
            self._mapped_crosshair_v,
            self._mapped_dot,
            self._calibration_preview.mapped_cursor_norm,
        )
        self._refresh_hint_text()

    def update_calibration_preview(self, state: UICalibrationPreviewState) -> None:
        self._calibration_preview = state
        self._update_mapping_preview()

    def _slider_value_from_progress(self, key: str, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if key in {"ui_cursor_scale_x", "ui_cursor_scale_y"}:
            value = self._ui_settings.UI_CURSOR_SCALE_MIN + progress * (
                self._ui_settings.UI_CURSOR_SCALE_MAX - self._ui_settings.UI_CURSOR_SCALE_MIN
            )
            return round(round(value / self._ui_settings.UI_CURSOR_SCALE_STEP) * self._ui_settings.UI_CURSOR_SCALE_STEP, 2)
        value = self._ui_settings.UI_CURSOR_OFFSET_MIN + progress * (
            self._ui_settings.UI_CURSOR_OFFSET_MAX - self._ui_settings.UI_CURSOR_OFFSET_MIN
        )
        return round(round(value / self._ui_settings.UI_CURSOR_OFFSET_STEP) * self._ui_settings.UI_CURSOR_OFFSET_STEP, 2)

    def _set_slider_value(self, key: str, value: float) -> float | None:
        if key == "ui_cursor_scale_x":
            next_value = self._ui_settings.set_ui_cursor_scale_x(value)
        elif key == "ui_cursor_scale_y":
            next_value = self._ui_settings.set_ui_cursor_scale_y(value)
        elif key == "ui_cursor_offset_x":
            next_value = self._ui_settings.set_ui_cursor_offset_x(value)
        elif key == "ui_cursor_offset_y":
            next_value = self._ui_settings.set_ui_cursor_offset_y(value)
        else:
            return None

        self._refresh_setting_values()
        if self._on_button_activated is not None:
            self._on_button_activated(f"set_{key}:{next_value}")
        return next_value

    def select_next_parameter(self, step: int = 1) -> str:
        self._selected_parameter_index = (self._selected_parameter_index + step) % len(self.PARAMETER_KEYS)
        self._refresh_slider_visuals()
        self._refresh_hint_text()
        return self.selected_parameter_key

    def adjust_selected_parameter(self, step_count: int) -> float | None:
        key = self.selected_parameter_key
        current_value = self._current_parameter_value(key)
        if key in {"ui_cursor_scale_x", "ui_cursor_scale_y"}:
            next_value = current_value + self._ui_settings.UI_CURSOR_SCALE_STEP * float(step_count)
        else:
            next_value = current_value + self._ui_settings.UI_CURSOR_OFFSET_STEP * float(step_count)
        return self._set_slider_value(key, next_value)

    def trigger_reset(self) -> None:
        if self._on_button_activated is not None:
            self._on_button_activated("reset_calibration")

    def _slider_key_at_cursor(self, state: UIInputState) -> str | None:
        if not state.visible:
            return None
        cursor_x, cursor_y = state.cursor_pixels
        for key, bounds in self._slider_bounds.items():
            if bounds.contains(cursor_x, cursor_y):
                return key
        return None

    def _update_slider_interaction(self, state: UIInputState, pinch_state: PinchState | None) -> None:
        hovered_slider = self._slider_key_at_cursor(state)
        self._hover_slider_key = hovered_slider
        is_dragging = pinch_state in {"pinched", "release_candidate"}
        if self._active_slider_key is None and pinch_state == "pinched" and hovered_slider is not None:
            self._active_slider_key = hovered_slider
            self._selected_parameter_index = self.PARAMETER_KEYS.index(hovered_slider)
        self._refresh_slider_visuals()
        if self._active_slider_key is None:
            return
        if not is_dragging or not state.visible:
            self._active_slider_key = None
            self._refresh_slider_visuals()
            return
        bounds = self._slider_bounds[self._active_slider_key]
        width = max(bounds.right - bounds.left, 1.0)
        progress = (state.cursor_pixels[0] - bounds.left) / width
        self._set_slider_value(self._active_slider_key, self._slider_value_from_progress(self._active_slider_key, progress))

    def set_ui_settings(self, settings: UISettingsState) -> None:
        self._ui_settings.data_panel_enabled = settings.data_panel_enabled
        self._ui_settings.cam_preview_enabled = settings.cam_preview_enabled
        self._ui_settings.cursor_scale = settings.cursor_scale
        self._ui_settings.cursor_opacity = settings.cursor_opacity
        self._ui_settings.brightness = settings.brightness
        self._ui_settings.volume = settings.volume
        self._ui_settings.ui_cursor_scale_x = settings.ui_cursor_scale_x
        self._ui_settings.ui_cursor_scale_y = settings.ui_cursor_scale_y
        self._ui_settings.ui_cursor_offset_x = settings.ui_cursor_offset_x
        self._ui_settings.ui_cursor_offset_y = settings.ui_cursor_offset_y
        self._ui_settings.calibration_preview_enabled = settings.calibration_preview_enabled
        self._refresh_setting_values()
        self._apply_cursor_style()

    def update_layout(self, force: bool = False) -> None:
        if self._root is None or self._title is None or self._subtitle is None or self._hint is None or self._status is None or self._notes is None:
            return
        width, height = self._window_size_provider()
        width = max(int(width), 800)
        height = max(int(height), 450)
        display_scale = apply_root_display_scale(self._root, self._display_scale_provider())
        next_size = (width, height, display_scale)
        if not force and next_size == self._last_layout_size:
            return

        self._last_layout_size = next_size
        self._root["frameSize"] = (0, width, -height, 0)
        if self._overlay_root is not None:
            self._overlay_root["frameSize"] = (0, width, -height, 0)

        short_edge = min(width, height)
        title_margin_right = max(int(width * 0.05), 36)
        title_margin_top = max(int(height * 0.07), 32)
        title_scale = max(short_edge * 0.044, 24)
        subtitle_scale = max(title_scale * 0.34, 14)
        self._title.setPos(width - title_margin_right, 0, -(title_margin_top + title_scale * 0.45))
        self._title.setScale(title_scale)
        self._subtitle.setPos(width - title_margin_right, 0, -(title_margin_top + title_scale * 1.15))
        self._subtitle.setScale(subtitle_scale)

        content_left = max(int(width * 0.055), 44)
        content_top = max(int(height * 0.17), 92)
        content_width = width - content_left * 2
        gutter = max(int(width * 0.025), 22)
        left_panel_width = max(int(content_width * 0.42), 280)

        back_width = max(int(width * 0.10), 118)
        back_height = max(int(height * 0.07), 52)
        self._layout_button("back_setting", content_left, max(int(height * 0.07), 34), back_width, back_height)
        self._layout_button("reset_calibration", content_left + back_width + 18, max(int(height * 0.07), 34), back_width, back_height)

        hint_scale = max(short_edge * 0.018, 13)
        status_scale = max(short_edge * 0.015, 10)
        notes_scale = max(short_edge * 0.014, 10)
        self._hint.setPos(content_left, 0, -(content_top - 10))
        self._hint.setScale(hint_scale)
        status_top = content_top + max(int(hint_scale * 2.2), 30)
        self._status.setPos(content_left, 0, -status_top)
        self._status.setScale(status_scale)
        notes_top = status_top + max(int(status_scale * 6.0), 52)
        self._notes.setPos(content_left, 0, -notes_top)
        self._notes.setScale(notes_scale)

        preview_gap = max(int(height * 0.03), 16)
        bottom_margin = max(int(height * 0.06), 28)
        row_gap = max(min(int(height * 0.022), 18), 8)
        slider_track_height = max(min(int(height * 0.020), 14), 10)
        controls_reserved = row_gap * 3 + slider_track_height * 4 + max(int(height * 0.19), 132)
        preview_top = notes_top + max(int(notes_scale * 4.8), 48)
        remaining_after_preview_top = max(height - bottom_margin - preview_top, 120)
        preview_height = max(min(int((remaining_after_preview_top - controls_reserved) * 0.60), 170), 88)
        preview_width = max(int((content_width - gutter) * 0.5), 220)
        if self._source_frame is not None and self._source_title is not None:
            self._source_frame.setPos(content_left, 0, -preview_top)
            self._source_frame["frameSize"] = (0, preview_width, -preview_height, 0)
            self._source_title.setPos(14, 0, -22)
            self._source_title.setScale(max(short_edge * 0.015, 11))
        if self._mapped_frame is not None and self._mapped_title is not None:
            mapped_left = content_left + preview_width + gutter
            self._mapped_frame.setPos(mapped_left, 0, -preview_top)
            self._mapped_frame["frameSize"] = (0, preview_width, -preview_height, 0)
            self._mapped_title.setPos(14, 0, -22)
            self._mapped_title.setScale(max(short_edge * 0.015, 11))

        parameter_top = preview_top + preview_height + preview_gap
        available_for_controls = max(height - bottom_margin - parameter_top, 120)
        row_height = max(min(int((available_for_controls - row_gap * 3) / 4), 58), 34)
        slider_label_scale = max(short_edge * 0.020, 15)
        slider_value_scale = max(short_edge * 0.022, 16)
        slider_track_width = max(int(left_panel_width - 40), 220)
        slider_value_x = content_left + left_panel_width

        for index, key in enumerate(self.PARAMETER_KEYS):
            top = parameter_top + index * (row_height + row_gap)
            label = self._row_labels[key]
            value = self._value_nodes[key]
            label.setPos(content_left, 0, -(top + 18))
            label.setScale(slider_label_scale)
            value.setPos(slider_value_x, 0, -(top + 18))
            value.setScale(slider_value_scale)
            slider_top = top + row_height - max(int(row_height * 0.34), 18)
            track = self._slider_tracks[key]
            track.setPos(content_left, 0, -(slider_top + slider_track_height * 0.5))
            track["frameSize"] = (0, slider_track_width, -slider_track_height, 0)
            self._slider_fills[key]["frameSize"] = (0, slider_track_width * self._slider_progress(key), -slider_track_height, 0)
            self._slider_bounds[key] = UIButtonBounds(
                left=content_left - 8,
                top=slider_top - 10,
                right=content_left + slider_track_width + 8,
                bottom=slider_top + slider_track_height + 10,
            )

        for index in range(len(self._buttons)):
            self._apply_button_visual_state(index, self._button_visual_states[index])
        self._refresh_setting_values()
        self._update_mapping_preview()

    def _layout_button(self, action: str, left: int, top: int, width_px: int, height_px: int) -> None:
        button_index = self._button_index_by_action[action]
        button = self._buttons[button_index]
        button.setPos(left, 0, -top)
        button["frameSize"] = (0, width_px, -height_px, 0)
        label = self._button_labels[button_index]
        label.setPos(width_px * 0.5, 0, -(height_px * 0.60))
        label.setScale(max(height_px * 0.30, 16))
        bounds = UIButtonBounds(
            left=left,
            top=top,
            right=left + width_px,
            bottom=top + height_px,
        )
        if button_index < len(self._button_bounds):
            self._button_bounds[button_index] = bounds
        else:
            self._button_bounds.append(bounds)

    def update_cursor(self, state: UIInputState, pinch_state: PinchState | None = None) -> None:
        if self._cursor is None:
            return
        self._last_cursor_state = state
        interaction_state = state if self._visible else UIInputState(visible=False)
        self._update_slider_interaction(interaction_state, pinch_state)
        snapshot = self._interaction_controller.update(
            interaction_state,
            pinch_state=pinch_state,
            button_bounds=self._button_bounds,
        )
        self._update_button_visuals(snapshot)
        if snapshot.activated_index is not None and self._on_button_activated is not None:
            self._on_button_activated(self._button_actions[snapshot.activated_index])
        if not self._visible or not state.visible:
            self._cursor.hide()
            return
        self._cursor.show()
        self._cursor.setPos(float(state.cursor_pixels[0]), 0, -float(state.cursor_pixels[1]))

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._root is None:
            return
        if visible:
            self.update_layout()
            self._root.show()
            return
        self._interaction_controller.reset()
        self._hover_slider_key = None
        self._active_slider_key = None
        self._refresh_slider_visuals()
        for index in range(len(self._buttons)):
            if self._button_visual_states[index] != "idle":
                self._apply_button_visual_state(index, "idle")
        if self._cursor is not None:
            self._cursor.hide()
        self._root.hide()

    def is_visible(self) -> bool:
        return self._visible

    def destroy(self) -> None:
        for fill in self._slider_fills.values():
            fill.destroy()
        self._slider_fills.clear()
        for knob in self._slider_knobs.values():
            knob.destroy()
        self._slider_knobs.clear()
        for track in self._slider_tracks.values():
            track.destroy()
        self._slider_tracks.clear()
        for value_node in self._value_nodes.values():
            value_node.removeNode()
        self._value_nodes.clear()
        for row_label in self._row_labels.values():
            row_label.removeNode()
        self._row_labels.clear()
        for label in self._button_labels:
            label.removeNode()
        self._button_labels.clear()
        for button in self._buttons:
            button.destroy()
        self._buttons.clear()
        if self._source_crosshair_h is not None:
            self._source_crosshair_h.destroy()
            self._source_crosshair_h = None
        if self._source_crosshair_v is not None:
            self._source_crosshair_v.destroy()
            self._source_crosshair_v = None
        if self._source_dot is not None:
            self._source_dot.destroy()
            self._source_dot = None
        if self._source_title is not None:
            self._source_title.removeNode()
            self._source_title = None
        if self._source_frame is not None:
            self._source_frame.destroy()
            self._source_frame = None
        if self._mapped_crosshair_h is not None:
            self._mapped_crosshair_h.destroy()
            self._mapped_crosshair_h = None
        if self._mapped_crosshair_v is not None:
            self._mapped_crosshair_v.destroy()
            self._mapped_crosshair_v = None
        if self._mapped_dot is not None:
            self._mapped_dot.destroy()
            self._mapped_dot = None
        if self._mapped_title is not None:
            self._mapped_title.removeNode()
            self._mapped_title = None
        if self._mapped_frame is not None:
            self._mapped_frame.destroy()
            self._mapped_frame = None
        if self._cursor is not None:
            self._cursor.destroy()
            self._cursor = None
        if self._hint is not None:
            self._hint.removeNode()
            self._hint = None
        if self._status is not None:
            self._status.removeNode()
            self._status = None
        if self._notes is not None:
            self._notes.removeNode()
            self._notes = None
        if self._subtitle is not None:
            self._subtitle.removeNode()
            self._subtitle = None
        if self._title is not None:
            self._title.removeNode()
            self._title = None
        if self._root is not None:
            self._root.destroy()
            self._root = None
        self._overlay_root = None
        logger.info("Calibration UI cleaned up")
