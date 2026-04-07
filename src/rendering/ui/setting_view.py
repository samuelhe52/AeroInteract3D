from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode

from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .state import UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.setting_view")


class SettingUIView:
    TITLE_TEXT = "settings"
    SUBTITLE_TEXT = "tune live ui controls"
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
    SLIDER_KEYS = ("cursor_scale", "cursor_opacity", "brightness", "volume")

    def __init__(
        self,
        pixel2d,
        window_size_provider: Callable[[], tuple[int, int]],
        on_button_activated: Callable[[str], None] | None = None,
    ) -> None:
        self._pixel2d = pixel2d
        self._window_size_provider = window_size_provider
        self._on_button_activated = on_button_activated
        self._root: Optional[DirectFrame] = None
        self._overlay_root: Optional[DirectFrame] = None
        self._title: Optional[NodePath] = None
        self._subtitle: Optional[NodePath] = None
        self._row_labels: dict[str, NodePath] = {}
        self._value_nodes: dict[str, NodePath] = {}
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_actions: list[str] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._button_index_by_action: dict[str, int] = {}
        self._slider_tracks: dict[str, DirectFrame] = {}
        self._slider_fills: dict[str, DirectFrame] = {}
        self._slider_knobs: dict[str, DirectFrame] = {}
        self._slider_bounds: dict[str, UIButtonBounds] = {}
        self._preview_panel: Optional[DirectFrame] = None
        self._preview_panel_title: Optional[NodePath] = None
        self._preview_cursor: Optional[DirectFrame] = None
        self._mapping_panel: Optional[DirectFrame] = None
        self._mapping_title: Optional[NodePath] = None
        self._mapping_hint: Optional[NodePath] = None
        self._mapping_crosshair_h: Optional[DirectFrame] = None
        self._mapping_crosshair_v: Optional[DirectFrame] = None
        self._mapping_dot: Optional[DirectFrame] = None
        self._interaction_controller = UIButtonInteractionController()
        self._cursor: Optional[DirectFrame] = None
        self._visible = False
        self._last_layout_size: tuple[int, int] | None = None
        self._last_cursor_state = UIInputState()
        self._active_slider_key: str | None = None
        self._ui_settings = UISettingsState()
        self.init_view()

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
            node_name=f"setting_button_label_{len(self._buttons)}",
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
        track = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.76, 0.78, 0.74, 1.0),
            relief=0,
        )
        fill = DirectFrame(
            parent=track,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.83, 0.33, 0.28, 1.0),
            relief=0,
        )
        knob = DirectFrame(
            parent=track,
            pos=(0, 0, 0),
            frameSize=(-10, 10, -14, 14),
            frameColor=(0.20, 0.24, 0.30, 1.0),
            relief=1,
            borderWidth=(1, 1),
        )
        self._slider_tracks[key] = track
        self._slider_fills[key] = fill
        self._slider_knobs[key] = knob

    def init_view(self) -> None:
        self._root = DirectFrame(
            parent=self._pixel2d,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.90, 0.91, 0.88, 1.0),
        )
        self._title = self._create_text_node(
            self._root,
            node_name="setting_title",
            text=self.TITLE_TEXT,
            align=TextNode.ARight,
            color=(0.11, 0.14, 0.18, 1.0),
        )
        self._subtitle = self._create_text_node(
            self._root,
            node_name="setting_subtitle",
            text=self.SUBTITLE_TEXT,
            align=TextNode.ARight,
            color=(0.33, 0.37, 0.41, 1.0),
        )

        for key, label in (
            ("cursor_scale", "cursor size"),
            ("cursor_opacity", "cursor opacity"),
            ("brightness", "brightness"),
            ("volume", "volume"),
        ):
            self._row_labels[key] = self._create_text_node(
                self._root,
                node_name=f"setting_row_label_{key}",
                text=label,
                align=TextNode.ALeft,
                color=(0.12, 0.15, 0.19, 1.0),
            )
            self._value_nodes[key] = self._create_text_node(
                self._root,
                node_name=f"setting_value_{key}",
                text="",
                align=TextNode.ARight,
                color=(0.12, 0.15, 0.19, 1.0),
            )
            self._create_slider_visuals(key)

        self._create_button(action="back_home", label="back")
        self._create_button(action="data_panel_toggle", label="data panel: on")
        self._create_button(action="cam_preview_toggle", label="cam preview: on")

        self._preview_panel = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.84, 0.86, 0.82, 1.0),
            relief=1,
            borderWidth=(1, 1),
        )
        self._preview_panel_title = self._create_text_node(
            self._preview_panel,
            node_name="setting_preview_panel_title",
            text="cursor preview",
            align=TextNode.ACenter,
            color=(0.16, 0.18, 0.22, 1.0),
        )
        self._preview_cursor = DirectFrame(
            parent=self._preview_panel,
            pos=(0, 0, 0),
            frameSize=(-18, 18, -18, 18),
            frameColor=(0.94, 0.10, 0.10, 0.92),
            relief=1,
            borderWidth=(2, 2),
        )

        self._mapping_panel = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.86, 0.88, 0.84, 1.0),
            relief=1,
            borderWidth=(1, 1),
        )
        self._mapping_title = self._create_text_node(
            self._mapping_panel,
            node_name="setting_mapping_title",
            text="cursor mapping preview",
            align=TextNode.ALeft,
            color=(0.16, 0.18, 0.22, 1.0),
        )
        self._mapping_hint = self._create_text_node(
            self._mapping_panel,
            node_name="setting_mapping_hint",
            text="live calibration panel placeholder",
            align=TextNode.ALeft,
            color=(0.34, 0.37, 0.40, 1.0),
        )
        self._mapping_crosshair_h = DirectFrame(
            parent=self._mapping_panel,
            pos=(0, 0, 0),
            frameSize=(-18, 18, -1, 1),
            frameColor=(0.88, 0.30, 0.26, 0.72),
            relief=0,
        )
        self._mapping_crosshair_v = DirectFrame(
            parent=self._mapping_panel,
            pos=(0, 0, 0),
            frameSize=(-1, 1, -18, 18),
            frameColor=(0.88, 0.30, 0.26, 0.72),
            relief=0,
        )
        self._mapping_dot = DirectFrame(
            parent=self._mapping_panel,
            pos=(0, 0, 0),
            frameSize=(-7, 7, -7, 7),
            frameColor=(0.90, 0.14, 0.12, 0.94),
            relief=0,
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
        logger.info("Setting UI initialized successfully")

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
        if key == "cursor_scale":
            return (settings.cursor_scale - settings.CURSOR_SCALE_MIN) / (settings.CURSOR_SCALE_MAX - settings.CURSOR_SCALE_MIN)
        if key == "cursor_opacity":
            return (settings.cursor_opacity - settings.CURSOR_OPACITY_MIN) / (settings.CURSOR_OPACITY_MAX - settings.CURSOR_OPACITY_MIN)
        if key == "brightness":
            return (settings.brightness - settings.BRIGHTNESS_MIN) / (settings.BRIGHTNESS_MAX - settings.BRIGHTNESS_MIN)
        if key == "volume":
            return (settings.volume - settings.VOLUME_MIN) / (settings.VOLUME_MAX - settings.VOLUME_MIN)
        return 0.0

    def _refresh_setting_values(self) -> None:
        self._value_nodes["cursor_scale"].node().setText(f"{self._ui_settings.cursor_scale:.2f}x")
        self._value_nodes["cursor_opacity"].node().setText(f"{self._ui_settings.cursor_opacity:.2f}")
        self._value_nodes["brightness"].node().setText(f"{int(round(self._ui_settings.brightness))}")
        self._value_nodes["volume"].node().setText(f"{int(round(self._ui_settings.volume))}")

        self._button_labels[self._button_index_by_action["data_panel_toggle"]].node().setText(
            f"data panel: {'on' if self._ui_settings.data_panel_enabled else 'off'}"
        )
        self._button_labels[self._button_index_by_action["cam_preview_toggle"]].node().setText(
            f"cam preview: {'on' if self._ui_settings.cam_preview_enabled else 'off'}"
        )

        for key, fill in self._slider_fills.items():
            track = self._slider_tracks[key]
            knob = self._slider_knobs[key]
            frame_size = track["frameSize"]
            width = max(float(frame_size[1]) - float(frame_size[0]), 1.0)
            progress = self._slider_progress(key)
            fill["frameSize"] = (0, width * progress, float(frame_size[2]), float(frame_size[3]))
            knob.setPos(width * progress, 0, (float(frame_size[2]) + float(frame_size[3])) * 0.5)

    def _apply_cursor_style(self) -> None:
        if self._cursor is not None:
            extent = 18.0 * self._ui_settings.cursor_scale
            self._cursor["frameSize"] = (-extent, extent, -extent, extent)
            self._cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)
        if self._preview_cursor is not None:
            preview_extent = 22.0 * self._ui_settings.cursor_scale
            self._preview_cursor["frameSize"] = (-preview_extent, preview_extent, -preview_extent, preview_extent)
            self._preview_cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)

    def _update_mapping_preview(self) -> None:
        if self._mapping_panel is None or self._mapping_dot is None or self._mapping_crosshair_h is None or self._mapping_crosshair_v is None:
            return
        frame_size = self._mapping_panel["frameSize"]
        panel_width = float(frame_size[1]) - float(frame_size[0])
        panel_height = abs(float(frame_size[2]) - float(frame_size[3]))
        cursor_norm = self._last_cursor_state.cursor_norm if self._last_cursor_state.visible else (0.5, 0.5)
        inset_left = 26.0
        inset_top = 72.0
        inset_right = 26.0
        inset_bottom = 26.0
        preview_width = max(panel_width - inset_left - inset_right, 40.0)
        preview_height = max(panel_height - inset_top - inset_bottom, 40.0)
        center_x = inset_left + preview_width * cursor_norm[0]
        center_y = -(inset_top + preview_height * cursor_norm[1])
        self._mapping_dot.setPos(center_x, 0, center_y)
        self._mapping_crosshair_h.setPos(center_x, 0, center_y)
        self._mapping_crosshair_v.setPos(center_x, 0, center_y)

    def _slider_value_from_progress(self, key: str, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if key == "cursor_scale":
            value = self._ui_settings.CURSOR_SCALE_MIN + progress * (
                self._ui_settings.CURSOR_SCALE_MAX - self._ui_settings.CURSOR_SCALE_MIN
            )
            return round(round(value / self._ui_settings.CURSOR_SCALE_STEP) * self._ui_settings.CURSOR_SCALE_STEP, 2)
        if key == "cursor_opacity":
            value = self._ui_settings.CURSOR_OPACITY_MIN + progress * (
                self._ui_settings.CURSOR_OPACITY_MAX - self._ui_settings.CURSOR_OPACITY_MIN
            )
            return round(round(value / self._ui_settings.CURSOR_OPACITY_STEP) * self._ui_settings.CURSOR_OPACITY_STEP, 2)
        if key == "brightness":
            value = self._ui_settings.BRIGHTNESS_MIN + progress * (
                self._ui_settings.BRIGHTNESS_MAX - self._ui_settings.BRIGHTNESS_MIN
            )
            return round(value)
        if key == "volume":
            value = self._ui_settings.VOLUME_MIN + progress * (
                self._ui_settings.VOLUME_MAX - self._ui_settings.VOLUME_MIN
            )
            return round(value)
        return 0.0

    def _set_slider_value(self, key: str, value: float) -> None:
        if key == "cursor_scale":
            next_value = self._ui_settings.set_cursor_scale(value)
        elif key == "cursor_opacity":
            next_value = self._ui_settings.set_cursor_opacity(value)
        elif key == "brightness":
            next_value = self._ui_settings.set_brightness(value)
        elif key == "volume":
            next_value = self._ui_settings.set_volume(value)
        else:
            return

        self._refresh_setting_values()
        self._apply_cursor_style()
        if self._on_button_activated is not None:
            self._on_button_activated(f"set_{key}:{next_value}")

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
        is_dragging = pinch_state in {"pinched", "release_candidate"}

        if self._active_slider_key is None and pinch_state == "pinched" and hovered_slider is not None:
            self._active_slider_key = hovered_slider

        if self._active_slider_key is None:
            return

        if not is_dragging or not state.visible:
            self._active_slider_key = None
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
        self._refresh_setting_values()
        self._apply_cursor_style()

    def update_layout(self, force: bool = False) -> None:
        if self._root is None or self._title is None or self._subtitle is None:
            return
        width, height = self._window_size_provider()
        width = max(int(width), 800)
        height = max(int(height), 450)
        next_size = (width, height)
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
        content_top = max(int(height * 0.17), 90)
        content_width = width - content_left * 2
        content_height = height - content_top - max(int(height * 0.08), 36)
        right_panel_width = max(int(content_width * 0.31), 250)
        gutter = max(int(width * 0.025), 22)
        left_panel_width = content_width - right_panel_width - gutter
        preview_panel_width = max(int(left_panel_width * 0.30), 170)
        left_controls_width = left_panel_width - preview_panel_width - gutter
        row_gap = max(int(height * 0.035), 20)
        row_height = max(int(height * 0.085), 58)
        toggle_height = max(int(height * 0.08), 56)

        back_width = max(int(width * 0.10), 118)
        back_height = max(int(height * 0.07), 52)
        self._layout_button("back_home", content_left, max(int(height * 0.07), 34), back_width, back_height)

        toggle_width = max(int((left_panel_width - gutter) * 0.5), 150)
        toggle_top = content_top
        self._layout_button("data_panel_toggle", content_left, toggle_top, toggle_width, toggle_height)
        self._layout_button("cam_preview_toggle", content_left + toggle_width + gutter, toggle_top, toggle_width, toggle_height)

        preview_top = toggle_top + toggle_height + row_gap
        preview_height = row_height * 2 + row_gap
        if self._preview_panel is not None and self._preview_panel_title is not None and self._preview_cursor is not None:
            self._preview_panel.setPos(content_left, 0, -preview_top)
            self._preview_panel["frameSize"] = (0, preview_panel_width, -preview_height, 0)
            self._preview_panel_title.setPos(preview_panel_width * 0.5, 0, -28)
            self._preview_panel_title.setScale(max(short_edge * 0.018, 14))
            self._preview_cursor.setPos(preview_panel_width * 0.5, 0, -(preview_height * 0.58))

        slider_left = content_left + preview_panel_width + gutter
        slider_label_scale = max(short_edge * 0.020, 15)
        slider_value_scale = max(short_edge * 0.022, 16)
        slider_track_height = max(int(row_height * 0.16), 10)
        slider_track_width = max(int(left_controls_width - 110), 180)
        slider_value_x = slider_left + left_controls_width

        row_positions = {
            "cursor_scale": preview_top,
            "cursor_opacity": preview_top + row_height + row_gap,
            "brightness": preview_top + preview_height + row_gap,
            "volume": preview_top + preview_height + row_gap + row_height + row_gap,
        }
        for key in self.SLIDER_KEYS:
            top = row_positions[key]
            label = self._row_labels[key]
            value = self._value_nodes[key]
            label.setPos(slider_left, 0, -(top + 18))
            label.setScale(slider_label_scale)
            value.setPos(slider_value_x, 0, -(top + 18))
            value.setScale(slider_value_scale)

            track_left = slider_left
            slider_controls_top = top + row_height - max(int(row_height * 0.34), 18)
            track = self._slider_tracks[key]
            track.setPos(track_left, 0, -(slider_controls_top + slider_track_height * 0.5))
            track["frameSize"] = (0, slider_track_width, -slider_track_height, 0)
            self._slider_fills[key]["frameSize"] = (0, slider_track_width * self._slider_progress(key), -slider_track_height, 0)
            self._slider_bounds[key] = UIButtonBounds(
                left=track_left,
                top=slider_controls_top,
                right=track_left + slider_track_width,
                bottom=slider_controls_top + slider_track_height,
            )

        mapping_left = content_left + left_panel_width + gutter
        mapping_top = toggle_top
        mapping_height = max(content_height, preview_height + row_height * 2 + row_gap * 3)
        if self._mapping_panel is not None and self._mapping_title is not None and self._mapping_hint is not None:
            self._mapping_panel.setPos(mapping_left, 0, -mapping_top)
            self._mapping_panel["frameSize"] = (0, right_panel_width, -mapping_height, 0)
            self._mapping_title.setPos(24, 0, -28)
            self._mapping_title.setScale(max(short_edge * 0.020, 15))
            self._mapping_hint.setPos(24, 0, -54)
            self._mapping_hint.setScale(max(short_edge * 0.015, 12))

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
        bounds = UIButtonBounds(left=left, top=top, right=left + width_px, bottom=top + height_px)
        if button_index < len(self._button_bounds):
            self._button_bounds[button_index] = bounds
        else:
            self._button_bounds.append(bounds)

    def update_cursor(self, state: UIInputState, pinch_state: PinchState | None = None) -> None:
        if self._cursor is None:
            return

        self._last_cursor_state = state
        self._update_slider_interaction(state, pinch_state)
        snapshot = self._interaction_controller.update(
            state if self._visible else UIInputState(visible=False),
            pinch_state=pinch_state,
            button_bounds=self._button_bounds,
        )
        self._update_button_visuals(snapshot)
        if snapshot.activated_index is not None and self._on_button_activated is not None:
            self._on_button_activated(self._button_actions[snapshot.activated_index])

        self._update_mapping_preview()

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
        self._active_slider_key = None
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
        if self._preview_cursor is not None:
            self._preview_cursor.destroy()
            self._preview_cursor = None
        if self._preview_panel_title is not None:
            self._preview_panel_title.removeNode()
            self._preview_panel_title = None
        if self._preview_panel is not None:
            self._preview_panel.destroy()
            self._preview_panel = None
        if self._mapping_crosshair_h is not None:
            self._mapping_crosshair_h.destroy()
            self._mapping_crosshair_h = None
        if self._mapping_crosshair_v is not None:
            self._mapping_crosshair_v.destroy()
            self._mapping_crosshair_v = None
        if self._mapping_dot is not None:
            self._mapping_dot.destroy()
            self._mapping_dot = None
        if self._mapping_hint is not None:
            self._mapping_hint.removeNode()
            self._mapping_hint = None
        if self._mapping_title is not None:
            self._mapping_title.removeNode()
            self._mapping_title = None
        if self._mapping_panel is not None:
            self._mapping_panel.destroy()
            self._mapping_panel = None
        if self._cursor is not None:
            self._cursor.destroy()
            self._cursor = None
        for button in self._buttons:
            button.destroy()
        self._buttons.clear()
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
        logger.info("Setting UI cleaned up")
