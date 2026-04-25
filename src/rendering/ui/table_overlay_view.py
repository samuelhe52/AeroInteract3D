from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode

from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .display_metrics import apply_root_display_scale
from .state import TableOverlay, UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.table_overlay_view")


@dataclass(slots=True, frozen=True)
class TableOverlayButtonSpec:
    action: str
    label: str


class TableOverlayUIView:
    MAX_BUTTON_COUNT = 4
    MAX_OBJECT_BUTTON_COUNT = 12
    MENU_BUTTONS = (
        TableOverlayButtonSpec("return_to_table", "resume table"),
        TableOverlayButtonSpec("open_option", "table options"),
        TableOverlayButtonSpec("back_home", "return home"),
    )
    BUTTON_STYLES = {
        "idle": {
            "frameColor": (0.18, 0.22, 0.28, 0.96),
            "textColor": (0.96, 0.95, 0.92, 1.0),
        },
        "hover": {
            "frameColor": (0.26, 0.32, 0.40, 0.98),
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
            "labelColor": (0.96, 0.95, 0.92, 1.0),
            "valueColor": (0.96, 0.95, 0.92, 1.0),
            "knobHalfWidth": 12.0,
            "knobHalfHeight": 16.0,
        },
        "hover": {
            "trackColor": (0.62, 0.66, 0.69, 1.0),
            "fillColor": (0.88, 0.42, 0.30, 1.0),
            "knobColor": (0.24, 0.31, 0.39, 1.0),
            "labelColor": (1.0, 0.98, 0.94, 1.0),
            "valueColor": (1.0, 0.98, 0.94, 1.0),
            "knobHalfWidth": 13.0,
            "knobHalfHeight": 17.0,
        },
        "active": {
            "trackColor": (0.50, 0.55, 0.58, 1.0),
            "fillColor": (0.93, 0.28, 0.22, 1.0),
            "knobColor": (0.86, 0.25, 0.21, 1.0),
            "labelColor": (1.0, 0.98, 0.94, 1.0),
            "valueColor": (0.93, 0.44, 0.39, 1.0),
            "knobHalfWidth": 15.0,
            "knobHalfHeight": 19.0,
        },
    }
    SLIDER_KEYS = ("brightness", "volume")

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
        self._mask: Optional[DirectFrame] = None
        self._panel: Optional[DirectFrame] = None
        self._right_panel: Optional[DirectFrame] = None
        self._title: Optional[NodePath] = None
        self._subtitle: Optional[NodePath] = None
        self._note: Optional[NodePath] = None
        self._right_title: Optional[NodePath] = None
        self._right_note: Optional[NodePath] = None
        self._object_buttons: list[DirectFrame] = []
        self._object_button_labels: list[NodePath] = []
        self._object_button_bounds: list[UIButtonBounds] = []
        self._object_button_visual_states: list[str] = []
        self._active_object_specs: tuple[TableOverlayButtonSpec, ...] = ()
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._active_button_specs: tuple[TableOverlayButtonSpec, ...] = ()
        self._slider_tracks: dict[str, DirectFrame] = {}
        self._slider_fills: dict[str, DirectFrame] = {}
        self._slider_knobs: dict[str, DirectFrame] = {}
        self._slider_bounds: dict[str, UIButtonBounds] = {}
        self._slider_visual_states: dict[str, str] = {}
        self._slider_labels: dict[str, NodePath] = {}
        self._slider_values: dict[str, NodePath] = {}
        self._object_button_bounds: list[UIButtonBounds] = []
        self._interaction_controller = UIButtonInteractionController()
        self._object_interaction_controller = UIButtonInteractionController()
        self._cursor: Optional[DirectFrame] = None
        self._visible = False
        self._last_layout_size: tuple[int, int, float] | None = None
        self._active_overlay = TableOverlay.NONE
        self._ui_settings = UISettingsState()
        self._object_items: list[dict[str, object]] = []
        self._hover_slider_key: str | None = None
        self._active_slider_key: str | None = None
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

    def init_view(self) -> None:
        self._root = DirectFrame(
            parent=self._pixel2d,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.02, 0.02, 0.03, 0.40),
            relief=1,
            sortOrder=120,
        )
        self._mask = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.0, 0.0, 0.0, 0.0),
            relief=None,
        )
        self._panel = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.11, 0.13, 0.16, 0.90),
            relief=1,
            borderWidth=(2, 2),
        )
        self._right_panel = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(0, 1, -1, 0),
            frameColor=(0.11, 0.13, 0.16, 0.90),
            relief=1,
            borderWidth=(2, 2),
        )

        self._title = self._create_text_node(
            self._panel,
            node_name="table_overlay_title",
            text="table menu",
            align=TextNode.ACenter,
            color=(0.96, 0.95, 0.92, 1.0),
        )
        self._subtitle = self._create_text_node(
            self._panel,
            node_name="table_overlay_subtitle",
            text="scene paused",
            align=TextNode.ACenter,
            color=(0.79, 0.81, 0.83, 1.0),
        )
        self._note = self._create_text_node(
            self._panel,
            node_name="table_overlay_note",
            text="",
            align=TextNode.ACenter,
            color=(0.79, 0.81, 0.83, 1.0),
        )
        self._right_title = self._create_text_node(
            self._right_panel,
            node_name="table_overlay_right_title",
            text="objects",
            align=TextNode.ALeft,
            color=(0.96, 0.95, 0.92, 1.0),
        )
        self._right_note = self._create_text_node(
            self._right_panel,
            node_name="table_overlay_right_note",
            text="object visibility controls land in module 12",
            align=TextNode.ALeft,
            color=(0.79, 0.81, 0.83, 1.0),
        )

        for index in range(self.MAX_OBJECT_BUTTON_COUNT):
            button = DirectFrame(
                parent=self._right_panel,
                pos=(0, 0, 0),
                frameSize=(0, 1, -1, 0),
                frameColor=self.BUTTON_STYLES["idle"]["frameColor"],
                relief=1,
                borderWidth=(1, 1),
            )
            button_label = self._create_text_node(
                button,
                node_name=f"table_overlay_object_button_label_{index}",
                text="",
                align=TextNode.ALeft,
                color=self.BUTTON_STYLES["idle"]["textColor"],
            )
            self._object_buttons.append(button)
            self._object_button_labels.append(button_label)
            self._object_button_visual_states.append("idle")

        for index in range(self.MAX_BUTTON_COUNT):
            button = DirectFrame(
                parent=self._panel,
                pos=(0, 0, 0),
                frameSize=(0, 1, -1, 0),
                frameColor=self.BUTTON_STYLES["idle"]["frameColor"],
                relief=1,
                borderWidth=(1, 1),
            )
            button_label = self._create_text_node(
                button,
                node_name=f"table_overlay_button_label_{index}",
                text="",
                align=TextNode.ACenter,
                color=self.BUTTON_STYLES["idle"]["textColor"],
            )
            self._buttons.append(button)
            self._button_labels.append(button_label)
            self._button_visual_states.append("idle")

        for key, label in (("brightness", "brightness"), ("volume", "volume")):
            self._slider_labels[key] = self._create_text_node(
                self._panel,
                node_name=f"table_overlay_slider_label_{key}",
                text=label,
                align=TextNode.ALeft,
                color=self.SLIDER_STYLES["idle"]["labelColor"],
            )
            self._slider_values[key] = self._create_text_node(
                self._panel,
                node_name=f"table_overlay_slider_value_{key}",
                text="",
                align=TextNode.ARight,
                color=self.SLIDER_STYLES["idle"]["valueColor"],
            )
            track = DirectFrame(
                parent=self._panel,
                pos=(0, 0, 0),
                frameSize=(0, 1, -1, 0),
                frameColor=self.SLIDER_STYLES["idle"]["trackColor"],
                relief=1,
                borderWidth=(1, 1),
            )
            fill = DirectFrame(
                parent=track,
                pos=(0, 0, 0),
                frameSize=(0, 1, -1, 0),
                frameColor=self.SLIDER_STYLES["idle"]["fillColor"],
                relief=0,
            )
            knob = DirectFrame(
                parent=track,
                pos=(0, 0, 0),
                frameSize=(-12, 12, -16, 16),
                frameColor=self.SLIDER_STYLES["idle"]["knobColor"],
                relief=1,
                borderWidth=(1, 1),
            )
            self._slider_tracks[key] = track
            self._slider_fills[key] = fill
            self._slider_knobs[key] = knob
            self._slider_visual_states[key] = "idle"

        self._cursor = DirectFrame(
            parent=self._root,
            pos=(0, 0, 0),
            frameSize=(-18, 18, -18, 18),
            frameColor=(0.94, 0.10, 0.10, 0.92),
            relief=1,
            borderWidth=(2, 2),
            sortOrder=130,
        )
        self._cursor.hide()
        self._apply_cursor_style()
        self.set_overlay(TableOverlay.NONE)
        self.update_layout(force=True)
        self._root.hide()
        logger.info("Table overlay UI initialized successfully")

    def _apply_cursor_style(self) -> None:
        if self._cursor is None:
            return
        extent = 18.0 * self._ui_settings.cursor_scale
        self._cursor["frameSize"] = (-extent, extent, -extent, extent)
        self._cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)

    def set_ui_settings(self, settings: UISettingsState) -> None:
        self._ui_settings.data_panel_enabled = settings.data_panel_enabled
        self._ui_settings.cam_preview_enabled = settings.cam_preview_enabled
        self._ui_settings.cursor_scale = settings.cursor_scale
        self._ui_settings.cursor_opacity = settings.cursor_opacity
        self._ui_settings.brightness = settings.brightness
        self._ui_settings.volume = settings.volume
        self._apply_cursor_style()
        if self._active_overlay != TableOverlay.NONE:
            self._refresh_overlay_content()

    def set_object_visibility_items(self, items: list[dict[str, object]]) -> None:
        self._object_items = [dict(item) for item in items[: self.MAX_OBJECT_BUTTON_COUNT]]
        if self._active_overlay == TableOverlay.OPTION:
            self._refresh_overlay_content()
            self.update_layout(force=True)

    def _button_specs_for_overlay(self, overlay: TableOverlay) -> tuple[TableOverlayButtonSpec, ...]:
        if overlay == TableOverlay.MENU:
            return self.MENU_BUTTONS
        if overlay == TableOverlay.OPTION:
            return (
                TableOverlayButtonSpec("back_to_menu", "back to menu"),
                TableOverlayButtonSpec("return_to_table", "return to table"),
                TableOverlayButtonSpec(
                    "toggle_data_panel",
                    f"data panel: {'on' if self._ui_settings.data_panel_enabled else 'off'}",
                ),
                TableOverlayButtonSpec(
                    "toggle_cam_preview",
                    f"cam preview: {'on' if self._ui_settings.cam_preview_enabled else 'off'}",
                ),
            )
        return ()

    def _slider_progress(self, key: str) -> float:
        if key == "brightness":
            return (self._ui_settings.brightness - self._ui_settings.BRIGHTNESS_MIN) / (
                self._ui_settings.BRIGHTNESS_MAX - self._ui_settings.BRIGHTNESS_MIN
            )
        if key == "volume":
            return (self._ui_settings.volume - self._ui_settings.VOLUME_MIN) / (
                self._ui_settings.VOLUME_MAX - self._ui_settings.VOLUME_MIN
            )
        return 0.0

    def _refresh_slider_values(self) -> None:
        self._slider_values["brightness"].node().setText(f"{int(round(self._ui_settings.brightness))}")
        self._slider_values["volume"].node().setText(f"{int(round(self._ui_settings.volume))}")
        for key, fill in self._slider_fills.items():
            track = self._slider_tracks[key]
            knob = self._slider_knobs[key]
            frame_size = track["frameSize"]
            width = max(float(frame_size[1]) - float(frame_size[0]), 1.0)
            progress = self._slider_progress(key)
            fill["frameSize"] = (0, width * progress, float(frame_size[2]), float(frame_size[3]))
            knob.setPos(width * progress, 0, (float(frame_size[2]) + float(frame_size[3])) * 0.5)
        self._refresh_slider_visuals()

    def _apply_slider_visual_state(self, key: str, state_name: str) -> None:
        if self._slider_visual_states.get(key) == state_name:
            return
        style = self.SLIDER_STYLES[state_name]
        track = self._slider_tracks[key]
        fill = self._slider_fills[key]
        knob = self._slider_knobs[key]
        label = self._slider_labels[key]
        value = self._slider_values[key]
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
        for key in self.SLIDER_KEYS:
            next_state = "idle"
            if self._hover_slider_key == key:
                next_state = "hover"
            if self._active_slider_key == key:
                next_state = "active"
            self._apply_slider_visual_state(key, next_state)

    def _set_option_slider_visible(self, visible: bool) -> None:
        for key in self.SLIDER_KEYS:
            targets = (
                self._slider_labels[key],
                self._slider_values[key],
                self._slider_tracks[key],
            )
            for target in targets:
                target.show() if visible else target.hide()

    def _object_button_specs(self) -> tuple[TableOverlayButtonSpec, ...]:
        specs: list[TableOverlayButtonSpec] = []
        for item in self._object_items:
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            label = str(item.get("label", object_id)).strip() or object_id
            visible = bool(item.get("visible", True))
            checkbox = "[x]" if visible else "[ ]"
            specs.append(TableOverlayButtonSpec(f"toggle_object_visibility:{object_id}", f"{checkbox} {label}"))
        return tuple(specs)

    def _refresh_overlay_content(self) -> None:
        self._active_button_specs = self._button_specs_for_overlay(self._active_overlay)
        option_active = self._active_overlay == TableOverlay.OPTION
        self._right_panel.show() if option_active else self._right_panel.hide()
        if self._right_title is not None:
            self._right_title.show() if option_active else self._right_title.hide()
        if self._right_note is not None:
            self._right_note.show() if option_active else self._right_note.hide()
        self._set_option_slider_visible(option_active)
        self._active_object_specs = self._object_button_specs() if option_active else ()

        if self._active_overlay == TableOverlay.MENU:
            self._title.node().setText("pause menu")
            self._subtitle.node().setText("")
            self._note.node().setText("")
        elif option_active:
            self._title.node().setText("table options")
            self._subtitle.node().setText("shared live scene controls")
            self._note.node().setText("")
            self._right_title.node().setText("object visibility")
            self._right_note.node().setText("")
            self._refresh_slider_values()
        else:
            self._title.node().setText("pause menu")
            self._subtitle.node().setText("")
            self._note.node().setText("")

        for index, button in enumerate(self._buttons):
            if index < len(self._active_button_specs):
                button.show()
                self._button_labels[index].node().setText(self._active_button_specs[index].label)
                self._apply_button_visual_state(index, "idle")
            else:
                button.hide()
                self._button_labels[index].node().setText("")

        for index, button in enumerate(self._object_buttons):
            if index < len(self._active_object_specs):
                button.show()
                self._object_button_labels[index].node().setText(self._active_object_specs[index].label)
                self._apply_object_button_visual_state(index, "idle")
            else:
                button.hide()
                self._object_button_labels[index].node().setText("")

    def set_overlay(self, overlay: TableOverlay | str) -> None:
        self._active_overlay = TableOverlay(overlay)
        self._interaction_controller.reset()
        self._object_interaction_controller.reset()
        self._hover_slider_key = None
        self._active_slider_key = None
        self._refresh_overlay_content()
        self.update_layout(force=True)

    def update_layout(self, force: bool = False) -> None:
        if self._root is None or self._mask is None or self._panel is None or self._title is None or self._subtitle is None or self._note is None:
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
        self._mask["frameSize"] = (0, width, -height, 0)

        if self._active_overlay == TableOverlay.OPTION:
            side_width = max(int(width * 0.176), 220)
            side_width = min(side_width, max(int(width * 0.22), 220))
            panel_width = side_width
            panel_height = height
            panel_left = 0
            panel_top = 0
            right_left = width - side_width
            self._right_panel.setPos(right_left, 0, 0)
            self._right_panel["frameSize"] = (0, side_width, -height, 0)
            self._right_title.setPos(side_width * 0.10, 0, -(max(min(width, height) * 0.038, 22) * 1.1))
            self._right_title.setScale(max(min(width, height) * 0.038, 22))
            self._right_note.setPos(side_width * 0.10, 0, -(max(min(width, height) * 0.038, 22) * 2.2))
            self._right_note.setScale(max(max(min(width, height) * 0.038, 22) * 0.34, 11))
        else:
            panel_width = max(int(width * 0.20), 240)
            panel_height = max(int(height * 0.34), 240)
            panel_left = int((width - panel_width) * 0.5)
            panel_top = -max(int(height * 0.18), 72)
            self._right_panel.hide()
            self._object_button_bounds = []

        self._panel.setPos(panel_left, 0, panel_top)
        self._panel["frameSize"] = (0, panel_width, -panel_height, 0)

        title_scale = max(min(width, height) * 0.038, 22)
        subtitle_scale = max(title_scale * 0.38, 12)
        note_scale = max(subtitle_scale * 0.86, 10)
        option_active = self._active_overlay == TableOverlay.OPTION
        title_x = panel_width * (0.10 if option_active else 0.5)
        title_align = TextNode.ALeft if option_active else TextNode.ACenter
        self._title.node().setAlign(title_align)
        self._subtitle.node().setAlign(title_align)
        self._note.node().setAlign(title_align)
        self._title.setPos(title_x, 0, -(title_scale * 1.1))
        self._title.setScale(title_scale)
        self._subtitle.setPos(title_x, 0, -(title_scale * 2.05))
        self._subtitle.setScale(subtitle_scale)
        self._note.setPos(title_x, 0, -(title_scale * 3.05))
        self._note.setScale(note_scale)

        button_count = len(self._active_button_specs)
        self._button_bounds = []
        if button_count > 0:
            button_width = int(panel_width * 0.82)
            button_height = max(42, int((height if option_active else panel_height) * (0.055 if option_active else 0.16)))
            button_gap = max(12, int((height if option_active else panel_height) * (0.018 if option_active else 0.07)))
            buttons_left = int((panel_width - button_width) * 0.5)
            if option_active:
                buttons_top = -(panel_height * 0.16)
            else:
                buttons_block_height = button_count * button_height + max(button_count - 1, 0) * button_gap
                buttons_top = -(panel_height * 0.48 - buttons_block_height * 0.5)
            label_scale = max(button_height * 0.28, 16)
            for index in range(button_count):
                button = self._buttons[index]
                button_top = int(buttons_top - index * (button_height + button_gap))
                button.setPos(buttons_left, 0, button_top)
                button["frameSize"] = (0, button_width, -button_height, 0)
                self._button_labels[index].setPos(button_width * 0.5, 0, -(button_height * 0.60))
                self._button_labels[index].setScale(label_scale)
                self._button_bounds.append(
                    UIButtonBounds(
                        left=panel_left + buttons_left,
                        top=(-panel_top) + (-button_top),
                        right=panel_left + buttons_left + button_width,
                        bottom=(-panel_top) + (-button_top) + button_height,
                    )
                )
                self._apply_button_visual_state(index, self._button_visual_states[index])

        if option_active:
            slider_label_scale = max(title_scale * 0.42, 13)
            slider_value_scale = slider_label_scale
            slider_track_width = int(panel_width * 0.74)
            slider_track_height = max(int(panel_height * 0.030), 18)
            slider_left = int(panel_width * 0.13)
            first_slider_top = -(panel_height * 0.52)
            slider_row_gap = max(int(panel_height * 0.16), 72)
            for index, key in enumerate(self.SLIDER_KEYS):
                row_top = int(first_slider_top - index * slider_row_gap)
                self._slider_labels[key].setPos(panel_width * 0.10, 0, row_top)
                self._slider_labels[key].setScale(slider_label_scale)
                self._slider_values[key].setPos(panel_width * 0.90, 0, row_top)
                self._slider_values[key].setScale(slider_value_scale)
                track_top = row_top - max(int(slider_track_height * 1.6), 24)
                track = self._slider_tracks[key]
                track.setPos(slider_left, 0, track_top)
                track["frameSize"] = (0, slider_track_width, -slider_track_height, 0)
                self._slider_bounds[key] = UIButtonBounds(
                    left=panel_left + slider_left,
                    top=(-panel_top) + (-track_top),
                    right=panel_left + slider_left + slider_track_width,
                    bottom=(-panel_top) + (-track_top) + slider_track_height,
                )
            self._object_button_bounds = []
            object_button_width = int(side_width * 0.80)
            object_button_height = max(40, int(panel_height * 0.052))
            object_button_gap = max(10, int(panel_height * 0.016))
            object_left = int(side_width * 0.10)
            object_top_start = -(panel_height * 0.22)
            object_label_scale = max(object_button_height * 0.20, 13)
            for index, spec in enumerate(self._active_object_specs):
                button = self._object_buttons[index]
                button_top = int(object_top_start - index * (object_button_height + object_button_gap))
                button.setPos(object_left, 0, button_top)
                button["frameSize"] = (0, object_button_width, -object_button_height, 0)
                self._object_button_labels[index].node().setAlign(TextNode.ALeft)
                self._object_button_labels[index].setPos(object_button_width * 0.07, 0, -(object_button_height * 0.60))
                self._object_button_labels[index].setScale(object_label_scale)
                self._object_button_bounds.append(
                    UIButtonBounds(
                        left=right_left + object_left,
                        top=-button_top,
                        right=right_left + object_left + object_button_width,
                        bottom=-button_top + object_button_height,
                    )
                )
                self._apply_object_button_visual_state(index, self._object_button_visual_states[index])
            self._refresh_slider_values()

    def _apply_button_visual_state(self, index: int, state_name: str) -> None:
        style = self.BUTTON_STYLES[state_name]
        self._buttons[index]["frameColor"] = style["frameColor"]
        self._button_labels[index].node().setTextColor(*style["textColor"])
        self._button_visual_states[index] = state_name

    def _update_button_visuals(self, snapshot: UIButtonInteractionSnapshot) -> None:
        for index in range(len(self._active_button_specs)):
            next_state = "idle"
            if snapshot.hovered_index == index:
                next_state = "hover"
            if snapshot.hovered_index == index and snapshot.pressed_index == index:
                next_state = "pressed"
            if self._button_visual_states[index] != next_state:
                self._apply_button_visual_state(index, next_state)

    def _apply_object_button_visual_state(self, index: int, state_name: str) -> None:
        style = self.BUTTON_STYLES[state_name]
        self._object_buttons[index]["frameColor"] = style["frameColor"]
        self._object_button_labels[index].node().setTextColor(*style["textColor"])
        self._object_button_visual_states[index] = state_name

    def _update_object_button_visuals(self, snapshot: UIButtonInteractionSnapshot) -> None:
        for index in range(len(self._active_object_specs)):
            next_state = "idle"
            if snapshot.hovered_index == index:
                next_state = "hover"
            if snapshot.hovered_index == index and snapshot.pressed_index == index:
                next_state = "pressed"
            if self._object_button_visual_states[index] != next_state:
                self._apply_object_button_visual_state(index, next_state)

    def _slider_key_at_cursor(self, state: UIInputState) -> str | None:
        if not state.visible:
            return None
        cursor_x, cursor_y = state.cursor_pixels
        for key, bounds in self._slider_bounds.items():
            if bounds.contains(cursor_x, cursor_y):
                return key
        return None

    def _emit_slider_action(self, key: str, value: float) -> None:
        if self._on_button_activated is None:
            return
        self._on_button_activated(f"set_{key}:{value:.2f}")

    def _apply_slider_value_from_cursor(self, key: str, cursor_x: float) -> None:
        bounds = self._slider_bounds.get(key)
        if bounds is None:
            return
        progress = max(0.0, min(1.0, (cursor_x - bounds.left) / max(bounds.right - bounds.left, 1.0)))
        if key == "brightness":
            value = self._ui_settings.BRIGHTNESS_MIN + progress * (
                self._ui_settings.BRIGHTNESS_MAX - self._ui_settings.BRIGHTNESS_MIN
            )
        else:
            value = self._ui_settings.VOLUME_MIN + progress * (
                self._ui_settings.VOLUME_MAX - self._ui_settings.VOLUME_MIN
            )
        self._emit_slider_action(key, round(value, 2))

    def _update_slider_interaction(self, state: UIInputState, pinch_state: PinchState | None) -> bool:
        if self._active_overlay != TableOverlay.OPTION:
            self._hover_slider_key = None
            self._active_slider_key = None
            self._refresh_slider_visuals()
            return False
        hovered_slider = self._slider_key_at_cursor(state)
        self._hover_slider_key = hovered_slider
        is_pressed = pinch_state in {"pinched", "release_candidate"}
        if self._active_slider_key is None and pinch_state == "pinched" and hovered_slider is not None:
            self._active_slider_key = hovered_slider
        if self._active_slider_key is not None and is_pressed and state.visible:
            self._apply_slider_value_from_cursor(self._active_slider_key, float(state.cursor_pixels[0]))
        if self._active_slider_key is not None and not is_pressed:
            self._active_slider_key = None
        self._refresh_slider_visuals()
        return self._active_slider_key is not None or hovered_slider is not None

    def update_cursor(self, state: UIInputState, pinch_state: PinchState | None = None) -> None:
        if self._cursor is None:
            return
        block_buttons = self._update_slider_interaction(state, pinch_state)
        button_bounds = () if block_buttons else self._button_bounds
        snapshot = self._interaction_controller.update(
            state if self._visible else UIInputState(visible=False),
            pinch_state=pinch_state,
            button_bounds=button_bounds,
        )
        self._update_button_visuals(snapshot)
        if snapshot.activated_index is not None and self._on_button_activated is not None:
            self._on_button_activated(self._active_button_specs[snapshot.activated_index].action)
        object_snapshot = self._object_interaction_controller.update(
            state if self._visible and not block_buttons and self._active_overlay == TableOverlay.OPTION else UIInputState(visible=False),
            pinch_state=pinch_state,
            button_bounds=self._object_button_bounds,
        )
        self._update_object_button_visuals(object_snapshot)
        if object_snapshot.activated_index is not None and self._on_button_activated is not None:
            self._on_button_activated(self._active_object_specs[object_snapshot.activated_index].action)
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
            self.update_layout(force=True)
        self._root.show() if visible else self._root.hide()
        if not visible:
            self._interaction_controller.reset()
            self._object_interaction_controller.reset()
            self._hover_slider_key = None
            self._active_slider_key = None
            self._refresh_slider_visuals()
            if self._cursor is not None:
                self._cursor.hide()
            for index in range(len(self._active_button_specs)):
                self._apply_button_visual_state(index, "idle")
            for index in range(len(self._active_object_specs)):
                self._apply_object_button_visual_state(index, "idle")

    def destroy(self) -> None:
        for label in self._button_labels:
            label.removeNode()
        self._button_labels.clear()
        for label in self._object_button_labels:
            label.removeNode()
        self._object_button_labels.clear()
        for key in self.SLIDER_KEYS:
            self._slider_labels[key].removeNode()
            self._slider_values[key].removeNode()
            self._slider_tracks[key].destroy()
        if self._cursor is not None:
            self._cursor.destroy()
            self._cursor = None
        for button in self._buttons:
            button.destroy()
        self._buttons.clear()
        for button in self._object_buttons:
            button.destroy()
        self._object_buttons.clear()
        if self._right_note is not None:
            self._right_note.removeNode()
            self._right_note = None
        if self._right_title is not None:
            self._right_title.removeNode()
            self._right_title = None
        if self._note is not None:
            self._note.removeNode()
            self._note = None
        if self._subtitle is not None:
            self._subtitle.removeNode()
            self._subtitle = None
        if self._title is not None:
            self._title.removeNode()
            self._title = None
        if self._right_panel is not None:
            self._right_panel.destroy()
            self._right_panel = None
        if self._root is not None:
            self._root.destroy()
            self._root = None
        self._mask = None
        self._panel = None
        logger.info("Table overlay UI cleaned up")
