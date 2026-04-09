from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode

from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .state import TableOverlay, UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.table_overlay_view")


@dataclass(slots=True, frozen=True)
class TableOverlayButtonSpec:
    action: str
    label: str


class TableOverlayUIView:
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
    MENU_BUTTONS = (
        TableOverlayButtonSpec("return_to_table", "return to table"),
        TableOverlayButtonSpec("open_option", "option"),
        TableOverlayButtonSpec("back_home", "back to home"),
    )
    OPTION_BUTTONS = (
        TableOverlayButtonSpec("back_to_menu", "back to menu"),
        TableOverlayButtonSpec("return_to_table", "return to table"),
    )

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
        self._mask: Optional[DirectFrame] = None
        self._panel: Optional[DirectFrame] = None
        self._title: Optional[NodePath] = None
        self._subtitle: Optional[NodePath] = None
        self._note: Optional[NodePath] = None
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._active_button_specs: tuple[TableOverlayButtonSpec, ...] = ()
        self._interaction_controller = UIButtonInteractionController()
        self._cursor: Optional[DirectFrame] = None
        self._visible = False
        self._last_layout_size: tuple[int, int] | None = None
        self._active_overlay = TableOverlay.NONE
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

        for index in range(3):
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

    def set_overlay(self, overlay: TableOverlay | str) -> None:
        next_overlay = TableOverlay(overlay)
        self._active_overlay = next_overlay
        self._interaction_controller.reset()

        if next_overlay == TableOverlay.MENU:
            self._active_button_specs = self.MENU_BUTTONS
            self._title.node().setText("table menu")
            self._subtitle.node().setText("scene paused")
            self._note.node().setText("long-hold opens this layer; buttons handle the next step")
        elif next_overlay == TableOverlay.OPTION:
            self._active_button_specs = self.OPTION_BUTTONS
            self._title.node().setText("option")
            self._subtitle.node().setText("layout skeleton is live")
            self._note.node().setText("brightness, volume, panel toggles, and object controls land in the next pass")
        else:
            self._active_button_specs = ()
            self._title.node().setText("table menu")
            self._subtitle.node().setText("scene paused")
            self._note.node().setText("")

        for index, button in enumerate(self._buttons):
            if index < len(self._active_button_specs):
                button.show()
                self._button_labels[index].node().setText(self._active_button_specs[index].label)
                self._apply_button_visual_state(index, "idle")
            else:
                button.hide()
                self._button_labels[index].node().setText("")

        self.update_layout(force=True)

    def update_layout(self, force: bool = False) -> None:
        if self._root is None or self._mask is None or self._panel is None or self._title is None or self._subtitle is None or self._note is None:
            return

        width, height = self._window_size_provider()
        width = max(int(width), 800)
        height = max(int(height), 450)
        next_size = (width, height)
        if not force and next_size == self._last_layout_size:
            return

        self._last_layout_size = next_size
        self._root["frameSize"] = (0, width, -height, 0)
        self._mask["frameSize"] = (0, width, -height, 0)

        if self._active_overlay == TableOverlay.OPTION:
            panel_width = max(int(width * 0.56), 540)
            panel_height = max(int(height * 0.42), 280)
            panel_left = int((width - panel_width) * 0.18)
            panel_top = -max(int(height * 0.10), 42)
        else:
            panel_width = max(int(width * 0.20), 240)
            panel_height = max(int(height * 0.34), 240)
            panel_left = int((width - panel_width) * 0.5)
            panel_top = -max(int(height * 0.18), 72)

        self._panel.setPos(panel_left, 0, panel_top)
        self._panel["frameSize"] = (0, panel_width, -panel_height, 0)

        title_scale = max(min(width, height) * 0.038, 22)
        subtitle_scale = max(title_scale * 0.38, 12)
        note_scale = max(subtitle_scale * 0.86, 10)
        self._title.setPos(panel_width * 0.5, 0, -(title_scale * 1.1))
        self._title.setScale(title_scale)
        self._subtitle.setPos(panel_width * 0.5, 0, -(title_scale * 2.05))
        self._subtitle.setScale(subtitle_scale)
        self._note.setPos(panel_width * 0.5, 0, -(title_scale * 3.05))
        self._note.setScale(note_scale)

        button_count = len(self._active_button_specs)
        self._button_bounds = []
        if button_count == 0:
            return

        button_width = int(panel_width * 0.82)
        button_height = max(int(panel_height * 0.16), 44)
        button_gap = max(int(panel_height * 0.07), 14)
        buttons_block_height = button_count * button_height + max(button_count - 1, 0) * button_gap
        buttons_left = int((panel_width - button_width) * 0.5)
        buttons_top = -(panel_height * 0.48 - buttons_block_height * 0.5)
        label_scale = max(button_height * 0.28, 16)

        for index in range(len(self._active_button_specs)):
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

    def update_cursor(self, state: UIInputState, pinch_state: PinchState | None = None) -> None:
        if self._cursor is None:
            return

        snapshot = self._interaction_controller.update(
            state if self._visible else UIInputState(visible=False),
            pinch_state=pinch_state,
            button_bounds=self._button_bounds,
        )
        self._update_button_visuals(snapshot)
        if snapshot.activated_index is not None and self._on_button_activated is not None:
            self._on_button_activated(self._active_button_specs[snapshot.activated_index].action)

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
            if self._cursor is not None:
                self._cursor.hide()
            for index in range(len(self._active_button_specs)):
                self._apply_button_visual_state(index, "idle")

    def destroy(self) -> None:
        for label in self._button_labels:
            label.removeNode()
        self._button_labels.clear()

        if self._cursor is not None:
            self._cursor.destroy()
            self._cursor = None

        for button in self._buttons:
            button.destroy()
        self._buttons.clear()

        if self._note is not None:
            self._note.removeNode()
            self._note = None
        if self._subtitle is not None:
            self._subtitle.removeNode()
            self._subtitle = None
        if self._title is not None:
            self._title.removeNode()
            self._title = None
        if self._root is not None:
            self._root.destroy()
            self._root = None
        self._mask = None
        self._panel = None

        logger.info("Table overlay UI cleaned up")