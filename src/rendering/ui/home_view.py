from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode
from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .state import UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.home_view")


class HomeUIView:
    TITLE_TEXT = "aerointeract3d"
    BUTTON_LABELS = ("table", "setting")
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
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._interaction_controller = UIButtonInteractionController()
        self._cursor: Optional[DirectFrame] = None
        self._visible = True
        self._last_layout_size: tuple[int, int] | None = None
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
        try:
            self._root = DirectFrame(
                parent=self._pixel2d,
                pos=(0, 0, 0),
                frameSize=(0, 1, -1, 0),
                frameColor=(0.94, 0.93, 0.90, 1.0),
            )

            self._title = self._create_text_node(
                self._root,
                node_name="home_title",
                text=self.TITLE_TEXT,
                align=TextNode.ARight,
                color=(0.10, 0.13, 0.18, 1.0),
            )

            for index, label in enumerate(self.BUTTON_LABELS):
                button = DirectFrame(
                    parent=self._root,
                    pos=(0, 0, 0),
                    frameSize=(0, 1, -1, 0),
                    frameColor=(0.18, 0.22, 0.28, 1.0),
                    relief=1,
                    borderWidth=(1, 1),
                )
                button_label = self._create_text_node(
                    button,
                    node_name=f"home_button_label_{index}",
                    text=label,
                    align=TextNode.ACenter,
                    color=(0.96, 0.95, 0.92, 1.0),
                )
                self._buttons.append(button)
                self._button_labels.append(button_label)
                self._button_visual_states.append("idle")

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
            self._apply_cursor_style()
            self.update_layout(force=True)
            logger.info("Home UI initialized successfully")
        except Exception:
            logger.exception("Failed to initialize home UI")
            raise

    def _apply_cursor_style(self) -> None:
        if self._cursor is None:
            return
        extent = 18.0 * self._ui_settings.cursor_scale
        self._cursor["frameSize"] = (-extent, extent, -extent, extent)
        self._cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)

    def _apply_brightness(self) -> None:
        brightness = self._ui_settings.brightness_scale
        targets = (self._root, self._overlay_root, self._title, *self._buttons, *self._button_labels)
        for target in targets:
            if target is None:
                continue
            set_color_scale = getattr(target, "setColorScale", None)
            if callable(set_color_scale):
                set_color_scale(brightness, brightness, brightness, 1.0)

    def set_ui_settings(self, settings: UISettingsState) -> None:
        self._ui_settings.data_panel_enabled = settings.data_panel_enabled
        self._ui_settings.cam_preview_enabled = settings.cam_preview_enabled
        self._ui_settings.cursor_scale = settings.cursor_scale
        self._ui_settings.cursor_opacity = settings.cursor_opacity
        self._ui_settings.brightness = settings.brightness
        self._ui_settings.volume = settings.volume
        self._apply_cursor_style()
        self._apply_brightness()

    def update_layout(self, force: bool = False) -> None:
        if self._root is None or self._title is None:
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
        title_margin_top = max(int(height * 0.08), 36)
        title_scale = max(short_edge * 0.045, 24)
        self._title.setPos(width - title_margin_right, 0, -(title_margin_top + title_scale * 0.45))
        self._title.setScale(title_scale)

        button_width = max(int(width * 0.14), 160)
        button_height = max(int(height * 0.09), 60)
        button_gap = max(int(height * 0.025), 18)
        button_margin_left = max(int(width * 0.045), 36)
        button_margin_bottom = max(int(height * 0.08), 36)
        first_button_top = -(height - button_margin_bottom - (button_height * 2 + button_gap))

        for index, button in enumerate(self._buttons):
            button_top = first_button_top - index * (button_height + button_gap)
            button.setPos(button_margin_left, 0, button_top)
            button["frameSize"] = (0, button_width, -button_height, 0)
            button_top_px = -button_top
            if index < len(self._button_bounds):
                self._button_bounds[index] = UIButtonBounds(
                    left=button_margin_left,
                    top=button_top_px,
                    right=button_margin_left + button_width,
                    bottom=button_top_px + button_height,
                )
            else:
                self._button_bounds.append(
                    UIButtonBounds(
                        left=button_margin_left,
                        top=button_top_px,
                        right=button_margin_left + button_width,
                        bottom=button_top_px + button_height,
                    )
                )

        label_scale = max(button_height * 0.32, 18)
        for index, button_label in enumerate(self._button_labels):
            button_label.setPos(button_width * 0.5, 0, -(button_height * 0.60))
            button_label.setScale(label_scale)
            self._apply_button_visual_state(index, self._button_visual_states[index])

    def _apply_button_visual_state(self, index: int, state_name: str) -> None:
        button = self._buttons[index]
        button_label = self._button_labels[index]
        style = self.BUTTON_STYLES[state_name]
        button["frameColor"] = style["frameColor"]
        button_label.node().setTextColor(*style["textColor"])
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

    def _handle_activated_button(self, index: int) -> None:
        if self._on_button_activated is None:
            return
        self._on_button_activated(self.BUTTON_LABELS[index])

    def update_cursor(self, state: UIInputState, pinch_state: PinchState | None = None) -> None:
        if self._cursor is None:
            return

        snapshot = self._interaction_controller.update(
            state if self._visible else UIInputState(visible=False),
            pinch_state=pinch_state,
            button_bounds=self._button_bounds,
        )
        self._update_button_visuals(snapshot)
        if snapshot.activated_index is not None:
            self._handle_activated_button(snapshot.activated_index)

        if not self._visible or not state.visible:
            self._cursor.hide()
            return

        self._cursor.show()
        cursor_x = float(state.cursor_pixels[0])
        cursor_y = -float(state.cursor_pixels[1])
        self._cursor.setPos(cursor_x, 0, cursor_y)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._root is None:
            return
        if visible:
            self.update_layout()
        self._root.show() if visible else self._root.hide()
        if not visible:
            self._interaction_controller.reset()
            for index in range(len(self._buttons)):
                if self._button_visual_states[index] != "idle":
                    self._apply_button_visual_state(index, "idle")
            if self._cursor:
                self._cursor.hide()

    def is_visible(self) -> bool:
        return self._visible

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

        if self._title is not None:
            self._title.removeNode()
            self._title = None

        if self._root is not None:
            self._root.destroy()
            self._root = None
        self._overlay_root = None

        logger.info("Home UI cleaned up")