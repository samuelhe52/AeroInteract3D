from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode
from src.contracts import PinchState

from .interaction import UIButtonBounds, UIButtonInteractionController, UIButtonInteractionSnapshot
from .display_metrics import apply_root_display_scale
from .state import UIInputState, UISettingsState


logger = logging.getLogger("rendering.ui.home_view")


class HomeUIView:
    TITLE_TEXT = "Aerointeract3d"
    BUTTON_LABELS = ("table", "setting")
    GESTURE_DEMOS = (
        ("drag", "drag", "primary pinch and drag"),
        ("scale", "scale", "primary+secondary pinch"),
        ("rotate", "rotate", "pinch and turn"),
    )
    CONTROL_OBJECT_COLOR = (0.78, 0.80, 0.82, 1.0)
    PRIMARY_FINGER_COLOR = (0.86, 0.29, 0.23, 1.0)
    SECONDARY_FINGER_COLOR = (0.27, 0.46, 0.74, 1.0)
    FINGER_HALF_SIZE = 4.5
    PINCH_OVERLAP_GAP = 1.8
    HOME_DEMO_CYCLE_SECONDS = 2.6
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
        self._demo_cards: list[dict[str, object]] = []
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._button_bounds: list[UIButtonBounds] = []
        self._button_visual_states: list[str] = []
        self._interaction_controller = UIButtonInteractionController()
        self._cursor: Optional[DirectFrame] = None
        self._visible = True
        self._last_layout_size: tuple[int, int, float] | None = None
        self._ui_settings = UISettingsState()
        self._animation_started_at = time.perf_counter()
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

            for key, label, subtitle in self.GESTURE_DEMOS:
                card = DirectFrame(
                    parent=self._root,
                    pos=(0, 0, 0),
                    frameSize=(0, 1, -1, 0),
                    frameColor=(0.89, 0.87, 0.82, 0.92),
                    relief=1,
                    borderWidth=(1, 1),
                )
                card_title = self._create_text_node(
                    card,
                    node_name=f"home_demo_title_{key}",
                    text=label,
                    align=TextNode.ALeft,
                    color=(0.13, 0.15, 0.20, 1.0),
                )
                card_subtitle = self._create_text_node(
                    card,
                    node_name=f"home_demo_subtitle_{key}",
                    text=subtitle,
                    align=TextNode.ALeft,
                    color=(0.34, 0.37, 0.42, 1.0),
                )
                stage = DirectFrame(
                    parent=card,
                    pos=(0, 0, 0),
                    frameSize=(0, 1, -1, 0),
                    frameColor=(0.95, 0.94, 0.91, 1.0),
                    relief=1,
                    borderWidth=(1, 1),
                )
                rotate_stage = None
                rotate_fingers: list[DirectFrame] = []
                controlled = None
                primary_finger_a = None
                primary_finger_b = None
                secondary_finger_a = None
                secondary_finger_b = None
                if key == "rotate":
                    stage.hide()
                    rotate_stage = DirectFrame(
                        parent=card,
                        pos=(0, 0, 0),
                        frameSize=(0, 1, -1, 0),
                        frameColor=(0.95, 0.94, 0.91, 1.0),
                        relief=1,
                        borderWidth=(1, 1),
                    )
                    rotate_mode_label = self._create_text_node(
                        rotate_stage,
                        node_name=f"home_demo_rotate_mode_{key}",
                        text="1. Pinch five fingers together to switch\n2. Pinch and drag to rotate",
                        align=TextNode.ACenter,
                        color=(0.18, 0.20, 0.24, 1.0),
                    )
                else:
                    rotate_mode_label = None
                if key != "rotate":
                    controlled = DirectFrame(
                        parent=stage,
                        pos=(0, 0, 0),
                        frameSize=(-16, 16, -16, 16),
                        frameColor=self.CONTROL_OBJECT_COLOR,
                        relief=1,
                        borderWidth=(1, 1),
                    )
                    primary_finger_a = DirectFrame(
                        parent=stage,
                        pos=(0, 0, 0),
                        frameSize=(-self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE, -self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE),
                        frameColor=self.PRIMARY_FINGER_COLOR,
                        relief=1,
                        borderWidth=(1, 1),
                    )
                    primary_finger_b = DirectFrame(
                        parent=stage,
                        pos=(0, 0, 0),
                        frameSize=(-self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE, -self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE),
                        frameColor=self.PRIMARY_FINGER_COLOR,
                        relief=1,
                        borderWidth=(1, 1),
                    )
                    secondary_finger_a = DirectFrame(
                        parent=stage,
                        pos=(0, 0, 0),
                        frameSize=(-self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE, -self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE),
                        frameColor=self.SECONDARY_FINGER_COLOR,
                        relief=1,
                        borderWidth=(1, 1),
                    )
                    secondary_finger_b = DirectFrame(
                        parent=stage,
                        pos=(0, 0, 0),
                        frameSize=(-self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE, -self.FINGER_HALF_SIZE, self.FINGER_HALF_SIZE),
                        frameColor=self.SECONDARY_FINGER_COLOR,
                        relief=1,
                        borderWidth=(1, 1),
                    )
                self._demo_cards.append(
                    {
                        "key": key,
                        "card": card,
                        "title": card_title,
                        "subtitle": card_subtitle,
                        "stage": stage,
                        "rotate_stage": rotate_stage,
                        "rotate_fingers": rotate_fingers,
                        "rotate_mode_label": rotate_mode_label,
                        "controlled": controlled,
                        "primary_a": primary_finger_a,
                        "primary_b": primary_finger_b,
                        "secondary_a": secondary_finger_a,
                        "secondary_b": secondary_finger_b,
                        "layout": {},
                    }
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

    @staticmethod
    def compute_layout_metrics(window_size: tuple[int, int]) -> dict[str, float]:
        width = max(int(window_size[0]), 1)
        height = max(int(window_size[1]), 1)
        short_edge = min(width, height)

        title_margin_right = max(int(width * 0.05), 16)
        title_margin_top = max(int(height * 0.08), 16)
        title_scale = max(short_edge * 0.052, 18.0)

        button_margin_left = max(int(width * 0.045), 16)
        button_margin_bottom = max(int(height * 0.08), 16)
        button_gap = max(int(height * 0.025), 10)
        max_button_width = max(width - (button_margin_left * 2), 96)
        button_width = min(max(int(width * 0.22), 120), max_button_width)

        button_count = len(HomeUIView.BUTTON_LABELS)
        top_safe_px = int(title_margin_top + title_scale * 1.9)
        available_height = max(
            height - button_margin_bottom - top_safe_px,
            button_count * 32 + button_gap * (button_count - 1),
        )
        max_button_height = max((available_height - button_gap * (button_count - 1)) // button_count, 32)
        button_height = min(max(int(height * 0.09), 32), max_button_height)
        stack_height = button_height * len(HomeUIView.BUTTON_LABELS) + button_gap * (len(HomeUIView.BUTTON_LABELS) - 1)
        bottom_target_top_px = height - button_margin_bottom - stack_height
        max_top_px = max(height - stack_height, top_safe_px)
        first_button_top_px = min(max(bottom_target_top_px, top_safe_px), max_top_px)

        label_scale = max(button_height * 0.32, 14.0)
        return {
            "width": float(width),
            "height": float(height),
            "title_margin_right": float(title_margin_right),
            "title_margin_top": float(title_margin_top),
            "title_scale": float(title_scale),
            "button_margin_left": float(button_margin_left),
            "button_margin_bottom": float(button_margin_bottom),
            "button_gap": float(button_gap),
            "button_width": float(button_width),
            "button_height": float(button_height),
            "first_button_top_px": float(first_button_top_px),
            "label_scale": float(label_scale),
        }

    def _layout_gesture_demos(self, metrics: dict[str, float]) -> None:
        if not self._demo_cards:
            return

        width = int(metrics["width"])
        height = int(metrics["height"])
        title_top = float(metrics["title_margin_top"] + metrics["title_scale"] * 2.1)
        side_margin = max(int(width * 0.06), 18)
        demo_gap = max(int(width * 0.015), 12)
        available_width = max(width - side_margin * 2, 180)
        desired_card_width = max(int(width * 0.21), 170)
        columns = 3
        if desired_card_width * 3 + demo_gap * 2 > available_width:
            columns = 2 if desired_card_width * 2 + demo_gap <= available_width else 1
        card_width = max((available_width - demo_gap * (columns - 1)) // columns, 150)
        rows = (len(self._demo_cards) + columns - 1) // columns
        card_height = max(int(height * 0.20), 136)
        if rows > 1:
            card_height = max(int(height * 0.17), 124)

        for index, demo in enumerate(self._demo_cards):
            column = index % columns
            row = index // columns
            card_left = side_margin + column * (card_width + demo_gap)
            card_top = int(title_top + row * (card_height + demo_gap))
            card = demo["card"]
            card.setPos(card_left, 0, -card_top)
            card["frameSize"] = (0, card_width, -card_height, 0)

            title = demo["title"]
            subtitle = demo["subtitle"]
            title.setPos(card_width * 0.08, 0, -(card_height * 0.18))
            title.setScale(max(card_height * 0.12, 15))
            subtitle_y_ratio = 0.34
            subtitle.setPos(card_width * 0.08, 0, -(card_height * subtitle_y_ratio))
            subtitle.setScale(max(card_height * 0.07, 10))

            stage = demo["stage"]
            stage_left = int(card_width * 0.08)
            stage_width = int(card_width * 0.84)
            if str(demo["key"]) == "rotate":
                rotate_stage = demo.get("rotate_stage")
                if isinstance(rotate_stage, DirectFrame):
                    rotate_stage_top = int(card_height * 0.48)
                    rotate_stage_height = int(card_height * 0.32)
                    rotate_stage.setPos(stage_left, 0, -rotate_stage_top)
                    rotate_stage["frameSize"] = (0, stage_width, -rotate_stage_height, 0)
                    rotate_stage.show()
                    rotate_mode_label = demo.get("rotate_mode_label")
                    if isinstance(rotate_mode_label, NodePath):
                        rotate_mode_label.setPos(stage_width * 0.50, 0, -(rotate_stage_height * 0.50))
                        rotate_mode_label.setScale(max(rotate_stage_height * 0.16, 11))
                stage.hide()
                stage_top = 0.0
                stage_height = 0.0
            else:
                rotate_stage = demo.get("rotate_stage")
                if isinstance(rotate_stage, DirectFrame):
                    rotate_stage.hide()
                stage.show()
                stage_top = int(card_height * 0.46)
                stage_height = int(card_height * 0.42)

            if str(demo["key"]) != "rotate":
                stage.setPos(stage_left, 0, -stage_top)
                stage["frameSize"] = (0, stage_width, -stage_height, 0)
            demo["layout"] = {
                "stage_width": float(stage_width),
                "stage_height": float(stage_height),
                "controlled_y": float(-stage_height * 0.56),
                "finger_y": float(-stage_height * 0.30),
                "rotate_stage_width": float(stage_width),
                "rotate_stage_height": float(int(card_height * 0.32) if str(demo["key"]) == "rotate" else 0.0),
            }

    def update_animation(self, current_time: float | None = None) -> None:
        if not self._demo_cards:
            return

        animation_time = (time.perf_counter() if current_time is None else float(current_time)) - self._animation_started_at
        for demo in self._demo_cards:
            layout = demo.get("layout")
            if not isinstance(layout, dict) or not layout:
                continue
            stage_width = float(layout["stage_width"])
            controlled_y = float(layout["controlled_y"])
            finger_y = float(layout["finger_y"])
            controlled = demo["controlled"]
            primary_a = demo["primary_a"]
            primary_b = demo["primary_b"]
            secondary_a = demo["secondary_a"]
            secondary_b = demo["secondary_b"]
            key = str(demo["key"])
            if key == "rotate":
                continue

            center_x = stage_width * 0.50
            phase = (animation_time / self.HOME_DEMO_CYCLE_SECONDS) % 1.0
            start_gap = 20.0
            pinch_gap = self.PINCH_OVERLAP_GAP

            def set_pair(left_block, right_block, pair_center_x: float, pair_center_y: float, gap: float) -> None:
                left_block.setPos(pair_center_x - gap * 0.5, 0, pair_center_y)
                right_block.setPos(pair_center_x + gap * 0.5, 0, pair_center_y)

            if key == "drag":
                primary_a.show()
                primary_b.show()
                secondary_a.hide()
                secondary_b.hide()
                if phase < 0.32:
                    ratio = phase / 0.32
                    object_x = center_x
                    pair_center_x = center_x + stage_width * 0.18 - stage_width * 0.18 * ratio
                    gap = start_gap + (pinch_gap - start_gap) * ratio
                elif phase < 0.68:
                    ratio = (phase - 0.32) / 0.36
                    object_x = center_x - ratio * stage_width * 0.22
                    pair_center_x = object_x
                    gap = pinch_gap
                else:
                    ratio = (phase - 0.68) / 0.32
                    object_x = center_x - (1.0 - ratio) * stage_width * 0.22
                    pair_center_x = object_x
                    gap = pinch_gap

                controlled.setPos(object_x, 0, controlled_y)
                controlled.setScale(1.0)
                controlled.setR(0.0)
                set_pair(primary_a, primary_b, pair_center_x, finger_y, gap)
            elif key == "scale":
                primary_a.show()
                primary_b.show()
                secondary_a.show()
                secondary_b.show()
                base_spread = stage_width * 0.16
                active_spread = stage_width * 0.28
                if phase < 0.32:
                    ratio = phase / 0.32
                    hand_spread = base_spread + (stage_width * 0.06 - base_spread) * ratio
                    gap = start_gap + (pinch_gap - start_gap) * ratio
                    object_scale = 1.0
                elif phase < 0.68:
                    ratio = (phase - 0.32) / 0.36
                    hand_spread = stage_width * 0.06 + (active_spread - stage_width * 0.06) * ratio
                    gap = pinch_gap
                    object_scale = 1.0 + ratio * 0.52
                else:
                    ratio = (phase - 0.68) / 0.32
                    hand_spread = active_spread + (base_spread - active_spread) * ratio
                    gap = pinch_gap
                    object_scale = 1.52 + (1.0 - 1.52) * ratio

                controlled.setPos(center_x, 0, controlled_y)
                controlled.setScale(object_scale)
                controlled.setR(0.0)
                set_pair(secondary_a, secondary_b, center_x - hand_spread, finger_y, gap)
                set_pair(primary_a, primary_b, center_x + hand_spread, finger_y, gap)

    def _apply_cursor_style(self) -> None:
        if self._cursor is None:
            return
        extent = 18.0 * self._ui_settings.cursor_scale
        self._cursor["frameSize"] = (-extent, extent, -extent, extent)
        self._cursor["frameColor"] = (0.94, 0.10, 0.10, self._ui_settings.cursor_opacity)

    def _apply_brightness(self) -> None:
        brightness = self._ui_settings.brightness_scale
        demo_nodes: list[object] = []
        for demo in self._demo_cards:
            demo_nodes.extend(
                [
                    demo.get("card"),
                    demo.get("title"),
                    demo.get("subtitle"),
                    demo.get("stage"),
                    demo.get("rotate_stage"),
                    demo.get("rotate_mode_label"),
                    demo.get("controlled"),
                    demo.get("primary_a"),
                    demo.get("primary_b"),
                    demo.get("secondary_a"),
                    demo.get("secondary_b"),
                ]
            )
            rotate_fingers = demo.get("rotate_fingers")
            if isinstance(rotate_fingers, list):
                demo_nodes.extend(rotate_fingers)
        targets = (self._root, self._overlay_root, self._title, *self._buttons, *self._button_labels, *demo_nodes)
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
        metrics = self.compute_layout_metrics((width, height))
        width = int(metrics["width"])
        height = int(metrics["height"])
        display_scale = apply_root_display_scale(self._root, self._display_scale_provider())
        next_size = (width, height, display_scale)
        if not force and next_size == self._last_layout_size:
            return

        self._last_layout_size = next_size
        self._root["frameSize"] = (0, width, -height, 0)
        if self._overlay_root is not None:
            self._overlay_root["frameSize"] = (0, width, -height, 0)

        title_margin_right = int(metrics["title_margin_right"])
        title_margin_top = int(metrics["title_margin_top"])
        title_scale = float(metrics["title_scale"])
        self._title.setPos(width - title_margin_right, 0, -(title_margin_top + title_scale * 0.45))
        self._title.setScale(title_scale)
        self._layout_gesture_demos(metrics)
        self.update_animation()

        button_width = int(metrics["button_width"])
        button_height = int(metrics["button_height"])
        button_gap = int(metrics["button_gap"])
        button_margin_left = int(metrics["button_margin_left"])
        first_button_top = -int(metrics["first_button_top_px"])

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

        label_scale = float(metrics["label_scale"])
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

        if self._visible:
            self.update_animation()

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

        for demo in self._demo_cards:
            title = demo.get("title")
            subtitle = demo.get("subtitle")
            rotate_mode_label = demo.get("rotate_mode_label")
            card = demo.get("card")
            if isinstance(title, NodePath):
                title.removeNode()
            if isinstance(subtitle, NodePath):
                subtitle.removeNode()
            if isinstance(rotate_mode_label, NodePath):
                rotate_mode_label.removeNode()
            if isinstance(card, DirectFrame):
                card.destroy()
        self._demo_cards.clear()

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
