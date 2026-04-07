from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from panda3d.core import NodePath, TextNode


logger = logging.getLogger("rendering.ui.home_view")


class HomeUIView:
    TITLE_TEXT = "aerointeract3d"
    BUTTON_LABELS = ("table", "setting")

    def __init__(self, pixel2d, window_size_provider: Callable[[], tuple[int, int]]) -> None:
        self._pixel2d = pixel2d
        self._window_size_provider = window_size_provider
        self._root: Optional[DirectFrame] = None
        self._title: Optional[NodePath] = None
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[NodePath] = []
        self._visible = True
        self._last_layout_size: tuple[int, int] | None = None
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
            self.update_layout(force=True)
            logger.info("Home UI initialized successfully")
        except Exception:
            logger.exception("Failed to initialize home UI")
            raise

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

        label_scale = max(button_height * 0.32, 18)
        for button_label in self._button_labels:
            button_label.setPos(button_width * 0.5, 0, -(button_height * 0.60))
            button_label.setScale(label_scale)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._root is None:
            return
        if visible:
            self.update_layout()
        self._root.show() if visible else self._root.hide()

    def is_visible(self) -> bool:
        return self._visible

    def destroy(self) -> None:
        for label in self._button_labels:
            label.removeNode()
        self._button_labels.clear()

        for button in self._buttons:
            button.destroy()
        self._buttons.clear()

        if self._title is not None:
            self._title.removeNode()
            self._title = None

        if self._root is not None:
            self._root.destroy()
            self._root = None

        logger.info("Home UI cleaned up")