from __future__ import annotations

import logging
from typing import Optional

from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


logger = logging.getLogger("rendering.ui.home_view")


class HomeUIView:
    TITLE_TEXT = "aerointeract3d"
    BUTTON_LABELS = ("table", "setting")

    def __init__(self, pixel2d) -> None:
        self._pixel2d = pixel2d
        self._root: Optional[DirectFrame] = None
        self._title: Optional[OnscreenText] = None
        self._buttons: list[DirectFrame] = []
        self._button_labels: list[OnscreenText] = []
        self._visible = True
        self.init_view()

    def init_view(self) -> None:
        try:
            self._root = DirectFrame(
                parent=self._pixel2d,
                pos=(0, 0, 0),
                frameSize=(0, 1600, -900, 0),
                frameColor=(0.94, 0.93, 0.90, 1.0),
            )

            self._title = OnscreenText(
                parent=self._root,
                text=self.TITLE_TEXT,
                pos=(1500, -72),
                align=TextNode.ARight,
                scale=42,
                fg=(0.10, 0.13, 0.18, 1.0),
                mayChange=False,
            )

            button_top = -700
            for index, label in enumerate(self.BUTTON_LABELS):
                button_y = button_top - (index * 108)
                button = DirectFrame(
                    parent=self._root,
                    pos=(72, 0, button_y),
                    frameSize=(0, 224, -84, 0),
                    frameColor=(0.18, 0.22, 0.28, 1.0),
                    relief=1,
                    borderWidth=(1, 1),
                )
                button_label = OnscreenText(
                    parent=button,
                    text=label,
                    pos=(112, -50),
                    align=TextNode.ACenter,
                    scale=26,
                    fg=(0.96, 0.95, 0.92, 1.0),
                    mayChange=False,
                )
                self._buttons.append(button)
                self._button_labels.append(button_label)
            logger.info("Home UI initialized successfully")
        except Exception:
            logger.exception("Failed to initialize home UI")
            raise

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._root is None:
            return
        self._root.show() if visible else self._root.hide()

    def is_visible(self) -> bool:
        return self._visible

    def destroy(self) -> None:
        for label in self._button_labels:
            label.destroy()
        self._button_labels.clear()

        for button in self._buttons:
            button.destroy()
        self._buttons.clear()

        if self._title is not None:
            self._title.destroy()
            self._title = None

        if self._root is not None:
            self._root.destroy()
            self._root = None

        logger.info("Home UI cleaned up")