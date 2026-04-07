from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RenderView(StrEnum):
    HOME = "home"
    TABLE = "table"
    SETTING = "setting"


@dataclass(slots=True)
class RenderingViewState:
    active_view: RenderView = RenderView.HOME

    def set_active_view(self, view: RenderView | str) -> RenderView:
        self.active_view = RenderView(view)
        return self.active_view


@dataclass(slots=True)
class UIInputState:
    cursor_norm: tuple[float, float] = (0.5, 0.5)
    cursor_pixels: tuple[float, float] = (0.0, 0.0)
    visible: bool = False