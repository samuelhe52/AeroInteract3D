from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.contracts import PinchState

from .state import UIInputState


@dataclass(slots=True, frozen=True)
class UIButtonBounds:
    left: float
    top: float
    right: float
    bottom: float

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(slots=True, frozen=True)
class UIButtonInteractionSnapshot:
    hovered_index: int | None = None
    pressed_index: int | None = None
    activated_index: int | None = None


class UIButtonInteractionController:
    def __init__(self) -> None:
        self._pressed_index: int | None = None
        self._last_hovered_index: int | None = None

    def reset(self) -> None:
        self._pressed_index = None
        self._last_hovered_index = None

    def update(
        self,
        input_state: UIInputState,
        *,
        pinch_state: PinchState | None,
        button_bounds: Sequence[UIButtonBounds],
    ) -> UIButtonInteractionSnapshot:
        hovered_index = None
        if input_state.visible:
            cursor_x, cursor_y = input_state.cursor_pixels
            for index, bounds in enumerate(button_bounds):
                if bounds.contains(cursor_x, cursor_y):
                    hovered_index = index
                    break
        if hovered_index is not None:
            self._last_hovered_index = hovered_index

        is_pressed = pinch_state in {"pinched", "release_candidate"}
        activated_index = None

        if self._pressed_index is None:
            if pinch_state == "pinched" and hovered_index is not None:
                self._pressed_index = hovered_index
        elif not is_pressed:
            if hovered_index == self._pressed_index or (
                not input_state.visible and self._last_hovered_index == self._pressed_index
            ):
                activated_index = self._pressed_index
            self._pressed_index = None
            self._last_hovered_index = hovered_index

        return UIButtonInteractionSnapshot(
            hovered_index=hovered_index,
            pressed_index=self._pressed_index if is_pressed else None,
            activated_index=activated_index,
        )


__all__ = ["UIButtonBounds", "UIButtonInteractionController", "UIButtonInteractionSnapshot"]