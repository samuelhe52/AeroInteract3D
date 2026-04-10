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

    def contains(self, x: float, y: float, *, padding: float = 0.0) -> bool:
        return (
            self.left - padding <= x <= self.right + padding
            and self.top - padding <= y <= self.bottom + padding
        )


@dataclass(slots=True, frozen=True)
class UIButtonInteractionSnapshot:
    hovered_index: int | None = None
    pressed_index: int | None = None
    activated_index: int | None = None


class UIButtonInteractionController:
    PRESS_START_STATES = {"pinch_candidate", "pinched"}
    PRESSED_STATES = {"pinch_candidate", "pinched", "release_candidate"}
    PRESS_SLOP_PX = 24.0
    RELEASE_SLOP_PX = 40.0

    def __init__(self) -> None:
        self._pressed_index: int | None = None
        self._last_hovered_index: int | None = None

    def reset(self) -> None:
        self._pressed_index = None
        self._last_hovered_index = None

    @staticmethod
    def _index_at_cursor(
        input_state: UIInputState,
        button_bounds: Sequence[UIButtonBounds],
        *,
        padding: float = 0.0,
    ) -> int | None:
        if not input_state.visible:
            return None
        cursor_x, cursor_y = input_state.cursor_pixels
        for index, bounds in enumerate(button_bounds):
            if bounds.contains(cursor_x, cursor_y, padding=padding):
                return index
        return None

    def update(
        self,
        input_state: UIInputState,
        *,
        pinch_state: PinchState | None,
        button_bounds: Sequence[UIButtonBounds],
    ) -> UIButtonInteractionSnapshot:
        hovered_index = self._index_at_cursor(input_state, button_bounds)
        press_candidate_index = hovered_index
        if press_candidate_index is None and pinch_state == "pinched":
            press_candidate_index = self._index_at_cursor(
                input_state,
                button_bounds,
                padding=self.PRESS_SLOP_PX,
            )
        if hovered_index is not None:
            self._last_hovered_index = hovered_index

        is_pressed = pinch_state in self.PRESSED_STATES
        activated_index = None

        if self._pressed_index is not None and self._pressed_index >= len(button_bounds):
            self._pressed_index = None
            self._last_hovered_index = hovered_index

        if self._pressed_index is None:
            if pinch_state in self.PRESS_START_STATES and press_candidate_index is not None:
                self._pressed_index = press_candidate_index
        elif not is_pressed:
            release_index = hovered_index
            if release_index != self._pressed_index and input_state.visible:
                cursor_x, cursor_y = input_state.cursor_pixels
                pressed_bounds = button_bounds[self._pressed_index]
                if pressed_bounds.contains(cursor_x, cursor_y, padding=self.RELEASE_SLOP_PX):
                    release_index = self._pressed_index
            if release_index == self._pressed_index or (
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
