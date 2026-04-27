from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts import GesturePacket, PinchState

from .state import UIInputState, UISettingsState


@dataclass(slots=True, frozen=True)
class UICursorSource:
    midpoint_x: float
    midpoint_y: float
    pinch_state: PinchState | None


class UIGestureInputAdapter:
    def __init__(self) -> None:
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

    def set_calibration_settings(self, settings: UISettingsState) -> None:
        self._scale_x = float(settings.ui_cursor_scale_x)
        self._scale_y = float(settings.ui_cursor_scale_y)
        self._offset_x = float(settings.ui_cursor_offset_x)
        self._offset_y = float(settings.ui_cursor_offset_y)

    @staticmethod
    def _midpoint_from_points(index_tip: Any, thumb_tip: Any) -> tuple[float, float] | None:
        if not isinstance(index_tip, dict) or not isinstance(thumb_tip, dict):
            return None
        try:
            midpoint_x = (float(index_tip["x"]) + float(thumb_tip["x"])) * 0.5
            midpoint_y = (float(index_tip["y"]) + float(thumb_tip["y"])) * 0.5
        except (KeyError, TypeError, ValueError):
            return None
        return midpoint_x, midpoint_y

    @staticmethod
    def _hand_payload(packet: GesturePacket, field_name: str) -> dict[str, Any] | None:
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return None

        payload = debug_payload.get(field_name)
        if isinstance(payload, dict):
            return payload

        dual_hand = debug_payload.get("dual_hand")
        if isinstance(dual_hand, dict):
            nested = dual_hand.get(field_name)
            if isinstance(nested, dict):
                return nested
        return None

    def cursor_source(self, packet: GesturePacket | None) -> UICursorSource | None:
        if packet is None:
            return None

        if packet.tracking_state == "tracked":
            midpoint_x = (float(packet.index_tip.x) + float(packet.thumb_tip.x)) * 0.5
            midpoint_y = (float(packet.index_tip.y) + float(packet.thumb_tip.y)) * 0.5
            return UICursorSource(
                midpoint_x=midpoint_x,
                midpoint_y=midpoint_y,
                pinch_state=packet.pinch_state,
            )

        secondary_hand = self._hand_payload(packet, "secondary_hand")
        if not isinstance(secondary_hand, dict):
            return None
        if secondary_hand.get("tracking_state") != "tracked":
            return None

        midpoint = self._midpoint_from_points(
            secondary_hand.get("index_tip"),
            secondary_hand.get("thumb_tip"),
        )
        if midpoint is None:
            return None

        pinch_state = secondary_hand.get("pinch_state")
        resolved_pinch_state = pinch_state if isinstance(pinch_state, str) else None
        return UICursorSource(
            midpoint_x=midpoint[0],
            midpoint_y=midpoint[1],
            pinch_state=resolved_pinch_state,
        )

    def to_ui_input(
        self,
        packet: GesturePacket | None,
        *,
        window_size: tuple[int, int],
    ) -> UIInputState:
        width = max(int(window_size[0]), 1)
        height = max(int(window_size[1]), 1)

        cursor_source = self.cursor_source(packet)
        if cursor_source is None:
            return UIInputState(cursor_pixels=(width * 0.5, height * 0.5), visible=False)

        raw_cursor_norm = (
            (1.0 - cursor_source.midpoint_x) * 0.5,
            (1.0 - cursor_source.midpoint_y) * 0.5,
        )

        cursor_norm = (
            max(0.0, min(1.0, (raw_cursor_norm[0] - 0.5) * self._scale_x + 0.5 + self._offset_x)),
            max(0.0, min(1.0, (raw_cursor_norm[1] - 0.5) * self._scale_y + 0.5 + self._offset_y)),
        )
        cursor_pixels = (
            cursor_norm[0] * width,
            cursor_norm[1] * height,
        )
        return UIInputState(
            cursor_norm=cursor_norm,
            cursor_pixels=cursor_pixels,
            visible=True,
        )


__all__ = ["UICursorSource", "UIGestureInputAdapter"]