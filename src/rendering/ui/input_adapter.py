from __future__ import annotations

from typing import Any

from src.contracts import GesturePacket

from .state import UIInputState, UISettingsState


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

    def to_ui_input(
        self,
        packet: GesturePacket | None,
        *,
        window_size: tuple[int, int],
    ) -> UIInputState:
        width = max(int(window_size[0]), 1)
        height = max(int(window_size[1]), 1)

        midpoint = self._cursor_midpoint(packet)
        if midpoint is None:
            return UIInputState(cursor_pixels=(width * 0.5, height * 0.5), visible=False)

        midpoint_x, midpoint_y = midpoint
        raw_cursor_norm = (
            (1.0 - midpoint_x) * 0.5,
            (1.0 - midpoint_y) * 0.5,
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

    def _cursor_midpoint(self, packet: GesturePacket | None) -> tuple[float, float] | None:
        if packet is None:
            return None
        if packet.tracking_state == "tracked":
            return (
                (float(packet.index_tip.x) + float(packet.thumb_tip.x)) * 0.5,
                (float(packet.index_tip.y) + float(packet.thumb_tip.y)) * 0.5,
            )

        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return None
        return self._secondary_hand_midpoint(debug_payload.get("secondary_hand"))

    @staticmethod
    def _secondary_hand_midpoint(secondary_hand: Any) -> tuple[float, float] | None:
        if not isinstance(secondary_hand, dict):
            return None
        if secondary_hand.get("tracking_state") != "tracked":
            return None
        index_tip = secondary_hand.get("index_tip")
        thumb_tip = secondary_hand.get("thumb_tip")
        if not isinstance(index_tip, dict) or not isinstance(thumb_tip, dict):
            return None
        try:
            return (
                (float(index_tip["x"]) + float(thumb_tip["x"])) * 0.5,
                (float(index_tip["y"]) + float(thumb_tip["y"])) * 0.5,
            )
        except (KeyError, TypeError, ValueError):
            return None