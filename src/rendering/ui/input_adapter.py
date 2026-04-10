from __future__ import annotations

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

        if packet is None or packet.tracking_state != "tracked":
            return UIInputState(cursor_pixels=(width * 0.5, height * 0.5), visible=False)

        midpoint_x = (float(packet.index_tip.x) + float(packet.thumb_tip.x)) * 0.5
        midpoint_y = (float(packet.index_tip.y) + float(packet.thumb_tip.y)) * 0.5
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