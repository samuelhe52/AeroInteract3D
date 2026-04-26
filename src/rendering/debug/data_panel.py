from __future__ import annotations

import logging
from typing import Optional

from panda3d.core import TextNode
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame

from .auto_scaling import AutoScalingManager
from src.contracts import GesturePacket

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("data_panel")


class DataPanelManager:
    """Data panel manager, responsible for data panel frame+text initialization, gesture data updates, and scaling adaptation"""

    PANEL_WIDTH = 384
    PANEL_HEIGHT = 400
    PANEL_MARGIN = 12
    PANEL_GAP = 12
    MENU_HOLD_INDICATOR_HEIGHT = 34
    MENU_HOLD_INDICATOR_GAP = 10
    MENU_HOLD_LAMP_SIZE = 18
    MENU_HOLD_LAMP_GAP = 14
    MENU_HOLD_LABEL_OFFSET_X = 18
    MENU_HOLD_LABEL_OFFSET_Y = 22
    MENU_HOLD_LABEL_SCALE = 29
    MENU_HOLD_LAMPS_START_X = 156
    TEXT_OFFSET_X = 18
    TEXT_OFFSET_Y = 40
    TEXT_SCALE = 18
    
    def __init__(self, auto_scaling: AutoScalingManager):
        """Initialize data panel manager (dependency injection: auto_scaling)"""
        self._auto_scaling: AutoScalingManager = auto_scaling
        self._pixel2d = auto_scaling._rendering_core.get_pixel2d()
        self._ui_scale: float = auto_scaling.get_ui_scale()
        self._status_frame: Optional[DirectFrame] = None
        self._status_panel: Optional[OnscreenText] = None
        self._menu_hold_frame: Optional[DirectFrame] = None
        self._menu_hold_label: Optional[OnscreenText] = None
        self._menu_hold_lamps: list[DirectFrame] = []
        self._last_world_norm_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_scene_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._menu_hold_lit_count: int = 0
        self._indicator_visible: bool = True
        self._panel_visible: bool = True
        self._scale_ratio: float = 1.0
        self._scaling_active: bool = False
        
        # ========== 新增代码开始 ==========
        # 新增：交互调试数据存储
        self._object_world_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._index_tip_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._thumb_tip_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._distance_to_object: float = 0.0
        self._interaction_state: str = "idle"
        self._interaction_mode: str = "normal"
        # ========== 新增代码结束 ==========
        
        self.init_panel()

    @classmethod
    def panel_top_margin(cls) -> int:
        return cls.PANEL_MARGIN + cls.MENU_HOLD_INDICATOR_HEIGHT + cls.MENU_HOLD_INDICATOR_GAP

    @classmethod
    def camera_preview_top_margin(cls) -> int:
        return cls.panel_top_margin() + cls.PANEL_HEIGHT + cls.PANEL_GAP
    
    def init_panel(self) -> None:
        """Initialize data panel (original logic preserved)"""
        try:
            # Create status frame
            self._status_frame = DirectFrame(
                parent=self._pixel2d,
                pos=(self.PANEL_MARGIN * self._ui_scale, 0, -self.panel_top_margin() * self._ui_scale),
                frameSize=(0, self.PANEL_WIDTH * self._ui_scale, -self.PANEL_HEIGHT * self._ui_scale, 0),
                frameColor=(0.0, 0.0, 0.0, 0.9),  # Black semi-transparent background (original logic preserved)
                relief=1,
                borderWidth=(1, 1),
                color=(60/255, 68/255, 86/255, 1.0)  # Border color (original logic preserved)
            )

            self._menu_hold_frame = DirectFrame(
                parent=self._pixel2d,
                pos=(self.PANEL_MARGIN * self._ui_scale, 0, -self.PANEL_MARGIN * self._ui_scale),
                frameSize=(0, self.PANEL_WIDTH * self._ui_scale, -self.MENU_HOLD_INDICATOR_HEIGHT * self._ui_scale, 0),
                frameColor=(0.0, 0.0, 0.0, 0.0),
                relief=None,
            )

            self._menu_hold_label = OnscreenText(
                parent=self._pixel2d,
                pos=((self.PANEL_MARGIN + self.MENU_HOLD_LABEL_OFFSET_X) * self._ui_scale, -(self.PANEL_MARGIN + self.MENU_HOLD_LABEL_OFFSET_Y) * self._ui_scale),
                align=TextNode.ALeft,
                scale=self.MENU_HOLD_LABEL_SCALE * self._ui_scale,
                fg=(0.08, 0.09, 0.10, 1.0),
                text="menu open",
                mayChange=False,
            )

            for _ in range(3):
                lamp = DirectFrame(
                    parent=self._menu_hold_frame,
                    pos=(0, 0, 0),
                    frameSize=(0, self.MENU_HOLD_LAMP_SIZE * self._ui_scale, -self.MENU_HOLD_LAMP_SIZE * self._ui_scale, 0),
                    frameColor=(0.30, 0.30, 0.33, 0.95),
                    relief=1,
                    borderWidth=(1, 1),
                )
                self._menu_hold_lamps.append(lamp)
            self._layout_menu_hold_indicator()
            self.update_menu_hold_progress(0, candidate_active=False, overlay_active=False)
            
            # Create status text panel
            self._status_panel = OnscreenText(
                parent=self._pixel2d,
                pos=(self.TEXT_OFFSET_X * self._ui_scale, -(self.panel_top_margin() + self.TEXT_OFFSET_Y) * self._ui_scale),
                align=TextNode.ALeft,
                scale=self.TEXT_SCALE * self._ui_scale,
                fg=(1.0, 1.0, 1.0, 1.0),  # White text (original logic preserved)
                wordwrap=34,
                text="""
                ----------------------------
                ----------------------------
                frame: 0
                tracking: idle
                pinch: idle
                confidence: 0.00
                pinch_distance: 0.000
                wrist: (+0.00, +0.00, +0.00)
                fps: 0.0""",  # Text content (original logic preserved)
                mayChange=True
            )

            logger.info("Data panel initialized successfully")
        except Exception as e:
            logger.error(f"Data panel initialization failed: {str(e)}")
            raise

    def _layout_menu_hold_indicator(self) -> None:
        if self._menu_hold_frame is None:
            return
        self._menu_hold_frame.setPos(self.PANEL_MARGIN * self._ui_scale, 0, -self.PANEL_MARGIN * self._ui_scale)
        self._menu_hold_frame["frameSize"] = (0, self.PANEL_WIDTH * self._ui_scale, -self.MENU_HOLD_INDICATOR_HEIGHT * self._ui_scale, 0)

        lamp_size = self.MENU_HOLD_LAMP_SIZE * self._ui_scale
        lamp_gap = self.MENU_HOLD_LAMP_GAP * self._ui_scale
        left = self.MENU_HOLD_LAMPS_START_X * self._ui_scale
        top = -(self.MENU_HOLD_INDICATOR_HEIGHT * self._ui_scale - lamp_size) * 0.5
        for index, lamp in enumerate(self._menu_hold_lamps):
            lamp.setPos(left + index * (lamp_size + lamp_gap), 0, top)
            lamp["frameSize"] = (0, lamp_size, -lamp_size, 0)
    
    def update_data(self, packet: Optional[GesturePacket] = None, fps: float = 0.0) -> None:
        """Update data panel display (original logic preserved)"""
        if not self._status_panel:
            return

        dual_hand_lines = self._dual_hand_lines(packet)

        if packet is None:
            rotation_lines = self._default_rotation_lines()
            lines = (
                "frame: 0",
                "tracking: idle",
                "pinch: idle",
                "confidence: 0.00",
                "pinch_distance: 0.000",
                "wrist: (+0.00, +0.00, +0.00)",
                f"fps: {fps:.1f}",
                f"scale_ratio: {self._scale_ratio:.2f}x | scaling: {'YES' if self._scaling_active else 'NO'}",
                f"world_norm: ({self._last_world_norm_pos[0]:+.2f}, {self._last_world_norm_pos[1]:+.2f}, {self._last_world_norm_pos[2]:+.2f})",
                f"scene_pos: ({self._last_scene_pos[0]:+.2f}, {self._last_scene_pos[1]:+.2f}, {self._last_scene_pos[2]:+.2f})",
            ) + rotation_lines + dual_hand_lines
        else:
            rotation_lines = self._rotation_lines(packet) or self._default_rotation_lines()
            lines = (
                f"frame: {packet.frame_id}",
                f"tracking: {packet.tracking_state}",
                f"pinch: {packet.pinch_state}",
                f"confidence: {packet.confidence:.2f}",
                f"pinch_distance: {0.0 if packet.pinch_distance is None else packet.pinch_distance:.3f}",
                f"wrist: ({packet.wrist.x:+.2f}, {packet.wrist.y:+.2f}, {packet.wrist.z:+.2f})",
                f"fps: {fps:.1f}",
                f"scale_ratio: {self._scale_ratio:.2f}x | scaling: {'YES' if self._scaling_active else 'NO'}",
                f"world_norm: ({self._last_world_norm_pos[0]:+.2f}, {self._last_world_norm_pos[1]:+.2f}, {self._last_world_norm_pos[2]:+.2f})",
                f"scene_pos: ({self._last_scene_pos[0]:+.2f}, {self._last_scene_pos[1]:+.2f}, {self._last_scene_pos[2]:+.2f})",
            ) + rotation_lines + dual_hand_lines

        # ========== 新增代码开始 ==========
        # 新增：交互调试信息（放在所有内容的最后）
        if packet is not None:
            interaction_debug_lines = (
                "--- Interaction Debug ---",
                f"object_pos: ({self._object_world_pos[0]:+.2f}, {self._object_world_pos[1]:+.2f}, {self._object_world_pos[2]:+.2f})",
                f"index_tip: ({self._index_tip_world[0]:+.2f}, {self._index_tip_world[1]:+.2f}, {self._index_tip_world[2]:+.2f})",
                f"thumb_tip: ({self._thumb_tip_world[0]:+.2f}, {self._thumb_tip_world[1]:+.2f}, {self._thumb_tip_world[2]:+.2f})",
                f"distance: {self._distance_to_object:.3f} | state: {self._interaction_state} | mode: {self._interaction_mode}",
            )
            lines = lines + interaction_debug_lines
        # ========== 新增代码结束 ==========

        self._status_panel.setText("\n".join(lines))

    @staticmethod
    def _default_rotation_lines() -> tuple[str, str]:
        return (
            "rot: MOVE_ONLY move/idle g00",
            "xyz: +00.0 +00.0 +00.0",
        )

    @staticmethod
    def _rotation_lines(packet: GesturePacket) -> tuple[str, ...]:
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return ()

        rotation = debug_payload.get("rotation")
        if not isinstance(rotation, dict):
            return ()

        deg_x = float(rotation.get("deg_x", 0.0))
        deg_y = float(rotation.get("deg_y", 0.0))
        deg_z = float(rotation.get("deg_z", 0.0))
        enabled = bool(rotation.get("enabled", False))
        rotating = bool(rotation.get("rotating", False))
        mode_name = str(rotation.get("mode_name", "MOVE_ONLY"))
        gate_count = int(rotation.get("gate_count", 0))
        mode_label = "rot" if enabled else "move"
        state_label = "live" if rotating else "idle"

        return (
            f"rot: {mode_name} {mode_label}/{state_label} g{gate_count:02d}",
            f"xyz: {deg_x:+05.1f} {deg_y:+05.1f} {deg_z:+05.1f}",
        )

    @staticmethod
    def _dual_hand_lines(packet: Optional[GesturePacket]) -> tuple[str, ...]:
        if packet is None:
            return (
                "--- Dual Hand ---",
                "1st: present NO | pinched NO",
                "2nd: present NO | pinched NO",
            )
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return ()
        dual_hand = debug_payload.get("dual_hand")
        if not isinstance(dual_hand, dict):
            return ()
        primary = dual_hand.get("primary_hand")
        secondary = dual_hand.get("secondary_hand")
        return (
            "--- Dual Hand ---",
            DataPanelManager._hand_status_line("1st", primary),
            DataPanelManager._hand_status_line("2nd", secondary),
        )

    @staticmethod
    def _hand_status_line(label: str, hand: object) -> str:
        if not isinstance(hand, dict):
            return f"{label}: present NO | pinched NO"

        tracking_state = str(hand.get("tracking_state", "not_detected"))
        pinch_state = str(hand.get("pinch_state", "open"))
        present = tracking_state != "not_detected"
        pinched = present and pinch_state == "pinched"
        present_label = "YES" if present else "NO"
        pinched_label = "YES" if pinched else "NO"
        return f"{label}: present {present_label} | pinched {pinched_label}"

    def update_coordinate_data(self, world_norm_pos: tuple, scene_pos: tuple) -> None:
        """Update coordinate data for display"""
        self._last_world_norm_pos = world_norm_pos
        self._last_scene_pos = scene_pos

    def update_menu_hold_progress(self, hold_ms: int, *, candidate_active: bool, overlay_active: bool) -> None:
        if overlay_active:
            lit_count = 3
        elif not candidate_active:
            lit_count = 0
        else:
            lit_count = min(3, max(int(hold_ms), 0) // 1000)

        self._menu_hold_lit_count = lit_count
        lamp_colors = (
            (0.16, 0.68, 0.32, 0.98),
            (0.92, 0.72, 0.18, 0.98),
            (0.88, 0.24, 0.20, 0.98),
        )
        off_color = (0.30, 0.30, 0.33, 0.95)
        for index, lamp in enumerate(self._menu_hold_lamps):
            lamp["frameColor"] = lamp_colors[index] if index < lit_count else off_color

    def update_scale_status(self, *, scale_ratio: float, scaling_active: bool) -> None:
        self._scale_ratio = float(scale_ratio)
        self._scaling_active = bool(scaling_active)

    def _update_scale_from_packet(self, packet: Optional[GesturePacket]) -> None:
        if packet is None:
            return
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return
        dual_hand = debug_payload.get("dual_hand")
        if not isinstance(dual_hand, dict):
            return

        ratio = dual_hand.get("scale_ratio")
        if isinstance(ratio, (int, float)):
            self._scale_ratio = float(ratio)

        # Fallback state when bridge-level dual-scale status is not available.
        both_pinched = dual_hand.get("both_pinched")
        if isinstance(both_pinched, bool) and not self._scaling_active:
            self._scaling_active = both_pinched

    # ========== 新增代码开始 ==========
    def update_interaction_debug_data(
        self,
        object_world_pos: tuple[float, float, float],
        index_tip_world: tuple[float, float, float],
        thumb_tip_world: tuple[float, float, float],
        distance_to_object: float,
        interaction_state: str,
        interaction_mode: str
    ) -> None:
        """更新交互调试数据，用于距离检测功能验证"""
        self._object_world_pos = object_world_pos
        self._index_tip_world = index_tip_world
        self._thumb_tip_world = thumb_tip_world
        self._distance_to_object = distance_to_object
        self._interaction_state = interaction_state
        self._interaction_mode = interaction_mode
    # ========== 新增代码结束 ==========
    
    def set_ui_scale(self, scale: float) -> None:
        """Set UI scale factor and update size and position of all UI elements"""
        self._ui_scale = scale
        
        # Scale data panel
        if self._status_frame:
            # Original position and size
            original_pos = (self.PANEL_MARGIN, 0, -self.panel_top_margin())
            original_size = (0, self.PANEL_WIDTH, -self.PANEL_HEIGHT, 0)
            # Calculate new position and size
            new_pos = (original_pos[0] * scale, original_pos[1], original_pos[2] * scale)
            new_size = (original_size[0], original_size[1] * scale, original_size[2] * scale, original_size[3])
            self._status_frame.setPos(*new_pos)
            self._status_frame['frameSize'] = new_size

        if self._menu_hold_frame:
            self._layout_menu_hold_indicator()
            self.update_menu_hold_progress(self._menu_hold_lit_count * 1000, candidate_active=self._menu_hold_lit_count > 0, overlay_active=self._menu_hold_lit_count >= 3)
        if self._menu_hold_label:
            original_pos = (self.PANEL_MARGIN + self.MENU_HOLD_LABEL_OFFSET_X, -(self.PANEL_MARGIN + self.MENU_HOLD_LABEL_OFFSET_Y))
            self._menu_hold_label['pos'] = (original_pos[0] * scale, original_pos[1] * scale)
            self._menu_hold_label['scale'] = self.MENU_HOLD_LABEL_SCALE * scale
        
        # Scale data text
        if self._status_panel:
            original_pos = (self.TEXT_OFFSET_X, -(self.panel_top_margin() + self.TEXT_OFFSET_Y))
            original_scale = self.TEXT_SCALE
            new_pos = (original_pos[0] * scale, original_pos[1] * scale)
            new_scale = original_scale * scale
            self._status_panel['pos'] = new_pos
            self._status_panel['scale'] = new_scale

    def set_visible(self, visible: bool) -> None:
        self.set_indicator_visible(visible)
        self.set_panel_visible(visible)

    def set_indicator_visible(self, visible: bool) -> None:
        self._indicator_visible = visible
        if self._menu_hold_frame:
            self._menu_hold_frame.show() if visible else self._menu_hold_frame.hide()
        if self._menu_hold_label:
            self._menu_hold_label.show() if visible else self._menu_hold_label.hide()

    def set_panel_visible(self, visible: bool) -> None:
        self._panel_visible = visible
        if self._status_frame:
            self._status_frame.show() if visible else self._status_frame.hide()
        if self._status_panel:
            self._status_panel.show() if visible else self._status_panel.hide()

    def set_brightness(self, brightness: float) -> None:
        for widget in (self._menu_hold_frame, self._menu_hold_label, *self._menu_hold_lamps, self._status_frame, self._status_panel):
            if widget is None:
                continue
            set_color_scale = getattr(widget, "setColorScale", None)
            if callable(set_color_scale):
                set_color_scale(brightness, brightness, brightness, 1.0)
    
    def destroy(self) -> None:
        """Clean up resources"""
        if self._status_panel:
            self._status_panel.destroy()
            self._status_panel = None

        if self._menu_hold_label:
            self._menu_hold_label.destroy()
            self._menu_hold_label = None

        for lamp in self._menu_hold_lamps:
            lamp.destroy()
        self._menu_hold_lamps.clear()

        if self._menu_hold_frame:
            self._menu_hold_frame.destroy()
            self._menu_hold_frame = None
        
        if self._status_frame:
            self._status_frame.destroy()
            self._status_frame = None
        
        logger.info("Data panel cleaned up")
