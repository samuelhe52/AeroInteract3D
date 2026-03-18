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
    
    def __init__(self, auto_scaling: AutoScalingManager):
        """Initialize data panel manager (dependency injection: auto_scaling)"""
        self._auto_scaling: AutoScalingManager = auto_scaling
        self._pixel2d = auto_scaling._rendering_core.get_pixel2d()
        self._ui_scale: float = auto_scaling.get_ui_scale()
        self._status_frame: Optional[DirectFrame] = None
        self._status_panel: Optional[OnscreenText] = None
        self._last_world_norm_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_scene_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.init_panel()
    
    def init_panel(self) -> None:
        """Initialize data panel (original logic preserved)"""
        try:
            # Create status frame
            self._status_frame = DirectFrame(
                parent=self._pixel2d,
                pos=(12 * self._ui_scale, 0, -12 * self._ui_scale),  # Position (original logic preserved)
                frameSize=(0, 512 * self._ui_scale, -288 * self._ui_scale, 0),  # Width 500, height 300 (original logic preserved)
                frameColor=(0.0, 0.0, 0.0, 0.9),  # Black semi-transparent background (original logic preserved)
                relief=1,
                borderWidth=(1, 1),
                color=(60/255, 68/255, 86/255, 1.0)  # Border color (original logic preserved)
            )
            
            # Create status text panel
            self._status_panel = OnscreenText(
                parent=self._pixel2d,
                pos=(30 * self._ui_scale, -70 * self._ui_scale),
                align=TextNode.ALeft,
                scale=28 * self._ui_scale,  # Font size 28 (original logic preserved)
                fg=(1.0, 1.0, 1.0, 1.0),  # White text (original logic preserved)
                wordwrap=65,
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
    
    def update_data(self, packet: Optional[GesturePacket] = None, fps: float = 0.0) -> None:
        """Update data panel display (original logic preserved)"""
        if not self._status_panel:
            return
        
        if packet is None:
            lines = (
                "frame: 0",
                "tracking: idle",
                "pinch: idle",
                "confidence: 0.00",
                "pinch_distance: 0.000",
                "wrist: (+0.00, +0.00, +0.00)",
                f"fps: {fps:.1f}",
            )
        else:
            lines = (
                f"frame: {packet.frame_id}",
                f"tracking: {packet.tracking_state}",
                f"pinch: {packet.pinch_state}",
                f"confidence: {packet.confidence:.2f}",
                f"pinch_distance: {0.0 if packet.pinch_distance is None else packet.pinch_distance:.3f}",
                f"wrist: ({packet.wrist.x:+.2f}, {packet.wrist.y:+.2f}, {packet.wrist.z:+.2f})",
                f"fps: {fps:.1f}",
                f"world_norm: ({self._last_world_norm_pos[0]:+.2f}, {self._last_world_norm_pos[1]:+.2f}, {self._last_world_norm_pos[2]:+.2f})",
                f"scene_pos: ({self._last_scene_pos[0]:+.2f}, {self._last_scene_pos[1]:+.2f}, {self._last_scene_pos[2]:+.2f})",
            )
        
        self._status_panel.setText("\n".join(lines))
    
    def update_coordinate_data(self, world_norm_pos: tuple, scene_pos: tuple) -> None:
        """Update coordinate data for display"""
        self._last_world_norm_pos = world_norm_pos
        self._last_scene_pos = scene_pos
    
    def set_ui_scale(self, scale: float) -> None:
        """Set UI scale factor and update size and position of all UI elements"""
        self._ui_scale = scale
        
        # Scale data panel
        if self._status_frame:
            # Original position and size
            original_pos = (12, 0, -12)
            original_size = (0, 512, -288, 0)  # 512x288 updated size
            # Calculate new position and size
            new_pos = (original_pos[0] * scale, original_pos[1], original_pos[2] * scale)
            new_size = (original_size[0], original_size[1] * scale, original_size[2] * scale, original_size[3])
            self._status_frame.setPos(*new_pos)
            self._status_frame['frameSize'] = new_size
        
        # Scale data text
        if self._status_panel:
            original_pos = (30, -70)
            original_scale = 28  # Font size 28 (original logic preserved)
            new_pos = (original_pos[0] * scale, original_pos[1] * scale)
            new_scale = original_scale * scale
            self._status_panel['pos'] = new_pos
            self._status_panel['scale'] = new_scale
    
    def destroy(self) -> None:
        """Clean up resources"""
        if self._status_panel:
            self._status_panel.destroy()
            self._status_panel = None
        
        if self._status_frame:
            self._status_frame.destroy()
            self._status_frame = None
        
        logger.info("Data panel cleaned up")