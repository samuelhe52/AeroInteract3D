from __future__ import annotations

import logging
import sys
import cv2
import numpy as np
from typing import Optional, Any, Tuple
from pathlib import Path

from panda3d.core import Texture, CardMaker, TextNode, NodePath
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame

# Allow this module to be imported both as a package module and from direct script execution.
try:
    from .auto_scaling import AutoScalingManager
except ImportError:  # pragma: no cover - fallback for direct script execution
    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from src.rendering.debug.auto_scaling import AutoScalingManager
from src.contracts import GesturePacket
from src.gesture.runtime import RawHandObservation
from src.gesture.debug.live_preview_runtime import HAND_CONNECTIONS, OverlayColors

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("cam_preview")


class CameraPreviewManager:
    """Camera preview manager, responsible for camera texture initialization, frame updates, GPU flipping, and position adaptation"""

    PREVIEW_WIDTH = 384
    PREVIEW_HEIGHT = 216
    PREVIEW_MARGIN = 12
    TITLE_OFFSET_X = 18
    TITLE_OFFSET_Y = 20
    STATUS_OFFSET_Y = 40
    
    def __init__(
        self,
        auto_scaling: AutoScalingManager,
        *,
        top_margin: int = PREVIEW_MARGIN,
        show_debug_chrome: bool = False,
    ):
        """Initialize camera preview manager (dependency injection: auto_scaling)"""
        self._auto_scaling: AutoScalingManager = auto_scaling
        self._pixel2d = auto_scaling._rendering_core.get_pixel2d()
        self._ui_scale: float = auto_scaling.get_ui_scale()
        self._top_margin = top_margin
        self._show_debug_chrome = show_debug_chrome
        self._camera_frame: Optional[np.ndarray] = None
        self._last_observation: Optional[RawHandObservation] = None
        self._last_packet: Optional[GesturePacket] = None
        self._camera_texture: Optional[Texture] = None
        self._camera_preview_node: Optional[NodePath] = None
        self._camera_preview_frame: Optional[DirectFrame] = None
        self._camera_preview_title: Optional[OnscreenText] = None
        self._camera_preview_status: Optional[OnscreenText] = None
        self._camera_preview_enabled: bool = False
        self._last_camera_update_time: float = 0
        self._camera_update_interval: float = 0.033  # 30fps
        self._colors = OverlayColors()
        self.init_preview()
    
    def init_preview(self, data_panel_raw_params=None) -> None:
        """Initialize camera preview window (original logic preserved)"""
        try:
            if self._show_debug_chrome:
                self._camera_preview_frame = DirectFrame(
                    parent=self._pixel2d,
                    pos=(self.PREVIEW_MARGIN * self._ui_scale, 0, -self._top_margin * self._ui_scale),
                    frameSize=(0, self.PREVIEW_WIDTH * self._ui_scale, -self.PREVIEW_HEIGHT * self._ui_scale, 0),
                    frameColor=(0.08, 0.09, 0.13, 0.96),
                    relief=1,
                    borderWidth=(1, 1),
                    color=(20 / 255, 24 / 255, 32 / 255, 1.0),
                )

                self._camera_preview_title = OnscreenText(
                    parent=self._pixel2d,
                    pos=((self.PREVIEW_MARGIN + self.TITLE_OFFSET_X) * self._ui_scale, -(self._top_margin + self.TITLE_OFFSET_Y) * self._ui_scale),
                    align=TextNode.ALeft,
                    scale=16 * self._ui_scale,
                    fg=(1.0, 1.0, 1.0, 1.0),
                    text="Camera Preview",
                    mayChange=False,
                )

                self._camera_preview_status = OnscreenText(
                    parent=self._pixel2d,
                    pos=((self.PREVIEW_MARGIN + self.TITLE_OFFSET_X) * self._ui_scale, -(self._top_margin + self.STATUS_OFFSET_Y) * self._ui_scale),
                    align=TextNode.ALeft,
                    scale=12 * self._ui_scale,
                    fg=(0.8, 0.8, 0.8, 1.0),
                    text="Camera: Not Connected",
                    mayChange=True,
                )
            
            # Create camera preview texture and node
            self._camera_texture = Texture("camera_preview")
            self._camera_texture.setup2dTexture(
                self.PREVIEW_WIDTH,
                self.PREVIEW_HEIGHT,
                Texture.T_unsigned_byte,
                Texture.F_rgb,
            )
            
            card_maker = CardMaker("camera_preview_card")
            card_maker.setFrame(0, self.PREVIEW_WIDTH, -self.PREVIEW_HEIGHT, 0)
            
            self._camera_preview_node = self._pixel2d.attachNewNode(card_maker.generate())
            self._apply_preview_node_transform()
            
            # Apply texture
            self._camera_preview_node.setTexture(self._camera_texture)
            
            # Enable preview
            self._camera_preview_enabled = True
            logger.info("Camera preview initialized successfully")
            
        except Exception as e:
            logger.warning(f"Camera preview initialization failed: {str(e)}")
            self._camera_preview_enabled = False
    
    def update_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        observation: Optional[RawHandObservation] = None,
        packet: Optional[GesturePacket] = None,
    ) -> None:
        """Update camera frame data"""
        if frame_bgr is not None and self._camera_preview_enabled:
            self._camera_frame = frame_bgr
            self._last_observation = observation
            self._last_packet = packet
    
    def enable_preview(self, enabled: bool = True) -> None:
        """Enable or disable camera preview"""
        self._camera_preview_enabled = enabled
        if not enabled and self._camera_preview_node is not None:
            self._camera_preview_node.removeNode()
            self._camera_preview_node = None
    
    def update_preview(self) -> None:
        """Update camera preview frame"""
        if self._camera_frame is None or self._camera_preview_node is None:
            return
            
        try:
            frame = cv2.resize(
                self._camera_frame,
                (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            
            if self._last_observation is not None:
                frame = self._draw_hand_skeleton(frame, self._last_observation)

            frame = self._draw_rotation_overlay(frame)
            
            self._camera_texture.setRamImageAs(np.ascontiguousarray(frame), "BGR")
            
            # Update status text
            if self._camera_preview_status:
                self._camera_preview_status.setText("Camera: Active")
                
        except Exception as e:
            logger.warning(f"Camera preview update failed: {str(e)}")
            if self._camera_preview_status:
                self._camera_preview_status.setText("Camera: Error")

    def _draw_hand_skeleton(self, frame: np.ndarray, observation: RawHandObservation) -> np.ndarray:
        """Draw hand skeleton (reuse live_preview logic)"""
        height, width = frame.shape[:2]
        
        # Draw connection lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(observation.landmarks) and end_idx < len(observation.landmarks):
                start_point = self._landmark_to_pixel(observation.landmarks[start_idx], width, height)
                end_point = self._landmark_to_pixel(observation.landmarks[end_idx], width, height)
                cv2.line(frame, start_point, end_point, self._colors.bones, 2)
        
        # Draw key points
        for landmark in observation.landmarks:
            point = self._landmark_to_pixel(landmark, width, height)
            cv2.circle(frame, point, 4, self._colors.landmarks, -1)
        
        return frame

    def _draw_rotation_overlay(self, frame: np.ndarray) -> np.ndarray:
        slot = 0
        slot_x = 0
        slot_y = 0
        slot_z = 0
        slot_count = 0
        deg_x = 0.0
        deg_y = 0.0
        deg_z = 0.0
        rotating = False
        mode_active = False
        mode_progress = 0
        mode_target = 1
        tip_spread = 0.0
        grab_detected = False
        packet = self._last_packet
        if packet is not None and isinstance(packet.debug, dict):
            rotation = packet.debug.get("rotation")
            if isinstance(rotation, dict):
                slot = int(rotation.get("slot", 0))
                slot_x = int(rotation.get("slot_x", slot))
                slot_y = int(rotation.get("slot_y", slot))
                slot_z = int(rotation.get("slot_z", slot))
                slot_count = int(rotation.get("slot_count", 0))
                deg_x = float(rotation.get("deg_x", 0.0))
                deg_y = float(rotation.get("deg_y", 0.0))
                deg_z = float(rotation.get("deg_z", 0.0))
                rotating = bool(rotation.get("rotating", False))
                mode_active = bool(rotation.get("mode_active", False))
                mode_progress = int(rotation.get("mode_progress", 0))
                mode_target = int(rotation.get("mode_target", 1))
                tip_spread = float(rotation.get("tip_spread", 0.0))
                grab_detected = bool(rotation.get("grab_detected", False))

        text = f"slot xyz: ({slot_x:02d},{slot_y:02d},{slot_z:02d})/{slot_count:02d}" if slot_count > 0 else f"slot xyz: ({slot_x:02d},{slot_y:02d},{slot_z:02d})"
        state = "YES" if rotating else "NO"
        cv2.rectangle(frame, (10, 12), (430, 116), (18, 22, 30), thickness=-1)
        cv2.rectangle(frame, (10, 12), (430, 116), (86, 96, 118), thickness=1)
        cv2.putText(frame, text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(frame, f"rotating: {state}", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 220, 140) if rotating else (140, 140, 140), 1, cv2.LINE_AA)
        mode_label = "ROTATE_ENABLED" if mode_active else "MOVE_ONLY"
        cv2.putText(frame, f"mode: {mode_label} ({mode_progress}/{max(mode_target, 1)})", (18, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 200, 255) if mode_active else (185, 190, 205), 1, cv2.LINE_AA)
        cv2.putText(frame, f"X: {deg_x:6.1f} deg", (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Y: {deg_y:6.1f} deg", (126, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 230, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Z: {deg_z:6.1f} deg", (236, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 170, 60), 1, cv2.LINE_AA)
        cv2.putText(frame, f"grab: {'YES' if grab_detected else 'NO'}", (18, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"spread: {tip_spread:.3f}  (grab < 0.270)", (108, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "For best results, face the camera and hold a standard OK pose.", (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1, cv2.LINE_AA)
        return frame

    def _landmark_to_pixel(self, landmark: Any, width: int, height: int) -> Tuple[int, int]:
        """Convert landmarks coordinates to pixel coordinates"""
        return (int(landmark.x * width), int(landmark.y * height))
    
    def set_ui_scale(self, scale: float) -> None:
        """Set UI scale factor and update size and position of all UI elements"""
        self._ui_scale = scale
        
        # Scale camera preview window
        if self._camera_preview_frame:
            original_pos = (self.PREVIEW_MARGIN, 0, -self._top_margin)
            original_size = (0, self.PREVIEW_WIDTH, -self.PREVIEW_HEIGHT, 0)
            new_pos = (original_pos[0] * scale, original_pos[1], original_pos[2] * scale)
            new_size = (original_size[0], original_size[1] * scale, original_size[2] * scale, original_size[3])
            self._camera_preview_frame.setPos(*new_pos)
            self._camera_preview_frame['frameSize'] = new_size
        
        # Scale camera preview title
        if self._camera_preview_title:
            try:
                original_pos = (self.PREVIEW_MARGIN + self.TITLE_OFFSET_X, -(self._top_margin + self.TITLE_OFFSET_Y))
                original_scale = 16
                new_pos = (original_pos[0] * scale, original_pos[1] * scale)
                new_scale = original_scale * scale
                self._camera_preview_title['pos'] = (new_pos[0], 0, new_pos[1])
                self._camera_preview_title['scale'] = new_scale
            except Exception as e:
                logger.warning(f"Failed to scale camera preview title: {e}")
        
        # Scale camera preview node
        if self._camera_preview_node:
            try:
                self._apply_preview_node_transform(scale)
            except Exception as e:
                logger.warning(f"Failed to scale camera preview node: {e}")
        
        # Scale camera preview status text
        if self._camera_preview_status:
            try:
                original_pos = (self.PREVIEW_MARGIN + self.TITLE_OFFSET_X, -(self._top_margin + self.STATUS_OFFSET_Y))
                original_scale = 12
                new_pos = (original_pos[0] * scale, original_pos[1] * scale)
                new_scale = original_scale * scale
                self._camera_preview_status['pos'] = (new_pos[0], 0, new_pos[1])
                self._camera_preview_status['scale'] = new_scale
            except Exception as e:
                logger.warning(f"Failed to scale camera preview status: {e}")

    def _apply_preview_node_transform(self, scale: float | None = None) -> None:
        if self._camera_preview_node is None:
            return

        active_scale = self._ui_scale if scale is None else scale
        # Keep vertical correction for texture orientation, but avoid horizontal mirroring.
        self._camera_preview_node.setScale(active_scale, 1, -active_scale)
        self._camera_preview_node.setPos(
            self.PREVIEW_MARGIN * active_scale,
            0,
            -(self._top_margin + self.PREVIEW_HEIGHT) * active_scale,
        )
    
    def destroy(self) -> None:
        """Clean up camera preview resources"""
        if self._camera_preview_node is not None:
            self._camera_preview_node.removeNode()
            self._camera_preview_node = None
        
        if self._camera_preview_frame:
            self._camera_preview_frame.destroy()
            self._camera_preview_frame = None
            
        if self._camera_preview_title:
            self._camera_preview_title.destroy()
            self._camera_preview_title = None
            
        if self._camera_preview_status:
            self._camera_preview_status.destroy()
            self._camera_preview_status = None
        
        self._camera_frame = None
        self._last_observation = None
        self._last_packet = None
        self._camera_texture = None
        self._camera_preview_enabled = False
        
        logger.info("Camera preview cleaned up")


if __name__ == "__main__":
    raise SystemExit("This module is not a standalone entrypoint. Run main.py from the project root.")