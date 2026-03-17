from __future__ import annotations

import logging
import cv2
import numpy as np
from typing import Optional

from panda3d.core import Texture, CardMaker, TextNode
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame

from src.gesture.debug.live_preview_runtime import HAND_CONNECTIONS, OverlayColors

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("cam_preview")


class CameraPreviewManager:
    """Camera preview manager, responsible for camera texture initialization, frame updates, GPU flipping, and position adaptation"""
    
    def __init__(self, auto_scaling):
        """Initialize camera preview manager (dependency injection: auto_scaling)"""
        self._auto_scaling = auto_scaling
        self._pixel2d = auto_scaling._rendering_core.get_pixel2d()
        self._ui_scale = auto_scaling.get_ui_scale()
        self._camera_frame = None
        self._last_observation = None
        self._camera_texture = None
        self._camera_preview_node = None
        self._camera_preview_frame = None
        self._camera_preview_title = None
        self._camera_preview_status = None
        self._camera_preview_enabled = False
        self._last_camera_update_time = 0
        self._camera_update_interval = 0.033  # 30fps
        self._colors = OverlayColors()
        self.init_preview()
    
    def init_preview(self, data_panel_raw_params=None) -> None:
        """Initialize camera preview window (original logic preserved)"""
        try:
            # Create camera preview background panel (placed below the data panel)
            self._camera_preview_frame = DirectFrame(
                parent=self._pixel2d,
                pos=(12 * self._ui_scale, 0, -300 * self._ui_scale),  # Moved up by 20 units
                frameSize=(0, 512 * self._ui_scale, -288 * self._ui_scale, 0),  # 512x288 updated size
                frameColor=(0.0, 0.0, 0.9, 0.9),
                relief=1,
                borderWidth=(1, 1),
                color=(20/255, 24/255, 32/255, 1.0)
            )
            
            # Create camera preview title
            self._camera_preview_title = OnscreenText(
                parent=self._pixel2d,
                pos=(30 * self._ui_scale, -310 * self._ui_scale),  # Moved up by 20 units
                align=TextNode.ALeft,
                scale=20 * self._ui_scale,
                fg=(1.0, 1.0, 1.0, 1.0),
                text="Camera Preview",
                mayChange=False
            )
            
            # Create camera preview status text
            self._camera_preview_status = OnscreenText(
                parent=self._pixel2d,
                pos=(30 * self._ui_scale, -330 * self._ui_scale),  # Moved up by 20 units
                align=TextNode.ALeft,
                scale=16 * self._ui_scale,
                fg=(0.8, 0.8, 0.8, 1.0),
                text="Camera: Not Connected",
                mayChange=True
            )
            
            # Create camera preview texture and node
            self._camera_texture = Texture("camera_preview")
            self._camera_texture.setup2dTexture(512, 288, Texture.T_unsigned_byte, Texture.F_rgb)  # 512x288 updated size
            
            card_maker = CardMaker("camera_preview_card")
            card_maker.setFrame(0, 512, -288, 0)  # 512x288 updated size
            
            self._camera_preview_node = self._pixel2d.attachNewNode(card_maker.generate())
            # GPU flipping (original logic preserved)
            self._camera_preview_node.setScale(-self._ui_scale, 1, -self._ui_scale)  # scale(-1,1,-1)
            self._camera_preview_node.setPos((12 + 512) * self._ui_scale, 0, (-300 - 288) * self._ui_scale)  # Moved up by 20 units
            
            # Apply texture
            self._camera_preview_node.setTexture(self._camera_texture)
            
            # Enable preview
            self._camera_preview_enabled = True
            logger.info("Camera preview initialized successfully")
            
        except Exception as e:
            logger.warning(f"Camera preview initialization failed: {str(e)}")
            self._camera_preview_enabled = False
    
    def update_frame(self, frame_bgr, observation=None) -> None:
        """Update camera frame data"""
        if frame_bgr is not None and self._camera_preview_enabled:
            self._camera_frame = frame_bgr
            self._last_observation = observation
    
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
            # Resize image with INTER_NEAREST interpolation for performance optimization
            frame = cv2.resize(self._camera_frame, (512, 288), interpolation=cv2.INTER_NEAREST)  # 512x288 updated size
            
            # If there is gesture data, draw hand skeleton (reuse live_preview logic)
            if self._last_observation is not None:
                frame = self._draw_hand_skeleton(frame, self._last_observation)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update texture
            self._camera_texture.setRamImage(frame_rgb)
            
            # Update status text
            if self._camera_preview_status:
                self._camera_preview_status.setText("Camera: Active")
                
        except Exception as e:
            logger.warning(f"Camera preview update failed: {str(e)}")
            if self._camera_preview_status:
                self._camera_preview_status.setText("Camera: Error")
    
    def _draw_hand_skeleton(self, frame, observation) -> np.ndarray:
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
    
    def _landmark_to_pixel(self, landmark, width, height) -> tuple[int, int]:
        """Convert landmarks coordinates to pixel coordinates"""
        return (int(landmark.x * width), int(landmark.y * height))
    
    def set_ui_scale(self, scale: float) -> None:
        """Set UI scale factor and update size and position of all UI elements"""
        self._ui_scale = scale
        
        # Scale camera preview window
        if self._camera_preview_frame:
            original_pos = (12, 0, -300)  # Moved up by 20 units
            original_size = (0, 512, -288, 0)  # 512x288 original size (original logic preserved)
            new_pos = (original_pos[0] * scale, original_pos[1], original_pos[2] * scale)
            new_size = (original_size[0], original_size[1] * scale, original_size[2] * scale, original_size[3])
            self._camera_preview_frame.setPos(*new_pos)
            self._camera_preview_frame['frameSize'] = new_size
        
        # Scale camera preview title
        if self._camera_preview_title:
            try:
                original_pos = (30, -310)
                original_scale = 20
                new_pos = (original_pos[0] * scale, original_pos[1] * scale)
                new_scale = original_scale * scale
                self._camera_preview_title.setPos(*new_pos)
                self._camera_preview_title['scale'] = new_scale
            except Exception as e:
                logger.warning(f"Failed to scale camera preview title: {e}")
        
        # Scale camera preview node
        if self._camera_preview_node:
            try:
                original_pos = (12 + 512, 0, -300 - 288)  # Moved up by 20 units
                new_pos = (original_pos[0] * scale, original_pos[1], original_pos[2] * scale)
                self._camera_preview_node.setPos(*new_pos)
                # GPU flipping (original logic preserved)
                self._camera_preview_node.setScale(-scale, 1, -scale)  # scale(-1,1,-1)
            except Exception as e:
                logger.warning(f"Failed to scale camera preview node: {e}")
        
        # Scale camera preview status text
        if self._camera_preview_status:
            try:
                original_pos = (30, -330)
                original_scale = 16
                new_pos = (original_pos[0] * scale, original_pos[1] * scale)
                new_scale = original_scale * scale
                self._camera_preview_status.setPos(*new_pos)
                self._camera_preview_status['scale'] = new_scale
            except Exception as e:
                logger.warning(f"Failed to scale camera preview status: {e}")
    
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
        self._camera_texture = None
        self._camera_preview_enabled = False
        
        logger.info("Camera preview cleaned up")