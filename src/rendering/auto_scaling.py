from __future__ import annotations

import logging
from typing import Optional, Callable

from direct.showbase.ShowBase import ShowBase

from .rendering_core import RenderingCoreManager

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("auto_scaling")


class AutoScalingManager:
    """Auto scaling manager, responsible for scaling factor calculation, main window size monitoring, and scaling range limits"""
    
    def __init__(self, rendering_core: RenderingCoreManager):
        """Initialize auto scaling manager (dependency injection: rendering_core)"""
        self._rendering_core: RenderingCoreManager = rendering_core
        self._last_window_size = (0, 0)  # Record last main window size
        self._base_window_size = (2560, 1440)  # Base size (original logic preserved)
        self._ui_scale = 1.0  # Exposed property
        self._scale_callback: Optional[Callable[[float], None]] = None
    
    def set_scale_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback function for scale changes"""
        self._scale_callback = callback
    
    def update_window_scale(self) -> None:
        """Update scaling based on window size changes (original logic preserved)"""
        base: Optional[ShowBase] = self._rendering_core.get_base()
        if base is None or base.win is None:
            return
        
        # Get current main window size
        current_size = (base.win.getXSize(), base.win.getYSize())
        
        # Only perform scaling calculation when size changes
        if current_size != self._last_window_size:
            # Calculate scale factor, take minimum of width and height scaling to ensure UI doesn't exceed window
            scale_x = current_size[0] / self._base_window_size[0]
            scale_y = current_size[1] / self._base_window_size[1]
            new_scale = min(scale_x, scale_y)
            
            # Scaling range limit (original logic preserved)
            new_scale = max(0.5, min(2.0, new_scale))
            
            # Only update if scale factor difference > 0.01 (original logic preserved)
            if abs(new_scale - self._ui_scale) > 0.01:
                self._ui_scale = new_scale
                if self._scale_callback:
                    self._scale_callback(new_scale)
                logger.debug(f"UI scale updated to: {new_scale}")
            
            # Update last window size
            self._last_window_size = current_size
    
    def get_ui_scale(self) -> float:
        """Get current UI scale factor"""
        return self._ui_scale
    
    def set_ui_scale(self, scale: float) -> None:
        """Set UI scale factor"""
        self._ui_scale = scale
        if self._scale_callback:
            self._scale_callback(scale)