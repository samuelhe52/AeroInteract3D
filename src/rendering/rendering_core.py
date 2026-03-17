from __future__ import annotations

import logging
from typing import Optional

from panda3d.core import (
    WindowProperties, AmbientLight, DirectionalLight,
    PerspectiveLens, NodePath
)
from direct.showbase.ShowBase import ShowBase

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_core")


class RenderingCoreManager:
    """Rendering core manager, responsible for window initialization, ShowBase management, frame driving, and lifecycle state management"""
    
    def __init__(self):
        self._base: Optional[ShowBase] = None
        self._is_initialized: bool = False
    
    def init_window(self, window_size: tuple = (2560, 1440), window_title: str = "AeroInteract3D Rendering") -> None:
        """Initialize rendering window (original logic preserved)"""
        if self._is_initialized:
            logger.info(f"Window already initialized ({window_size}), skipping duplicate creation")
            return
        try:
            window_props = WindowProperties()
            window_props.setSize(*window_size)
            window_props.setTitle(window_title)
            # Use correct way to set window properties
            self._base = ShowBase()
            # Set window background to white (original logic preserved)
            self._base.setBackgroundColor(1, 1, 1, 1)
            self._base.win.requestProperties(window_props)
            self._is_initialized = True
            logger.info(f"Window initialized successfully: size={window_size}, title={window_title}")
        except Exception as e:
            logger.error(f"Window initialization failed: {str(e)}")
            raise RuntimeError(f"Window initialization failed: {str(e)}") from e
    
    def config_camera_for_world_norm(self) -> None:
        """Configure world coordinate system camera (original logic preserved)"""
        if not self._is_initialized:
            raise RuntimeError("Window is not initialized; cannot configure the camera")
        try:
            # A perspective camera gives a clearer 3D view than an orthographic one.
            lens = PerspectiveLens()
            lens.setFov(60)  # Set the field of view. (original logic preserved)
            lens.setNearFar(0.1, 100.0)  # Use a more practical near/far clip range.
            self._base.cam.node().setLens(lens)
            
            # Position the camera with a shallow 10-degree俯角 view of the objects.
            self._base.cam.setPos(0.0, 5.0, 0.9)  # Low angle view (≈10 degrees)
            self._base.cam.lookAt(0.0, 0.0, 0.0)  # Look at the origin.
            logger.info("Camera configured, using perspective camera for 3D scene")
        except Exception as e:
            logger.error(f"Camera configuration failed: {str(e)}")
            raise RuntimeError(f"Camera configuration failed: {str(e)}") from e
    
    def create_base_lights(self) -> None:
        """Create basic lighting setup (original logic preserved)"""
        if not self._is_initialized:
            raise RuntimeError("Window is not initialized; cannot create lights")
        try:
            # Ambient light.
            amb_light = AmbientLight("ambient_light")
            amb_light.setColor((0.2, 0.2, 0.2, 1.0))
            amb_light_np = self._base.render.attachNewNode(amb_light)
            self._base.render.setLight(amb_light_np)
            # Directional light.
            dir_light = DirectionalLight("directional_light")
            dir_light.setColor((0.8, 0.8, 0.8, 1.0))
            dir_light_np = self._base.render.attachNewNode(dir_light)
            dir_light_np.setHpr(45, -45, 0)
            self._base.render.setLight(dir_light_np)
            logger.info("Basic lights created successfully")
        except Exception as e:
            logger.error(f"Light creation failed: {str(e)}")
            raise RuntimeError(f"Light creation failed: {str(e)}") from e
    
    def get_base(self) -> Optional[ShowBase]:
        """Get ShowBase instance (dependency injection)"""
        return self._base
    
    def get_pixel2d(self):
        """Get pixel2d node (dependency injection)"""
        if not self._is_initialized or self._base is None:
            return None
        return self._base.pixel2d
    
    def is_initialized(self) -> bool:
        """Check if initialized"""
        return self._is_initialized
    
    def reset_scene(self, scene_root: NodePath) -> None:
        """Reset scene graph"""
        if not self._is_initialized:
            raise RuntimeError("Window is not initialized; cannot reset the scene")
        scene_root.get_children().detach()
        logger.info("Scene reset safely (window/camera/lights preserved)")

    def step(self) -> None:
        """Advance Panda3D by one frame, process window events and present scene"""
        if not self._is_initialized or self._base is None:
            return
        self._base.taskMgr.step()