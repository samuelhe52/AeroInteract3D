from __future__ import annotations

import logging
from typing import Optional
import tkinter as tk
from collections.abc import Callable

from panda3d.core import (
    AntialiasAttrib, WindowProperties, AmbientLight, DirectionalLight,
    PerspectiveLens, NodePath, GraphicsWindow, loadPrcFileData
)
from direct.showbase.ShowBase import ShowBase

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_core")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")
DEFAULT_WINDOW_ASPECT_RATIO = (16, 9)
DEFAULT_WINDOW_SCREEN_SCALE = 0.8
REFERENCE_WINDOW_SIZE = (1600, 900)
MIN_WINDOW_SIZE = (800, 450)
QUIT_SHORTCUT_EVENTS = ("meta-w", "meta-q", "control-w", "control-q")


class RenderingCoreManager:
    """Rendering core manager, responsible for window initialization, ShowBase management, frame driving, and lifecycle state management"""
    
    def __init__(self):
        self._base: Optional[ShowBase] = None
        self._is_initialized: bool = False
        self._last_window_size: tuple[int, int] | None = None
        self._pending_window_size: tuple[int, int] | None = None
        self._quit_callback: Callable[[], None] | None = None

    @staticmethod
    def compute_window_size(
        *,
        screen_size: tuple[int, int] | None = None,
        aspect_ratio: tuple[int, int] = DEFAULT_WINDOW_ASPECT_RATIO,
        screen_scale: float = DEFAULT_WINDOW_SCREEN_SCALE,
    ) -> tuple[int, int]:
        """Fit a fixed-aspect window into the current display using a relative scale."""
        fallback_width, fallback_height = REFERENCE_WINDOW_SIZE
        if screen_size is None:
            screen_size = RenderingCoreManager._detect_screen_size()
        if screen_size is None:
            return fallback_width, fallback_height

        screen_width, screen_height = screen_size
        if screen_width <= 0 or screen_height <= 0:
            return fallback_width, fallback_height

        ratio_width, ratio_height = aspect_ratio
        if ratio_width <= 0 or ratio_height <= 0:
            return fallback_width, fallback_height

        min_width, min_height = MIN_WINDOW_SIZE
        usable_width = max(int(screen_width * screen_scale), min_width)
        usable_height = max(int(screen_height * screen_scale), min_height)
        aspect_value = ratio_width / ratio_height

        width = usable_width
        height = int(width / aspect_value)
        if height > usable_height:
            height = usable_height
            width = int(height * aspect_value)

        return max(width, min_width), max(height, min_height)

    @staticmethod
    def compute_aspect_locked_size(
        window_size: tuple[int, int],
        *,
        previous_size: tuple[int, int] | None = None,
        aspect_ratio: tuple[int, int] = DEFAULT_WINDOW_ASPECT_RATIO,
    ) -> tuple[int, int]:
        min_width, min_height = MIN_WINDOW_SIZE
        width = max(int(window_size[0]), min_width)
        height = max(int(window_size[1]), min_height)

        ratio_width, ratio_height = aspect_ratio
        if ratio_width <= 0 or ratio_height <= 0:
            return width, height

        aspect_value = ratio_width / ratio_height
        if previous_size is None:
            return width, max(int(round(width / aspect_value)), min_height)

        prev_width = max(int(previous_size[0]), min_width)
        prev_height = max(int(previous_size[1]), min_height)
        width_delta = abs(width - prev_width)
        height_delta = abs(height - prev_height)

        if width_delta >= height_delta:
            locked_height = max(int(round(width / aspect_value)), min_height)
            return width, locked_height

        locked_width = max(int(round(height * aspect_value)), min_width)
        return locked_width, height

    @staticmethod
    def _detect_screen_size() -> tuple[int, int] | None:
        try:
            root = tk.Tk()
            root.withdraw()
            width = int(root.winfo_screenwidth())
            height = int(root.winfo_screenheight())
            root.destroy()
            return width, height
        except Exception:
            logger.debug("Unable to detect screen size; falling back to reference window size", exc_info=True)
            return None

    @staticmethod
    def reference_window_size() -> tuple[int, int]:
        return REFERENCE_WINDOW_SIZE

    @staticmethod
    def aspect_ratio() -> float:
        return DEFAULT_WINDOW_ASPECT_RATIO[0] / DEFAULT_WINDOW_ASPECT_RATIO[1]

    def set_quit_handler(self, callback: Callable[[], None] | None) -> None:
        self._quit_callback = callback
        self._register_quit_shortcuts()
    
    def init_window(
        self,
        window_size: tuple[int, int] | None = None,
        window_title: str = "AeroInteract3D Rendering",
    ) -> None:
        """Initialize rendering window (original logic preserved)"""
        if self._is_initialized:
            logger.info(f"Window already initialized ({window_size}), skipping duplicate creation")
            return
        try:
            resolved_window_size = window_size or self.compute_window_size()
            window_props = WindowProperties()
            window_props.setSize(*resolved_window_size)
            window_props.setFixedSize(False)
            window_props.setTitle(window_title)
            WindowProperties.setDefault(window_props)
            self._base = ShowBase()
            WindowProperties.clearDefault()
            # Set window background to white (original logic preserved)
            if self._base:
                self._base.setBackgroundColor(1, 1, 1, 1)
                self._base.render.setAntialias(AntialiasAttrib.MAuto)
                win: Optional[GraphicsWindow] = self._base.win
                if win:
                    win.requestProperties(window_props)
                    self._is_initialized = True
                    self._last_window_size = resolved_window_size
                    self._sync_camera_aspect_ratio(*resolved_window_size)
                    self._register_quit_shortcuts()
            logger.info(
                "Window initialized successfully: size=%s, title=%s",
                resolved_window_size,
                window_title,
            )
        except Exception as e:
            WindowProperties.clearDefault()
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
            
            # Position the camera with a 15-degree downward-tilt view of the objects.
            self._base.cam.setPos(0.0, 5.0, 1.340)  # 15-degree view (z = 5 * tan(15°))
            self._base.cam.lookAt(0.0, 0.0, 0.0)  # Look at the origin.
            self._base.cam.setH(180)  # Flip camera horizontally
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
    
    def get_pixel2d(self) -> Optional[NodePath]:
        """Get pixel2d node (dependency injection)"""
        if not self._is_initialized or self._base is None:
            return None
        return getattr(self._base, "pixel2d", None)
    
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
        self.enforce_window_aspect_ratio()
        self._base.taskMgr.step()

    def enforce_window_aspect_ratio(self) -> None:
        if self._base is None or self._base.win is None:
            return
        win = self._base.win
        if not hasattr(win, "getXSize") or not hasattr(win, "getYSize"):
            return

        current_size = (int(win.getXSize()), int(win.getYSize()))
        if current_size[0] <= 0 or current_size[1] <= 0:
            return

        if self._last_window_size is None:
            self._last_window_size = current_size
            self._sync_camera_aspect_ratio(*current_size)
            return

        target_size = self.compute_aspect_locked_size(
            current_size,
            previous_size=self._last_window_size,
        )
        if current_size != target_size:
            if target_size != self._pending_window_size:
                window_props = WindowProperties()
                window_props.setSize(*target_size)
                win.requestProperties(window_props)
                self._pending_window_size = target_size
            self._last_window_size = target_size
            self._sync_camera_aspect_ratio(*target_size)
            return

        self._pending_window_size = None
        self._last_window_size = current_size
        self._sync_camera_aspect_ratio(*current_size)

    def _sync_camera_aspect_ratio(self, width: int, height: int) -> None:
        if self._base is None or height <= 0:
            return
        lens = getattr(self._base, "camLens", None)
        if lens is None:
            return
        set_aspect_ratio = getattr(lens, "setAspectRatio", None)
        if callable(set_aspect_ratio):
            set_aspect_ratio(width / height)

    def _register_quit_shortcuts(self) -> None:
        if self._base is None or self._quit_callback is None:
            return

        accept = getattr(self._base, "accept", None)
        if not callable(accept):
            return

        for event_name in QUIT_SHORTCUT_EVENTS:
            accept(event_name, self._handle_quit_shortcut)

    def _handle_quit_shortcut(self) -> None:
        if self._quit_callback is None:
            return
        logger.info("Quit shortcut received from rendering window")
        self._quit_callback()
