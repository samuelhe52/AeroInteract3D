from __future__ import annotations

from dataclasses import dataclass
import time
import logging
from typing import List, Dict, Optional, Set, Any, Callable

from panda3d.core import (
    WindowProperties, Material, AmbientLight, DirectionalLight,
    PerspectiveLens, Vec4, NodePath, TextNode, Texture, CardMaker
)
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame
from direct.showbase.ShowBase import ShowBase
import cv2
import numpy as np

from src.gesture.debug.live_preview_runtime import GesturePreviewWindow, HAND_CONNECTIONS, OverlayColors

from src.constants import MAX_ERROR_HISTORY, RENDER_POSE_LOG_DEBOUNCE_MS
from src.contracts import SceneCommand
from src.ports import RenderOutputPort
from src.utils.contracts import EXPECTED_CONTRACT_VERSION, validate_scene_command
from src.utils.runtime import (
    LIFECYCLE_INITIALIZING, LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED, LIFECYCLE_STOPPED,
    build_health, classify_frame, error_entry
)

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_service")
VALID_PAYLOAD_KEYS = {
    "init_scene": {"objects"},
    "set_object_pose": {"coordinate_space", "position", "hpr"},
    "set_object_state": {"interaction_state"},
    "reset_interaction": set(),
    "heartbeat": {"interaction_state"},
}


# Initial object state.
@dataclass(slots=True)
class ObjectInitialState:
    pos: tuple[float, float, float]
    hpr: tuple[float, float, float]
    state: str = "idle"


@dataclass(slots=True)
class RenderingMetrics:
    commands_seen: int = 0
    commands_applied: int = 0
    duplicate_commands: int = 0
    stale_commands: int = 0
    rejected_commands: int = 0
    resets_processed: int = 0
    pose_updates: int = 0
    state_updates: int = 0
    init_scene_commands: int = 0
    heartbeats_received: int = 0
    render_steps: int = 0


class Panda3DWindowAdapter:
    """Panda3D window adapter for window, camera, and light lifecycle management."""
    
    def __init__(self):
        self._base: Optional[ShowBase] = None
        self._is_initialized: bool = False
    
    def init_window(self, window_size: tuple = (2560, 1440), window_title: str = "AeroInteract3D Rendering") -> None:
        """Initialize the rendering window."""
        if self._is_initialized:
            logger.info(f"Window already initialized ({window_size}), skipping duplicate creation")
            return
        try:
            window_props = WindowProperties()
            window_props.setSize(*window_size)
            window_props.setTitle(window_title)
            # Use correct way to set window properties
            self._base = ShowBase()
            # Set window background to white
            self._base.setBackgroundColor(1, 1, 1, 1)
            self._base.win.requestProperties(window_props)
            self._is_initialized = True
            logger.info(f"Window initialized successfully: size={window_size}, title={window_title}")
        except Exception as e:
            logger.error(f"Window initialization failed: {str(e)}")
            raise RuntimeError(f"Window initialization failed: {str(e)}") from e
    
    def config_camera_for_world_norm(self) -> None:
        """Configure the camera for the world_norm coordinate space."""
        if not self._is_initialized:
            raise RuntimeError("Window is not initialized; cannot configure the camera")
        try:
            # A perspective camera gives a clearer 3D view than an orthographic one.
            lens = PerspectiveLens()
            lens.setFov(60)  # Set the field of view.
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
        """Create the base lighting setup."""
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
        return self._base
    
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def reset_scene(self, scene_root: NodePath) -> None:
        """Reset the scene graph."""
        if not self._is_initialized:
            raise RuntimeError("Window is not initialized; cannot reset the scene")
        scene_root.get_children().detach()
        logger.info("Scene reset safely (window/camera/lights preserved)")

    def step(self) -> None:
        """Advance Panda3D by one frame to process window events and present the scene."""
        if not self._is_initialized or self._base is None:
            return
        self._base.taskMgr.step()


class RenderingServiceImpl(RenderOutputPort):
    """Core RenderOutputPort implementation for rendering SceneCommand streams."""
    
    def __init__(self, window_adapter_factory: Callable[[], Panda3DWindowAdapter] | None = None):
        super().__init__()
        self._expected_contract_version = EXPECTED_CONTRACT_VERSION
        self._window_adapter_factory = window_adapter_factory or Panda3DWindowAdapter
        self._window_adapter = self._window_adapter_factory()
        # Material cache keyed by interaction state.
        self._material_cache: Dict[str, Material] = self._init_materials()
        self._status: str = LIFECYCLE_STOPPED
        self._errors: List[Dict[str, Any]] = []
        self._last_command_ts: Optional[int] = None
        self._scene_root: Optional[NodePath] = None
        self._object_cache: Dict[str, NodePath] = {}
        self._object_initial_states: Dict[str, ObjectInitialState] = {}
        self._executed_command_ids: Set[str] = set()
        self._latest_frame_id: Optional[int] = None
        self._pending_commands: List[SceneCommand] = []
        self._is_resetting: bool = False
        self._last_pose_log_ts: Optional[int] = None
        self._suppressed_pose_logs: int = 0
        self._metrics = RenderingMetrics()
        # For storing gesture data
        self._last_gesture_packet = None
        self._last_fps = 0.0
        # For storing coordinate data
        self._last_world_norm_pos = (0.0, 0.0, 0.0)
        self._last_scene_pos = (0.0, 0.0, 0.0)
        # FPS calculation related
        self._frame_times = []
        self._frame_time_window = 1.0  # 1秒窗口
        # Reuse live_preview drawing logic
        self._colors = OverlayColors()
        # Camera preview related
        self._camera_frame = None
        self._last_observation = None
        self._last_packet = None
        self._camera_texture = None
        self._camera_preview_node = None
        self._camera_preview_enabled = False
        self._last_camera_update_time = 0
        self._camera_update_interval = 0.033  # 30fps
    
    def _init_materials(self) -> Dict[str, Material]:
        """Initialize materials for each interaction state."""
        material_cache = {}
        
        # 1. idle material: gray and opaque.
        idle_mat = Material()
        idle_mat.setAmbient(Vec4(0.5, 0.5, 0.5, 1.0))  # Ambient reflection, alpha=1 for full opacity.
        idle_mat.setDiffuse(Vec4(0.5, 0.5, 0.5, 1.0))  # Diffuse reflection, alpha=1 for full opacity.
        idle_mat.setSpecular(Vec4(0.1, 0.1, 0.1, 1.0)) # Specular highlight, alpha=1 for full opacity.
        idle_mat.setShininess(5.0)                     # Highlight intensity.
        material_cache["idle"] = idle_mat
        
        # 2. hover material: blue and semi-transparent.
        hover_mat = Material()
        hover_mat.setAmbient(Vec4(0.0, 0.0, 0.8, 0.7))  # Ambient reflection, alpha=0.7 for semi-transparency.
        hover_mat.setDiffuse(Vec4(0.0, 0.0, 0.8, 0.7))  # Diffuse reflection, alpha=0.7 for semi-transparency.
        hover_mat.setSpecular(Vec4(0.2, 0.2, 0.8, 0.7)) # Specular highlight, alpha=0.7 for semi-transparency.
        hover_mat.setShininess(10.0)
        material_cache["hover"] = hover_mat
        
        # 3. grabbed material: red and emphasized.
        grabbed_mat = Material()
        grabbed_mat.setAmbient(Vec4(0.8, 0.0, 0.0, 0.9))  # Ambient reflection, alpha=0.9 for slight transparency.
        grabbed_mat.setDiffuse(Vec4(0.8, 0.0, 0.0, 0.9))  # Diffuse reflection, alpha=0.9 for slight transparency.
        grabbed_mat.setSpecular(Vec4(0.8, 0.2, 0.2, 0.9)) # Specular highlight, alpha=0.9 for slight transparency.
        grabbed_mat.setShininess(15.0)
        material_cache["grabbed"] = grabbed_mat
        
        return material_cache

    @staticmethod
    def _world_norm_to_scene_pos(position: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        """Map contract world_norm axes into Panda3D scene axes.

        Contract world_norm:
        - x: right
        - y: up
        - z: toward user/camera

        Panda3D scene axes:
        - x: right
        - y: forward/depth
        - z: up
        """
        # Adjust scale factor to match hand gesture movement range
        scale_factor = 4.0
        x, y, z = (float(value) for value in position)
        return (x * scale_factor, z * scale_factor, y * scale_factor)
    
    def start(self) -> None:
        """Start the module and initialize the environment into RUNNING or DEGRADED."""
        if self._status == LIFECYCLE_RUNNING:
            return None
        
        self._status = LIFECYCLE_INITIALIZING
        self._reset_runtime_state()
        self._errors = []
        self._metrics = RenderingMetrics()
        self._window_adapter = self._window_adapter_factory()
        
        try:
            # Initialize window/camera/lights
            self._window_adapter.init_window()
            self._window_adapter.config_camera_for_world_norm()
            self._window_adapter.create_base_lights()
            base = self._window_adapter.get_base()
            self._status_frame = DirectFrame(
                parent=base.pixel2d,
                pos=(12, 0, -12),
                frameSize=(0, 512, -288, 0),
                frameColor=(0.0, 0.0, 0.0, 0.9),
                relief=1,
                borderWidth=(1, 1),
                color=(60/255, 68/255, 86/255, 1.0)
            )
            # Text control in the panel, compatible with original update logic
            self._status_panel = OnscreenText(
                parent=base.pixel2d,
                pos=(30, -70),
                align=TextNode.ALeft,
                scale=28,
                fg=(1.0, 1.0, 1.0, 1.0),
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
                fps: 0.0""",
                mayChange=True
            )
            
            # Initialize camera preview window (placed below the data panel)
            self._init_camera_preview(base)
            # Create scene root node
            self._scene_root = NodePath("scene_root")
            self._scene_root.reparentTo(self._window_adapter.get_base().render)
            # Switch state to RUNNING
            self._status = LIFECYCLE_RUNNING
            logger.info("Rendering module started successfully, state switched to RUNNING")
        except Exception as e:
            # Initialization failed → DEGRADED
            self._status = LIFECYCLE_DEGRADED
            error = error_entry(
                "rendering.init.failed",
                "Panda3D initialization failed",
                recoverable=False,
                hint="Check if Panda3D is properly installed and your system meets the requirements.",
                details={"error": str(e)}
            )
            self._record_error(error)
            logger.error(f"Module startup failed: {error['message']} (code: {error['code']})")
            raise RuntimeError(f"Module startup failed: {error['message']} (code: {error['code']})") from e
    
    def push(self, command: SceneCommand) -> None:
        """Push a command through the main entry point with fault-tolerant handling."""
        try:
            self._metrics.commands_seen += 1

            if not self._validate_command(command):
                self._metrics.rejected_commands += 1
                if self._status == LIFECYCLE_RUNNING:
                    self._status = LIFECYCLE_DEGRADED
                return

            if self._status in [LIFECYCLE_INITIALIZING, LIFECYCLE_STOPPED]:
                logger.warning(f"Module in {self._status} state, ignoring command (ID: {command.command_id}")
                return
            
            self._last_command_ts = command.timestamp_ms
            
            if self._status == LIFECYCLE_DEGRADED:
                logger.info(f"Module DEGRADED, recording command but not executing (ID: {command.command_id}")
                return
            
            if self._is_resetting:
                self._pending_commands.append(command)
                logger.info(f"During reset, queuing command (ID: {command.command_id}), will execute after reset completes")
                return
            
            if not self._validate_command_effectiveness(command):
                return
            
            # 2. Dispatch by command type.
            command_type = command.command_type
            if command_type == "init_scene":
                self._handle_init_scene(command)
            elif command_type == "set_object_pose":
                self._handle_set_object_pose(command)
            elif command_type == "set_object_state":
                self._handle_set_object_state(command)
            elif command_type == "reset_interaction":
                self._handle_reset_interaction(command)
            elif command_type == "heartbeat":
                self._metrics.heartbeats_received += 1
                self._metrics.commands_applied += 1
                logger.info(f"Received heartbeat command, module state: {self._status}")
            else:
                self._record_error(
                    error_entry(
                        "rendering.command_type.unknown",
                        "Unknown command type received",
                        recoverable=True,
                        hint="Emit only supported scene command types.",
                        details={"command_id": command.command_id, "command_type": command_type},
                    )
                )
                logger.warning(f"Unknown command type: {command_type} (ID: {command.command_id}), ignoring")
            
        except Exception as e:
            error = error_entry(
                "rendering.command.validate.failed",
                "Command validation failed",
                recoverable=True,
                hint="Ensure the command has all required fields and correct types.",
                details={"error": str(e), "command_id": getattr(command, "command_id", "unknown")}
            )
            self._record_error(error)
            details_msg = error.get("details") or error.get("message") or str(e)
            if self._status == LIFECYCLE_RUNNING:
                self._status = LIFECYCLE_DEGRADED
                logger.error(f"Command processing failed, module switched to DEGRADED: {details_msg}")
            else:
                logger.warning(f"Command processing failed: {details_msg}")

    def step(self) -> None:
        """Advance the Panda3D event/render loop without leaving the app's main loop."""
        if not self._window_adapter.is_initialized():
            return

        # Calculate FPS
        current_time = time.time()
        self._frame_times.append(current_time)
        # Remove times older than 1 second
        self._frame_times = [t for t in self._frame_times if current_time - t < self._frame_time_window]
        if len(self._frame_times) > 1:
            self._last_fps = (len(self._frame_times) - 1) / (current_time - self._frame_times[0])
        else:
            self._last_fps = 0.0

        # Update data panel
        if hasattr(self, "_status_panel") and self._status_panel is not None:
            self.update_runtime_status(self._last_gesture_packet, self._last_fps)

        # Update camera preview
        if self._camera_preview_enabled and current_time - self._last_camera_update_time > self._camera_update_interval:
            self._update_camera_preview()
            self._last_camera_update_time = current_time

        if hasattr(self._window_adapter, "step"):
            self._window_adapter.step()
        else:
            base = self._window_adapter.get_base()
            if base is None:
                return
            base.taskMgr.step()
        self._metrics.render_steps += 1
    

    
    def health(self) -> Dict[str, Any]:
        """Return structured health information, including logging-related state."""
        return build_health(
            component="rendering",
            lifecycle_state=self._status,
            errors=self._errors,
            stats={
                "commands_seen": self._metrics.commands_seen,
                "commands_applied": self._metrics.commands_applied,
                "duplicate_commands": self._metrics.duplicate_commands,
                "stale_commands": self._metrics.stale_commands,
                "rejected_commands": self._metrics.rejected_commands,
                "resets_processed": self._metrics.resets_processed,
                "pose_updates": self._metrics.pose_updates,
                "state_updates": self._metrics.state_updates,
                "init_scene_commands": self._metrics.init_scene_commands,
                "heartbeats_received": self._metrics.heartbeats_received,
                "render_steps": self._metrics.render_steps,
                "last_command_ts": self._last_command_ts,
                "window_initialized": self._window_adapter.is_initialized(),
                "executed_command_count": len(self._executed_command_ids),
                "latest_frame_id": self._latest_frame_id,
                "pending_commands_count": len(self._pending_commands)
            }
        )
    
    def stop(self) -> None:
        """Stop the module, release resources, and switch to STOPPED."""
        if self._status == LIFECYCLE_STOPPED:
            logger.info("Module already stopped, no need for repeated operation")
            return None

        self._flush_pose_log_summary()
        
        # Clean up camera preview resources
        self._cleanup_camera_preview()
        
        # Stop task loop, release window
        if self._window_adapter.is_initialized():
            base = self._window_adapter.get_base()
            base.taskMgr.stop()
            base.win.close()
            base.destroy()
        
        self._window_adapter = self._window_adapter_factory()
        self._reset_runtime_state()
        self._status = LIFECYCLE_STOPPED
        logger.info("Rendering module stopped, all resources released")
        return None
    

    
    def _handle_set_object_pose(self, command: SceneCommand) -> None:
        """Handle a set_object_pose command."""
        try:
            # 1. Parse command parameters.
            object_id = command.object_id
            payload = command.payload
            
            # 2. Parse position parameters (support dict{x,y,z} or 3D list/tuple)
            pos_data = payload.get("position", [0.0, 0.0, 0.0])
            if isinstance(pos_data, dict):
                # Handle dict format: {"x": value, "y": value, "z": value}
                if all(key in pos_data for key in ["x", "y", "z"]):
                    pos = [pos_data["x"], pos_data["y"], pos_data["z"]]
                else:
                    self._record_error(
                        error_entry(
                            "rendering.set_object_pose.position.keys_missing",
                            "Position payload is missing required keys",
                            recoverable=True,
                            hint="Provide position as a dict with x, y, z keys.",
                            details={"command_id": command.command_id, "position": pos_data},
                        )
                    )
                    logger.warning(f"set_object_pose command format error: position dict missing required keys (ID: {command.command_id}")
                    return
            elif isinstance(pos_data, (list, tuple)):
                # Handle list/tuple format: [x, y, z]
                pos = list(pos_data)
            else:
                self._record_error(
                    error_entry(
                        "rendering.set_object_pose.position.invalid_type",
                        "Position payload must be a dict or 3-dimensional list",
                        recoverable=True,
                        hint="Provide position as either {x, y, z} or [x, y, z].",
                        details={"command_id": command.command_id, "payload_type": type(pos_data).__name__},
                    )
                )
                logger.warning(f"set_object_pose command format error: position must be dict or 3-dimensional list (ID: {command.command_id}")
                return
            
            # 3. Parse hpr parameters (support dict{h,p,r} or 3D list/tuple)
            hpr_data = payload.get("hpr", [0.0, 0.0, 0.0])
            if isinstance(hpr_data, dict):
                # Handle dict format: {"h": value, "p": value, "r": value}
                if all(key in hpr_data for key in ["h", "p", "r"]):
                    hpr = [hpr_data["h"], hpr_data["p"], hpr_data["r"]]
                else:
                    self._record_error(
                        error_entry(
                            "rendering.set_object_pose.hpr.keys_missing",
                            "Rotation payload is missing required keys",
                            recoverable=True,
                            hint="Provide hpr as a dict with h, p, r keys.",
                            details={"command_id": command.command_id, "hpr": hpr_data},
                        )
                    )
                    logger.warning(f"set_object_pose command format error: hpr dict missing required keys (ID: {command.command_id}")
                    return
            elif isinstance(hpr_data, (list, tuple)):
                # Handle list/tuple format: [h, p, r]
                hpr = list(hpr_data)
            else:
                self._record_error(
                    error_entry(
                        "rendering.set_object_pose.hpr.invalid_type",
                        "Rotation payload must be a dict or 3-dimensional list",
                        recoverable=True,
                        hint="Provide hpr as either {h, p, r} or [h, p, r].",
                        details={"command_id": command.command_id, "payload_type": type(hpr_data).__name__},
                    )
                )
                logger.warning(f"set_object_pose command format error: hpr must be dict or 3-dimensional list (ID: {command.command_id}")
                return
            
            # 4. Validate format and convert to float
            def validate_and_convert_to_float(values):
                if len(values) != 3:
                    return False, []
                try:
                    return True, [float(v) for v in values]
                except (ValueError, TypeError):
                    return False, []
            
            # Validate position
            pos_valid, pos_float = validate_and_convert_to_float(pos)
            if not pos_valid:
                self._record_error(
                    error_entry(
                        "rendering.set_object_pose.position.invalid_value",
                        "Position payload must contain exactly three numeric values",
                        recoverable=True,
                        hint="Provide position as three numeric components.",
                        details={"command_id": command.command_id, "position": pos},
                    )
                )
                logger.warning(f"set_object_pose command format error: position must be 3-dimensional with numeric values (ID: {command.command_id}")
                return
            
            # Validate hpr
            hpr_valid, hpr_float = validate_and_convert_to_float(hpr)
            if not hpr_valid:
                self._record_error(
                    error_entry(
                        "rendering.set_object_pose.hpr.invalid_value",
                        "Rotation payload must contain exactly three numeric values",
                        recoverable=True,
                        hint="Provide hpr as three numeric components.",
                        details={"command_id": command.command_id, "hpr": hpr},
                    )
                )
                logger.warning(f"set_object_pose command format error: hpr must be 3-dimensional with numeric values (ID: {command.command_id}")
                return
            
            # 5. Handle invalid object_id values.
            if object_id not in self._object_cache:
                error = error_entry(
                    "rendering.object.not_found",
                    "Object not found",
                    recoverable=True,
                    hint="Ensure the object ID exists in the scene.",
                    details={"object_id": object_id, "command_id": command.command_id}
                )
                self._record_error(error)
                logger.warning(f"{error['message']}: {error['details']}")
                return
            
            # 6. Validate coordinate ranges and clip to world_norm [-1.0, 1.0].
            clipped_pos = self._clip_coordinate(pos_float)
            clipped_hpr = self._clip_coordinate(hpr_float, rotation=True)  # Rotation is type-checked only and not range-limited.
            scene_pos = self._world_norm_to_scene_pos(clipped_pos)
            
            # 7. Update the object transform.
            obj_np = self._object_cache[object_id]
            obj_np.setPos(*scene_pos)
            obj_np.setHpr(*clipped_hpr)
            self._metrics.pose_updates += 1
            self._metrics.commands_applied += 1
            
            # 保存坐标数据用于显示
            self._last_world_norm_pos = tuple(clipped_pos)
            self._last_scene_pos = scene_pos
            
            # 8. Logging
            if tuple(clipped_pos) != tuple(pos_float):
                logger.warning(f"Coordinate out of world_norm range, automatically clipped: original{pos_float} → clipped{clipped_pos} (ID: {command.command_id}")
                error = error_entry(
                    "rendering.coordinate.out_of_range",
                    "Coordinate out of range",
                    recoverable=True,
                    hint="Ensure coordinates are within the world_norm range [-1.0, 1.0].",
                    details={"object_id": object_id, "original_coordinate": pos_float, "clipped_coordinate": clipped_pos}
                )
                self._record_error(error)
            self._log_pose_update(command, object_id=object_id, position=clipped_pos, rotation=clipped_hpr)
            
        except Exception as e:
            logger.error(f"set_object_pose processing failed (ID: {command.command_id}): {str(e)}")
            self._record_error(
                error_entry(
                    "rendering.set_object_pose.failed",
                    "Failed to update object pose",
                    recoverable=True,
                    hint="Check object existence and pose payload structure.",
                    details={"command_id": command.command_id, "error": str(e)},
                )
            )
    
    def _handle_set_object_state(self, command: SceneCommand) -> None:
        """Handle a set_object_state command."""
        try:
            # 1. Parse command parameters.
            object_id = command.object_id
            payload = command.payload
            state = payload.get("interaction_state", "idle")
            
            # 2. State validation (only process idle/hover/grabbed)
            valid_states = ["idle", "hover", "grabbed"]
            if state not in valid_states:
                self._record_error(
                    error_entry(
                        "rendering.interaction_state.unknown",
                        "Unknown interaction state received",
                        recoverable=True,
                        hint="Emit one of idle, hover, or grabbed.",
                        details={"command_id": command.command_id, "interaction_state": state},
                    )
                )
                logger.warning(f"Unknown interaction_state: {state} (ID: {command.command_id}), defaulting to idle")
                state = "idle"
            
            # 3. Invalid object_id handling
            if object_id not in self._object_cache:
                error = error_entry(
                    "rendering.object.not_found",
                    "Object not found",
                    recoverable=True,
                    hint="Ensure the object ID exists in the scene.",
                    details={"object_id": object_id, "command_id": command.command_id}
                )
                self._record_error(error)
                logger.warning(f"{error['message']}: {error['details']}")
                return
            
            # 4. Update object state
            obj_np = self._object_cache[object_id]
            mat = self._material_cache[state]
            obj_np.setMaterial(mat, 1)  # 1=force replace material
            self._metrics.state_updates += 1
            self._metrics.commands_applied += 1
            
            # 5. Logging
            logger.info(f"Successfully updated object state: ID={object_id}, interaction_state={state} (command ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"set_object_state processing failed (command ID: {command.command_id}): {str(e)}")
            self._record_error(
                error_entry(
                    "rendering.set_object_state.failed",
                    "Failed to update object state",
                    recoverable=True,
                    hint="Check object existence and interaction_state payload structure.",
                    details={"command_id": command.command_id, "error": str(e)},
                )
            )
    
    def _handle_reset_interaction(self, command: SceneCommand) -> None:
        """Handle a reset_interaction command."""
        try:
            # 1. Mark as resetting to prevent concurrency issues
            self._is_resetting = True
            self._metrics.resets_processed += 1
            self._metrics.commands_applied += 1
            logger.info(f"Starting interaction state reset (command ID: {command.command_id}")
            
            # 2. No initialized scene handling
            if not self._object_initial_states:
                logger.warning(f"No initialized object states, skipping reset (command ID: {command.command_id}")
                self._is_resetting = False
                return
            
            # 3. Restore all objects to initial states
            for object_id, init_state in self._object_initial_states.items():
                if object_id not in self._object_cache:
                    logger.warning(f"Object ID {object_id} does not exist, skipping reset")
                    continue
                obj_np = self._object_cache[object_id]
                # Restore position
                obj_np.setPos(*init_state.pos)
                # Restore rotation
                obj_np.setHpr(*init_state.hpr)
                # Restore state (idle)
                obj_np.setMaterial(self._material_cache[init_state.state], 1)
                logger.debug(f"Reset object {object_id} to initial state: pos={init_state.pos}, hpr={init_state.hpr}, state={init_state.state}")
            
            # 4. Clear command cache (deduplication/outdated frames)
            self._executed_command_ids.clear()
            self._latest_frame_id = None
            logger.info("Cleared command_id/frame_id cache, reset completed")
            
            # 5. Set reset flag to False, execute queued commands
            self._is_resetting = False
            pending_count = len(self._pending_commands)
            if pending_count > 0:
                logger.info(f"Executing {pending_count} commands queued during reset")
                for pending_cmd in self._pending_commands:
                    self.push(pending_cmd)
                self._pending_commands.clear()
            
            # 6. Logging
            logger.info(f"Interaction state reset completed (command ID: {command.command_id}), module remains in RUNNING state")
            
        except Exception as e:
            logger.error(f"reset_interaction processing failed (command ID: {command.command_id}): {str(e)}")
            self._is_resetting = False
            self._record_error(
                error_entry(
                    "rendering.reset_interaction.failed",
                    "Failed to reset interaction state",
                    recoverable=True,
                    hint="Check object cache integrity before resetting scene state.",
                    details={"command_id": command.command_id, "error": str(e)},
                )
            )
    
    def _handle_init_scene(self, command: SceneCommand) -> None:
        """Handle an init_scene command."""
        try:
            self._metrics.init_scene_commands += 1
            self._metrics.commands_applied += 1
            # Reset scene
            if self._scene_root is not None and not self._scene_root.isEmpty():
                self._window_adapter.reset_scene(self._scene_root)
                self._object_cache.clear()
                self._object_initial_states.clear()
                logger.info("Duplicate init_scene received, scene cache reset")
            
            # Load cube model
            base = self._window_adapter.get_base()
            cube_model = base.loader.loadModel("box")
            if cube_model.isEmpty():
                raise RuntimeError("Failed to load cube model")
            # Forcefully disable all textures to eliminate noise completely
            cube_model.setTextureOff(1)
            # Set solid color to match idle material
            cube_model.setColor(0.5, 0.5, 0.5, 1.0)
            
            # Parse objects from payload
            objects = command.payload.get("objects", [])
            
            # Validate objects format
            if not isinstance(objects, list):
                self._record_error(
                    error_entry(
                        "rendering.init_scene.objects.invalid_type",
                        "init_scene objects payload must be a list",
                        recoverable=True,
                        hint="Provide init_scene objects as a list of object descriptors.",
                        details={"command_id": command.command_id, "payload_type": type(objects).__name__},
                    )
                )
                logger.warning(f"init_scene command format error: objects must be a list (ID: {command.command_id}")
                return
            
            # Process each object
            for obj_data in objects:
                # Validate object data format
                if not isinstance(obj_data, dict):
                    self._record_error(
                        error_entry(
                            "rendering.init_scene.object.invalid_type",
                            "init_scene object entry must be a dictionary",
                            recoverable=True,
                            hint="Provide each init_scene object as a dict.",
                            details={"command_id": command.command_id, "object_type": type(obj_data).__name__},
                        )
                    )
                    logger.warning(f"init_scene command format error: object must be a dict (ID: {command.command_id}")
                    continue
                
                # Extract required fields
                object_id = obj_data.get("object_id")
                init_pos_data = obj_data.get("init_pos")
                init_hpr_data = obj_data.get("init_hpr")
                
                # Validate required fields
                if not object_id:
                    self._record_error(
                        error_entry(
                            "rendering.init_scene.object_id.missing",
                            "init_scene object is missing object_id",
                            recoverable=True,
                            hint="Provide a non-empty object_id for each init_scene object.",
                            details={"command_id": command.command_id, "object": obj_data},
                        )
                    )
                    logger.warning(f"init_scene command format error: object missing object_id (ID: {command.command_id}")
                    continue
                
                # Parse init_pos_data (support dict{x,y,z} or 3D list/tuple)
                if isinstance(init_pos_data, dict):
                    # Handle dict format: {"x": value, "y": value, "z": value}
                    if all(key in init_pos_data for key in ["x", "y", "z"]):
                        init_pos = (init_pos_data["x"], init_pos_data["y"], init_pos_data["z"])
                    else:
                        self._record_error(
                            error_entry(
                                "rendering.init_scene.init_pos.keys_missing",
                                "init_scene init_pos is missing required keys",
                                recoverable=True,
                                hint="Provide init_pos as a dict with x, y, z keys.",
                                details={"command_id": command.command_id, "object_id": object_id, "init_pos": init_pos_data},
                            )
                        )
                        logger.warning(f"init_scene command format error: init_pos dict missing required keys (ID: {command.command_id}")
                        continue
                elif isinstance(init_pos_data, (list, tuple)) and len(init_pos_data) == 3:
                    # Handle list/tuple format: [x, y, z]
                    init_pos = tuple(init_pos_data)
                else:
                    self._record_error(
                        error_entry(
                            "rendering.init_scene.init_pos.invalid",
                            "init_scene object is missing a valid init_pos",
                            recoverable=True,
                            hint="Provide init_pos as either {x, y, z} or [x, y, z].",
                            details={"command_id": command.command_id, "object_id": object_id, "init_pos": init_pos_data},
                        )
                    )
                    logger.warning(f"init_scene command format error: object {object_id} missing or invalid init_pos (ID: {command.command_id}")
                    continue
                
                # Parse init_hpr_data (support dict{h,p,r} or 3D list/tuple)
                if isinstance(init_hpr_data, dict):
                    # Handle dict format: {"h": value, "p": value, "r": value}
                    if all(key in init_hpr_data for key in ["h", "p", "r"]):
                        init_hpr = (init_hpr_data["h"], init_hpr_data["p"], init_hpr_data["r"])
                    else:
                        self._record_error(
                            error_entry(
                                "rendering.init_scene.init_hpr.keys_missing",
                                "init_scene init_hpr is missing required keys",
                                recoverable=True,
                                hint="Provide init_hpr as a dict with h, p, r keys.",
                                details={"command_id": command.command_id, "object_id": object_id, "init_hpr": init_hpr_data},
                            )
                        )
                        logger.warning(f"init_scene command format error: init_hpr dict missing required keys (ID: {command.command_id}")
                        continue
                elif isinstance(init_hpr_data, (list, tuple)) and len(init_hpr_data) == 3:
                    # Handle list/tuple format: [h, p, r]
                    init_hpr = tuple(init_hpr_data)
                else:
                    self._record_error(
                        error_entry(
                            "rendering.init_scene.init_hpr.invalid",
                            "init_scene object is missing a valid init_hpr",
                            recoverable=True,
                            hint="Provide init_hpr as either {h, p, r} or [h, p, r].",
                            details={"command_id": command.command_id, "object_id": object_id, "init_hpr": init_hpr_data},
                        )
                    )
                    logger.warning(f"init_scene command format error: object {object_id} missing or invalid init_hpr (ID: {command.command_id}")
                    continue
                
                # Convert to float
                try:
                    init_pos = tuple(float(v) for v in init_pos)
                    init_hpr = tuple(float(v) for v in init_hpr)
                except (ValueError, TypeError):
                    self._record_error(
                        error_entry(
                            "rendering.init_scene.numeric_values.invalid",
                            "init_scene object contains invalid numeric values",
                            recoverable=True,
                            hint="Provide numeric init_pos and init_hpr values.",
                            details={"command_id": command.command_id, "object_id": object_id},
                        )
                    )
                    logger.warning(f"init_scene command format error: object {object_id} has invalid numeric values (ID: {command.command_id}")
                    continue
                
                # Create NodePath
                cube_np = self._scene_root.attachNewNode(object_id)
                cube_model.reparentTo(cube_np)
                
                # Set the initial pose and interaction state.
                scene_init_pos = self._world_norm_to_scene_pos(init_pos)
                cube_np.setPos(*scene_init_pos)
                cube_np.setHpr(*init_hpr)
                cube_np.setMaterial(self._material_cache["idle"], 1)
                cube_np.setScale(0.2)  # Fit within world_norm.
                
                # Cache the object and its initial state.
                self._object_cache[object_id] = cube_np
                self._object_initial_states[object_id] = ObjectInitialState(pos=scene_init_pos, hpr=init_hpr)
                
                logger.info(f"init_scene executed: created object {object_id}, initial state pos={init_pos}, hpr={init_hpr}, state=idle")
            
            # Log if no objects were created
            if not objects:
                self._record_error(
                    error_entry(
                        "rendering.init_scene.objects.empty",
                        "init_scene command received with an empty objects list",
                        recoverable=True,
                        hint="Provide at least one object descriptor when initializing the scene.",
                        details={"command_id": command.command_id},
                    )
                )
                logger.warning(f"init_scene command received with empty objects list (ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"init_scene processing failed (command ID: {command.command_id}): {str(e)}")
            self._record_error(
                error_entry(
                    "rendering.init_scene.failed",
                    "Failed to initialize scene objects",
                    recoverable=True,
                    hint="Check model loading and init_scene payload structure.",
                    details={"command_id": command.command_id, "error": str(e)},
                )
            )
    
    def _validate_command_effectiveness(self, command: SceneCommand) -> bool:
        """Validate command effectiveness with deduplication and stale-frame checks."""
        frame_status = classify_frame(self._latest_frame_id, command.frame_id)
        if command.command_id in self._executed_command_ids or frame_status == "duplicate":
            self._metrics.duplicate_commands += 1
            self._record_error(
                error_entry(
                    "rendering.command.duplicate",
                    "Ignoring duplicate scene command",
                    recoverable=True,
                    hint="Emit each scene command once per frame.",
                    details={"command_id": command.command_id, "frame_id": command.frame_id},
                )
            )
            logger.warning(f"Command ID {command.command_id} already executed, ignoring (deduplication logic)")
            return False

        if frame_status == "stale":
            self._metrics.stale_commands += 1
            self._record_error(
                error_entry(
                    "rendering.command.stale",
                    "Ignoring stale scene command",
                    recoverable=True,
                    hint="Do not replay older scene command frames into the live renderer.",
                    details={"command_id": command.command_id, "frame_id": command.frame_id, "last_frame_id": self._latest_frame_id},
                )
            )
            logger.warning(f"Command ID {command.command_id} frame_id={command.frame_id} outdated (latest={self._latest_frame_id}), ignoring")
            return False

        self._executed_command_ids.add(command.command_id)
        self._latest_frame_id = command.frame_id
        logger.debug(f"Updated latest frame_id: {self._latest_frame_id} (command ID: {command.command_id})")
        
        return True

    def _validate_command(self, command: SceneCommand) -> bool:
        errors = validate_scene_command(command, expected_version=self._expected_contract_version)
        if errors:
            for error in errors:
                self._record_error(error)
            return False

        unknown_keys = [key for key in command.payload if key not in VALID_PAYLOAD_KEYS.get(command.command_type, set())]
        if unknown_keys:
            logger.info(f"Command ID {command.command_id} contains unknown payload fields: {unknown_keys}, ignored (forward compatibility)")

        return True

    def _reset_runtime_state(self) -> None:
        self._last_command_ts = None
        self._scene_root = None
        self._object_cache.clear()
        self._object_initial_states.clear()
        self._executed_command_ids.clear()
        self._latest_frame_id = None
        self._pending_commands.clear()
        self._is_resetting = False
        self._last_pose_log_ts = None
        self._suppressed_pose_logs = 0

    def _record_error(self, error: Dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]



    def update_runtime_status(self, packet=None, fps: float = 0.0) -> None:
        """Update the top-left data panel with externally passed gesture data packets and FPS"""
        if not hasattr(self, "_status_panel") or self._status_panel is None:
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
                f"frame: {getattr(packet, 'frame_id', 0)}",
                f"tracking: {getattr(packet, 'tracking_state', 'idle')}",
                f"pinch: {getattr(packet, 'pinch_state', 'idle')}",
                f"confidence: {getattr(packet, 'confidence', 0.0):.2f}",
                f"pinch_distance: {0.0 if getattr(packet, 'pinch_distance', None) is None else packet.pinch_distance:.3f}",
                f"wrist: ({getattr(packet.wrist, 'x', 0.0):+.2f}, {getattr(packet.wrist, 'y', 0.0):+.2f}, {getattr(packet.wrist, 'z', 0.0):+.2f})",
                f"fps: {fps:.1f}",
                f"world_norm: ({self._last_world_norm_pos[0]:+.2f}, {self._last_world_norm_pos[1]:+.2f}, {self._last_world_norm_pos[2]:+.2f})",
                f"scene_pos: ({self._last_scene_pos[0]:+.2f}, {self._last_scene_pos[1]:+.2f}, {self._last_scene_pos[2]:+.2f})",
            )
        self._status_panel.setText("\n".join(lines))
    
    def update_gesture_data(self, packet) -> None:
        """Update gesture data"""
        self._last_gesture_packet = packet
    
    def update_camera_frame(self, frame, observation=None, packet=None) -> None:
        """Update camera frame data"""
        if frame is not None and self._camera_preview_enabled:
            self._camera_frame = frame
            self._last_observation = observation
            self._last_packet = packet

    def enable_camera_preview(self, enabled: bool = True) -> None:
        """Enable or disable camera preview"""
        self._camera_preview_enabled = enabled
        if not enabled and self._camera_preview_node is not None:
            self._camera_preview_node.removeNode()
            self._camera_preview_node = None

    def _init_camera_preview(self, base) -> None:
        """Initialize camera preview window"""
        try:
            # Create camera preview background panel (placed below the data panel)
            self._camera_preview_frame = DirectFrame(
                parent=base.pixel2d,
                pos=(12, 0, -320),  # 20 pixels below the data panel
                frameSize=(0, 512, -288, 0),  # 512x288 pixel preview window
                frameColor=(0.0, 0.0, 0.0, 0.9),
                relief=1,
                borderWidth=(1, 1),
                color=(20/255, 24/255, 32/255, 1.0)
            )
            
            # Create camera preview title
            self._camera_preview_title = OnscreenText(
                parent=base.pixel2d,
                pos=(30, -330),
                align=TextNode.ALeft,
                scale=20,
                fg=(1.0, 1.0, 1.0, 1.0),
                text="Camera Preview",
                mayChange=False
            )
            
            # Create camera preview status text
            self._camera_preview_status = OnscreenText(
                parent=base.pixel2d,
                pos=(30, -350),
                align=TextNode.ALeft,
                scale=16,
                fg=(0.8, 0.8, 0.8, 1.0),
                text="Camera: Not Connected",
                mayChange=True
            )
            
            # Create camera preview texture and node
            self._camera_texture = Texture("camera_preview")
            self._camera_texture.setup2dTexture(512, 288, Texture.T_unsigned_byte, Texture.F_rgb)
            
            card_maker = CardMaker("camera_preview_card")
            card_maker.setFrame(0, 512, -288, 0)
            
            self._camera_preview_node = base.pixel2d.attachNewNode(card_maker.generate())
            # Set negative scale to flip: -1 for left-right mirror, -1 for up-down flip
            # Adjust position to keep the image in place
            self._camera_preview_node.setScale(-1, 1, -1)
            self._camera_preview_node.setPos(12 + 512, 0, -300 - 288)
            
            # Apply texture
            self._camera_preview_node.setTexture(self._camera_texture)
            
            # Enable preview
            self._camera_preview_enabled = True
            logger.info("Camera preview initialized successfully")
            
        except Exception as e:
            logger.warning(f"Camera preview initialization failed: {str(e)}")
            self._camera_preview_enabled = False
    
    def _update_camera_preview(self) -> None:
        """Update camera preview frame"""
        if self._camera_frame is None or self._camera_preview_node is None:
            return
            
        try:
            # Resize image
            frame = cv2.resize(self._camera_frame, (512, 288))
            
            # If there is gesture data, draw hand skeleton (reuse live_preview logic)
            if self._last_observation is not None:
                frame = self._draw_hand_skeleton(frame, self._last_observation)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update texture
            self._camera_texture.setRamImage(frame_rgb)
            
            # Update status text
            if hasattr(self, "_camera_preview_status"):
                self._camera_preview_status.setText("Camera: Active")
                
        except Exception as e:
            logger.warning(f"Camera preview update failed: {str(e)}")
            if hasattr(self, "_camera_preview_status"):
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
        """Convert landmark coordinates to pixel coordinates"""
        return (int(landmark.x * width), int(landmark.y * height))
    
    def _cleanup_camera_preview(self) -> None:
        """Clean up camera preview resources"""
        if self._camera_preview_node is not None:
            self._camera_preview_node.removeNode()
            self._camera_preview_node = None
        
        if hasattr(self, "_camera_preview_frame"):
            self._camera_preview_frame.destroy()
            
        if hasattr(self, "_camera_preview_title"):
            self._camera_preview_title.destroy()
            
        if hasattr(self, "_camera_preview_status"):
            self._camera_preview_status.destroy()
        
        self._camera_frame = None
        self._last_observation = None
        self._last_packet = None
        self._camera_preview_enabled = False

    def _log_pose_update(
        self,
        command: SceneCommand,
        *,
        object_id: str,
        position: list[float],
        rotation: list[float],
    ) -> None:
        if self._last_pose_log_ts is None:
            self._last_pose_log_ts = command.timestamp_ms
            logger.info(
                f"Updated object pose: ID={object_id}, position={position}, rotation={rotation} "
                f"(ID: {command.command_id})"
            )
            return

        elapsed_ms = command.timestamp_ms - self._last_pose_log_ts
        if elapsed_ms < RENDER_POSE_LOG_DEBOUNCE_MS:
            self._suppressed_pose_logs += 1
            return

        suppressed_count = self._suppressed_pose_logs
        self._suppressed_pose_logs = 0
        self._last_pose_log_ts = command.timestamp_ms
        logger.info(
            f"Updated object pose: ID={object_id}, position={position}, rotation={rotation} "
            f"(ID: {command.command_id}, suppressed_updates={suppressed_count})"
        )

    def _flush_pose_log_summary(self) -> None:
        if self._suppressed_pose_logs <= 0:
            return

        logger.info(f"Suppressed {self._suppressed_pose_logs} repetitive pose update log entries")
        self._suppressed_pose_logs = 0
    
    def _clip_coordinate(self, coord: list, rotation: bool = False) -> list:
        """Clip coordinates automatically when they exceed world_norm [-1.0, 1.0]."""
        if rotation:
            # Rotation values are converted to float but not clipped.
            return [float(v) for v in coord]
        # Position values are clipped to [-1.0, 1.0].
        clipped = []
        for v in coord:
            val = float(v)
            if val < -1.0:
                clipped.append(-1.0)
            elif val > 1.0:
                clipped.append(1.0)
            else:
                clipped.append(val)
        return clipped