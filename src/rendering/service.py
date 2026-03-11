from __future__ import annotations

from dataclasses import dataclass
import time
import logging
from typing import List, Dict, Optional, Set, Any, Callable

from panda3d.core import (
    WindowProperties, Material, AmbientLight, DirectionalLight,
    PerspectiveLens, Vec4, NodePath
)
from direct.showbase.ShowBase import ShowBase

from src.contracts import SceneCommand
from src.ports import RenderOutputPort
from src.utils.contracts import EXPECTED_CONTRACT_VERSION, validate_scene_command
from src.utils.runtime import (
    LIFECYCLE_INITIALIZING, LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED, LIFECYCLE_STOPPED,
    build_health, classify_frame, error_entry
)

# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_service")

MAX_ERROR_HISTORY = 10
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


class Panda3DWindowAdapter:
    """Panda3D window adapter for window, camera, and light lifecycle management."""
    
    def __init__(self):
        self._base: Optional[ShowBase] = None
        self._is_initialized: bool = False
    
    def init_window(self, window_size: tuple = (800, 600), window_title: str = "AeroInteract3D Rendering") -> None:
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
            
            # Position the camera for a useful angled view of the objects.
            self._base.cam.setPos(0.0, 3.0, 2.0)  # View from above and at an angle.
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
        scene_root.removeChildren()
        logger.info("Scene reset safely (window/camera/lights preserved)")


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
        self._metrics = RenderingMetrics()
    
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
        
        # Stop task loop, release window
        if self._window_adapter.is_initialized():
            base = self._window_adapter.get_base()
            base.task_mgr.stop()
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
                    logger.warning(f"set_object_pose command format error: position dict missing required keys (ID: {command.command_id}")
                    return
            elif isinstance(pos_data, (list, tuple)):
                # Handle list/tuple format: [x, y, z]
                pos = list(pos_data)
            else:
                logger.warning(f"set_object_pose command format error: position must be dict or 3-dimensional list (ID: {command.command_id}")
                return
            
            # 3. Parse hpr parameters (support dict{h,p,r} or 3D list/tuple)
            hpr_data = payload.get("hpr", [0.0, 0.0, 0.0])
            if isinstance(hpr_data, dict):
                # Handle dict format: {"h": value, "p": value, "r": value}
                if all(key in hpr_data for key in ["h", "p", "r"]):
                    hpr = [hpr_data["h"], hpr_data["p"], hpr_data["r"]]
                else:
                    logger.warning(f"set_object_pose command format error: hpr dict missing required keys (ID: {command.command_id}")
                    return
            elif isinstance(hpr_data, (list, tuple)):
                # Handle list/tuple format: [h, p, r]
                hpr = list(hpr_data)
            else:
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
                logger.warning(f"set_object_pose command format error: position must be 3-dimensional with numeric values (ID: {command.command_id}")
                return
            
            # Validate hpr
            hpr_valid, hpr_float = validate_and_convert_to_float(hpr)
            if not hpr_valid:
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
            
            # 7. Update the object transform.
            obj_np = self._object_cache[object_id]
            obj_np.setPos(*clipped_pos)
            obj_np.setHpr(*clipped_hpr)
            self._metrics.pose_updates += 1
            self._metrics.commands_applied += 1
            
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
            logger.info(f"Successfully updated object pose: ID={object_id}, position={clipped_pos}, rotation={clipped_hpr} (ID: {command.command_id}")
            
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
            
            # Parse objects from payload
            objects = command.payload.get("objects", [])
            
            # Validate objects format
            if not isinstance(objects, list):
                logger.warning(f"init_scene command format error: objects must be a list (ID: {command.command_id}")
                return
            
            # Process each object
            for obj_data in objects:
                # Validate object data format
                if not isinstance(obj_data, dict):
                    logger.warning(f"init_scene command format error: object must be a dict (ID: {command.command_id}")
                    continue
                
                # Extract required fields
                object_id = obj_data.get("object_id")
                init_pos_data = obj_data.get("init_pos")
                init_hpr_data = obj_data.get("init_hpr")
                
                # Validate required fields
                if not object_id:
                    logger.warning(f"init_scene command format error: object missing object_id (ID: {command.command_id}")
                    continue
                
                # Parse init_pos_data (support dict{x,y,z} or 3D list/tuple)
                if isinstance(init_pos_data, dict):
                    # Handle dict format: {"x": value, "y": value, "z": value}
                    if all(key in init_pos_data for key in ["x", "y", "z"]):
                        init_pos = (init_pos_data["x"], init_pos_data["y"], init_pos_data["z"])
                    else:
                        logger.warning(f"init_scene command format error: init_pos dict missing required keys (ID: {command.command_id}")
                        continue
                elif isinstance(init_pos_data, (list, tuple)) and len(init_pos_data) == 3:
                    # Handle list/tuple format: [x, y, z]
                    init_pos = tuple(init_pos_data)
                else:
                    logger.warning(f"init_scene command format error: object {object_id} missing or invalid init_pos (ID: {command.command_id}")
                    continue
                
                # Parse init_hpr_data (support dict{h,p,r} or 3D list/tuple)
                if isinstance(init_hpr_data, dict):
                    # Handle dict format: {"h": value, "p": value, "r": value}
                    if all(key in init_hpr_data for key in ["h", "p", "r"]):
                        init_hpr = (init_hpr_data["h"], init_hpr_data["p"], init_hpr_data["r"])
                    else:
                        logger.warning(f"init_scene command format error: init_hpr dict missing required keys (ID: {command.command_id}")
                        continue
                elif isinstance(init_hpr_data, (list, tuple)) and len(init_hpr_data) == 3:
                    # Handle list/tuple format: [h, p, r]
                    init_hpr = tuple(init_hpr_data)
                else:
                    logger.warning(f"init_scene command format error: object {object_id} missing or invalid init_hpr (ID: {command.command_id}")
                    continue
                
                # Convert to float
                try:
                    init_pos = tuple(float(v) for v in init_pos)
                    init_hpr = tuple(float(v) for v in init_hpr)
                except (ValueError, TypeError):
                    logger.warning(f"init_scene command format error: object {object_id} has invalid numeric values (ID: {command.command_id}")
                    continue
                
                # Create NodePath
                cube_np = self._scene_root.attachNewNode(object_id)
                cube_model.reparentTo(cube_np)
                
                # Set the initial pose and interaction state.
                cube_np.setPos(*init_pos)
                cube_np.setHpr(*init_hpr)
                cube_np.setMaterial(self._material_cache["idle"], 1)
                cube_np.setScale(0.2)  # Fit within world_norm.
                
                # Cache the object and its initial state.
                self._object_cache[object_id] = cube_np
                self._object_initial_states[object_id] = ObjectInitialState(pos=init_pos, hpr=init_hpr)
                
                logger.info(f"init_scene executed: created object {object_id}, initial state pos={init_pos}, hpr={init_hpr}, state=idle")
            
            # Log if no objects were created
            if not objects:
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

    def _record_error(self, error: Dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]
    
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
