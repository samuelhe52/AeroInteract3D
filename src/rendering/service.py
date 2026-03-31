from __future__ import annotations

from dataclasses import dataclass
import time
import logging
from typing import List, Dict, Optional, Set, Any, Callable

from panda3d.core import (
    Material, Vec4, NodePath, Vec3 as PandaVec3
)

from src.constants import MAX_ERROR_HISTORY, RENDER_POSE_LOG_DEBOUNCE_MS
from src.contracts import SceneCommand
from src.ports import RenderOutputPort
from src.utils.contracts import EXPECTED_CONTRACT_VERSION, validate_scene_command
from src.utils.runtime import (
    LIFECYCLE_INITIALIZING, LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED, LIFECYCLE_STOPPED,
    build_health, classify_frame, error_entry
)

from .rendering_core import RenderingCoreManager
from .debug.auto_scaling import AutoScalingManager
from .debug.data_panel import DataPanelManager
from .debug.cam_preview import CameraPreviewManager
from .interaction import VirtualHand
# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_service")
VALID_PAYLOAD_KEYS = {
    "init_scene": {"objects"},
    "set_object_pose": {"coordinate_space", "position", "hpr"},
    "set_object_state": {"interaction_state"},
    "set_hand_pose": {"coordinate_space", "visible", "points"},
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
    hand_pose_updates: int = 0
    init_scene_commands: int = 0
    heartbeats_received: int = 0
    render_steps: int = 0


class RenderingServiceImpl(RenderOutputPort):
    """Core RenderOutputPort implementation for rendering SceneCommand stream (integrates all submodules)"""
    
    def __init__(
        self,
        window_adapter_factory: Callable[[], RenderingCoreManager] | None = None,
        *,
        debug_stats_enabled: bool = False,
        position_sensitivity: float = 1.0,
        virtual_hand_config: dict | None = None,
    ):
        super().__init__()
        self._expected_contract_version = EXPECTED_CONTRACT_VERSION
        self._window_adapter_factory = window_adapter_factory or RenderingCoreManager
        self._window_adapter = self._window_adapter_factory()
        self._rendering_core: Optional[RenderingCoreManager] = self._window_adapter
        self._debug_stats_enabled = debug_stats_enabled
        self._position_sensitivity = max(float(position_sensitivity), 0.001)
        self._quit_callback: Callable[[], None] | None = None
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
        # For storing debug-facing gesture data
        self._last_gesture_packet = None
        self._last_observation = None
        self._last_hand_points_world: Dict[str, tuple[float, float, float]] = {}
        # Virtual hand configuration
        self._virtual_hand_config = virtual_hand_config or {}
        self._last_fps = 0.0
        # FPS calculation related
        self._frame_times = []
        self._frame_time_window = 1.0  # 1-second window
        # Camera preview related
        self._last_camera_update_time = 0
        self._camera_update_interval = 0.033  # 30fps
        # Submodules
        self._data_panel: Optional[DataPanelManager] = None
        self._camera_preview: Optional[CameraPreviewManager] = None
        self._auto_scaling: Optional[AutoScalingManager] = None
        self._last_world_norm_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_scene_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def _box_model_center_offset() -> tuple[float, float, float]:
        # Panda3D's built-in "box" model is anchored at a corner, so the visual
        # must be centered under the transform node to make rotations happen
        # around the cube's center.
        return (-0.5, -0.5, -0.5)
    
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
        
        # 2. pending_grab material: warm highlight before pinch.
        pending_grab_mat = Material()
        pending_grab_mat.setAmbient(Vec4(0.85, 0.65, 0.0, 0.78))
        pending_grab_mat.setDiffuse(Vec4(0.85, 0.65, 0.0, 0.78))
        pending_grab_mat.setSpecular(Vec4(0.95, 0.82, 0.2, 0.78))
        pending_grab_mat.setShininess(12.0)
        material_cache["pending_grab"] = pending_grab_mat
        
        # 3. grabbed material: red and emphasized.
        grabbed_mat = Material()
        grabbed_mat.setAmbient(Vec4(0.8, 0.0, 0.0, 0.9))  # Ambient reflection, alpha=0.9 for slight transparency.
        grabbed_mat.setDiffuse(Vec4(0.8, 0.0, 0.0, 0.9))  # Diffuse reflection, alpha=0.9 for slight transparency.
        grabbed_mat.setSpecular(Vec4(0.8, 0.2, 0.2, 0.9)) # Specular highlight, alpha=0.9 for slight transparency.
        grabbed_mat.setShininess(15.0)
        material_cache["grabbed"] = grabbed_mat

        rotating_mat = Material()
        rotating_mat.setAmbient(Vec4(0.0, 0.72, 0.72, 0.85))
        rotating_mat.setDiffuse(Vec4(0.0, 0.72, 0.72, 0.85))
        rotating_mat.setSpecular(Vec4(0.4, 0.95, 0.95, 0.9))
        rotating_mat.setShininess(18.0)
        material_cache["rotating"] = rotating_mat
        
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
        x, y, z = (float(value) for value in position)
        return (x, z, y)

    def _scale_world_norm_position(
        self,
        position: tuple[float, float, float] | list[float],
    ) -> tuple[float, float, float]:
        return tuple(float(value) * self._position_sensitivity for value in position)
    
    def start(self) -> None:
        """Start module and initialize environment to RUNNING or DEGRADED (original logic preserved)"""
        if self._status == LIFECYCLE_RUNNING:
            return None
        
        self._status = LIFECYCLE_INITIALIZING
        self._reset_runtime_state()
        self._errors = []
        self._metrics = RenderingMetrics()
        self._window_adapter = self._window_adapter_factory()
        self._rendering_core = self._window_adapter
        if self._quit_callback is not None:
            self._rendering_core.set_quit_handler(self._quit_callback)
        
        try:
            # Initialize window/camera/lights
            self._rendering_core.init_window()
            self._rendering_core.config_camera_for_world_norm()
            self._rendering_core.create_base_lights()
            
            # Initialize submodules (dependency injection)
            self._auto_scaling = AutoScalingManager(self._rendering_core)
            self._auto_scaling.set_scale_callback(self._handle_scale_change)
            if self._supports_debug_overlay(self._rendering_core):
                if self._debug_stats_enabled:
                    self._data_panel = DataPanelManager(self._auto_scaling)
                camera_top_margin = (
                    DataPanelManager.camera_preview_top_margin()
                    if self._debug_stats_enabled
                    else CameraPreviewManager.PREVIEW_MARGIN
                )
                self._camera_preview = CameraPreviewManager(
                    self._auto_scaling,
                    top_margin=camera_top_margin,
                    show_debug_chrome=self._debug_stats_enabled,
                )
            else:
                logger.info("Rendering adapter does not expose pixel2d; skipping debug overlay initialization")
            
            # Create scene root node
            self._scene_root = NodePath("scene_root")
            self._scene_root.reparentTo(self._rendering_core.get_base().render)
            
            # Initialize virtual hand
            base = self._rendering_core.get_base()
            # 使用base.render作为父节点，确保虚拟手在正确的渲染层级
            self._virtual_hand = VirtualHand(
                base=base, 
                root_np=base.render,
                config=self._virtual_hand_config
            )
            logger.info("Virtual hand initialized successfully with base.render as parent")
            
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
    
    def _handle_scale_change(self, scale: float) -> None:
        """Handle UI scale changes"""
        if self._data_panel:
            self._data_panel.set_ui_scale(scale)
        if self._camera_preview:
            self._camera_preview.set_ui_scale(scale)
    
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
            elif command_type == "set_hand_pose":
                self._handle_set_hand_pose(command)
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

    def set_quit_callback(self, callback: Callable[[], None] | None) -> None:
        self._quit_callback = callback
        if self._rendering_core is not None:
            self._rendering_core.set_quit_handler(callback)

    def step(self) -> None:
        """Advance Panda3D event/rendering loop without leaving application main loop (original logic preserved)"""
        if not self._rendering_core or not self._rendering_core.is_initialized():
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
        if self._data_panel:
            self._data_panel.update_data(self._last_gesture_packet, self._last_fps)

        # Update camera preview
        if self._camera_preview and current_time - self._last_camera_update_time > self._camera_update_interval:
            self._camera_preview.update_preview()
            self._last_camera_update_time = current_time
        
        # Main window size monitoring + auto-scaling logic
        if self._auto_scaling:
            self._auto_scaling.update_window_scale()

        if hasattr(self._rendering_core, "step"):
            self._rendering_core.step()
        else:
            base = self._rendering_core.get_base()
            if base is None:
                return
            base.taskMgr.step()
        self._metrics.render_steps += 1
    

    
    def health(self) -> Dict[str, Any]:
        """Return structured health information, including log-related status"""
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
                "hand_pose_updates": self._metrics.hand_pose_updates,
                "init_scene_commands": self._metrics.init_scene_commands,
                "heartbeats_received": self._metrics.heartbeats_received,
                "render_steps": self._metrics.render_steps,
                "last_command_ts": self._last_command_ts,
                "window_initialized": self._rendering_core.is_initialized() if self._rendering_core else False,
                "executed_command_count": len(self._executed_command_ids),
                "latest_frame_id": self._latest_frame_id,
                "pending_commands_count": len(self._pending_commands)
            }
        )
    
    def stop(self) -> None:
        """Stop the module, release resources, and switch to STOPPED state (original logic preserved)"""
        if self._status == LIFECYCLE_STOPPED:
            logger.info("Module already stopped, no need for repeated operation")
            return None

        self._flush_pose_log_summary()
        
        # Clean up submodules
        if self._camera_preview:
            self._camera_preview.destroy()
        if self._data_panel:
            self._data_panel.destroy()
        
        # Stop task loop, release window
        if self._rendering_core and self._rendering_core.is_initialized():
            base = self._rendering_core.get_base()
            base.taskMgr.stop()
            base.userExit()
        
        self._window_adapter = self._window_adapter_factory()
        self._rendering_core = self._window_adapter
        self._reset_runtime_state()
        # Reset submodules
        self._data_panel = None
        self._camera_preview = None
        self._auto_scaling = None
        # Clean up virtual hand
        if hasattr(self, '_virtual_hand') and self._virtual_hand:
            self._virtual_hand.root.removeNode()
            self._virtual_hand = None
        self._status = LIFECYCLE_STOPPED
        logger.info("Rendering module stopped, all resources released")
        return None
    

    
    def _handle_set_object_pose(self, command: SceneCommand) -> None:
        """Handle a set_object_pose command."""
        try:
            # 1. Parse command parameters.
            object_id = command.object_id
            payload = command.payload

            has_position = "position" in payload
            has_hpr = "hpr" in payload

            # 2. Parse position parameters (support dict{x,y,z} or 3D list/tuple).
            pos: list[float] | None = None
            if has_position:
                pos_data = payload["position"]
                if isinstance(pos_data, dict):
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

            # 3. Parse hpr parameters (support dict{h,p,r} or 3D list/tuple).
            hpr: list[float] | None = None
            if has_hpr:
                hpr_data = payload["hpr"]
                if isinstance(hpr_data, dict):
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
            pos_float: list[float] | None = None
            if pos is not None:
                pos_valid, pos_float = validate_and_convert_to_float(pos)
            else:
                pos_valid = True
            if not pos_valid or pos_float is None and has_position:
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
            hpr_float: list[float] | None = None
            if hpr is not None:
                hpr_valid, hpr_float = validate_and_convert_to_float(hpr)
            else:
                hpr_valid = True
            if not hpr_valid or hpr_float is None and has_hpr:
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
            
            obj_np = self._object_cache[object_id]
            raw_scene_pos = getattr(obj_np, "pos", (0.0, 0.0, 0.0))
            current_scene_pos = tuple(raw_scene_pos) if isinstance(raw_scene_pos, (list, tuple)) and len(raw_scene_pos) == 3 else (0.0, 0.0, 0.0)
            raw_hpr = getattr(obj_np, "hpr", (0.0, 0.0, 0.0))
            current_hpr = list(raw_hpr) if isinstance(raw_hpr, (list, tuple)) and len(raw_hpr) == 3 else [0.0, 0.0, 0.0]

            # 6. Validate coordinate ranges and clip to world_norm [-1.0, 1.0].
            clipped_pos = list(self._last_world_norm_pos)
            scene_pos = current_scene_pos
            if pos_float is not None:
                clipped_pos = self._clip_coordinate(pos_float)
                scaled_pos = self._scale_world_norm_position(clipped_pos)
                scene_pos = self._world_norm_to_scene_pos(scaled_pos)

            clipped_hpr = current_hpr
            if hpr_float is not None:
                clipped_hpr = self._clip_coordinate(hpr_float, rotation=True)  # Rotation is type-checked only and not range-limited.

            # 7. Update the object transform.
            if pos_float is not None:
                obj_np.setPos(*scene_pos)
            if hpr_float is not None:
                obj_np.setHpr(*clipped_hpr)
            self._metrics.pose_updates += 1
            self._metrics.commands_applied += 1
            
            # Save coordinate data for display
            if self._data_panel and pos_float is not None:
                self._data_panel.update_coordinate_data(tuple(clipped_pos), scene_pos)
            if pos_float is not None:
                self._last_world_norm_pos = tuple(clipped_pos)
                self._last_scene_pos = scene_pos
            
            # 8. Logging
            if pos_float is not None and tuple(clipped_pos) != tuple(pos_float):
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
            
            # 2. State validation (only process idle/pending_grab/grabbed/rotating)
            valid_states = ["idle", "pending_grab", "grabbed", "rotating"]
            if state not in valid_states:
                self._record_error(
                    error_entry(
                        "rendering.interaction_state.unknown",
                        "Unknown interaction state received",
                        recoverable=True,
                        hint="Emit one of idle, pending_grab, grabbed, or rotating.",
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

    def _handle_set_hand_pose(self, command: SceneCommand) -> None:
        try:
            payload = command.payload
            if not hasattr(self, "_virtual_hand") or self._virtual_hand is None:
                return

            visible = bool(payload.get("visible", False))
            if not visible:
                self._last_hand_points_world = {}
                self._virtual_hand.update_points(None)
                self._metrics.hand_pose_updates += 1
                self._metrics.commands_applied += 1
                return

            points = payload.get("points")
            if not isinstance(points, dict):
                self._record_error(
                    error_entry(
                        "rendering.set_hand_pose.points.invalid",
                        "Hand pose payload must include a points dictionary",
                        recoverable=True,
                        hint="Provide wrist, thumb_tip, index_tip, and anchor in world_norm.",
                        details={"command_id": command.command_id},
                    )
                )
                return

            scene_points: dict[str, PandaVec3] = {}
            cached_points: dict[str, tuple[float, float, float]] = {}
            for point_name in ("wrist", "thumb_tip", "index_tip", "anchor"):
                point = points.get(point_name)
                if not isinstance(point, dict):
                    self._record_error(
                        error_entry(
                            "rendering.set_hand_pose.point.invalid",
                            "Hand pose point is missing or invalid",
                            recoverable=True,
                            hint="Provide all required hand points as {x, y, z}.",
                            details={"command_id": command.command_id, "point_name": point_name},
                        )
                    )
                    return

                world_point = [
                    float(point["x"]),
                    float(point["y"]),
                    float(point["z"]),
                ]
                clipped = self._clip_coordinate(world_point)
                scene_point = self._world_norm_to_scene_pos(self._scale_world_norm_position(clipped))
                scene_points[point_name] = PandaVec3(*scene_point)
                cached_points[point_name] = tuple(clipped)

            self._last_hand_points_world = cached_points
            self._virtual_hand.update_points(scene_points)
            self._metrics.hand_pose_updates += 1
            self._metrics.commands_applied += 1
        except Exception as e:
            logger.error(f"set_hand_pose processing failed (command ID: {command.command_id}): {str(e)}")
            self._record_error(
                error_entry(
                    "rendering.set_hand_pose.failed",
                    "Failed to update hand pose",
                    recoverable=True,
                    hint="Check hand pose payload structure and coordinate values.",
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
        """Handle init_scene command."""
        try:
            self._metrics.init_scene_commands += 1
            self._metrics.commands_applied += 1
            # Reset scene
            if self._scene_root is not None and not self._scene_root.isEmpty():
                self._rendering_core.reset_scene(self._scene_root)
                self._object_cache.clear()
                self._object_initial_states.clear()
                logger.info("Duplicate init_scene received, scene cache reset")
            
            # Load cube model
            base = self._rendering_core.get_base()
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
                cube_visual_np = cube_np.attachNewNode(f"{object_id}_visual")
                cube_model.reparentTo(cube_visual_np)
                cube_visual_np.setPos(*self._box_model_center_offset())
                
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
        if command.command_id in self._executed_command_ids:
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
        self._last_gesture_packet = None
        self._last_hand_points_world = {}
        self._last_fps = 0.0
        self._frame_times = []
        self._last_world_norm_pos = (0.0, 0.0, 0.0)
        self._last_scene_pos = (0.0, 0.0, 0.0)

    @staticmethod
    def _supports_debug_overlay(rendering_core: RenderingCoreManager) -> bool:
        get_pixel2d = getattr(rendering_core, "get_pixel2d", None)
        return callable(get_pixel2d)

    def _record_error(self, error: Dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]
    
    def update_gesture_data(self, packet) -> None:
        """Update gesture data"""
        self._last_gesture_packet = packet
    
    def update_camera_frame(self, frame, observation=None, packet=None) -> None:
        """Update camera frame data"""
        if self._camera_preview:
            self._camera_preview.update_frame(frame, observation, packet or self._last_gesture_packet)
        # Store observation for virtual hand
        self._last_observation = observation

    def enable_camera_preview(self, enabled: bool = True) -> None:
        """Enable or disable camera preview"""
        if self._camera_preview:
            self._camera_preview.enable_preview(enabled)

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
    
    def _clip_coordinate(self, coord: list[float], rotation: bool = False) -> list[float]:
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
