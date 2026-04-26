from __future__ import annotations

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
from .calibration_store import CalibrationSettingsStore
from .object_visibility_store import ObjectVisibilityStore
from .debug.auto_scaling import AutoScalingManager
from .debug.data_panel import DataPanelManager
from .debug.cam_preview import CameraPreviewManager
from .interaction import VirtualHand
from .model_factory import ModelResourceFactory
from .service_models import ObjectInitialState, ObjectVisualProfile, RenderingMetrics, SceneObjectDescriptor
from .service_ui import RenderingServiceUIMixin
from .ui import CalibrationUIView, HomeUIView, RenderView, RenderingViewState, SettingUIView, TableOverlay, TableOverlayState, TableOverlayUIView, UICalibrationPreviewState, UIGestureInputAdapter, UISettingsState
# Logger configuration should be completed at the application entry point.
logger = logging.getLogger("rendering_service")
VALID_PAYLOAD_KEYS = {
    "init_scene": {"objects"},
    "set_object_pose": {"coordinate_space", "position", "hpr", "scale", "debug"},
    "set_object_state": {"interaction_state"},
    "set_hand_pose": {"coordinate_space", "visible", "points"},
    "reset_interaction": set(),
    "heartbeat": {"interaction_state"},
}
TABLE_INTERACTION_LOCKED_COMMAND_TYPES = frozenset({"set_object_pose", "set_object_state", "reset_interaction"})


class RenderingServiceImpl(RenderingServiceUIMixin, RenderOutputPort):
    """Core RenderOutputPort implementation for rendering SceneCommand stream (integrates all submodules)"""
    
    def __init__(
        self,
        window_adapter_factory: Callable[[], RenderingCoreManager] | None = None,
        *,
        debug_stats_enabled: bool = False,
        position_sensitivity: float = 1.0,
        virtual_hand_config: dict | None = None,
        custom_models_dir: str | None = None,
    ):
        super().__init__()
        # ========== 100%正确的路径计算，无拼写错误 ==========
        import os
        import sys
        # 直接从main.py所在目录获取项目根目录，兼容WSL环境
        if hasattr(sys.modules['__main__'], '__file__'):
            main_file_path = sys.modules['__main__'].__file__
            project_root = os.path.dirname(os.path.abspath(main_file_path))
        else:
            project_root = os.getcwd()
        
        # 拼接默认的自定义模型文件夹绝对路径
        DEFAULT_MODELS_DIR = os.path.join(project_root, "assets", "custom_models")
        # 【修复】正确赋值给self._custom_models_dir，无拼写错误
        self._custom_models_dir = custom_models_dir or DEFAULT_MODELS_DIR
        
        logger.debug(
            "Resolved custom models directory: path=%s exists=%s",
            self._custom_models_dir,
            os.path.isdir(self._custom_models_dir),
        )
        # ========== 路径计算结束 ==========

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
        self._object_interaction_states: Dict[str, str] = {}
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
        self._last_hand_points_world_by_id: Dict[str, Dict[str, tuple[float, float, float]]] = {}
        # Virtual hand configuration
        self._virtual_hand_config = virtual_hand_config or {}
        self._virtual_hands: Dict[str, VirtualHand] = {}
        self._dual_scale_ratio: float = 1.0
        self._dual_scale_active: bool = False
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
        # 初始化模型工厂占位符
        self._model_factory: Optional[ModelResourceFactory] = None
        self._object_visual_profiles: Dict[str, ObjectVisualProfile] = {}
        self._view_state = RenderingViewState()
        self._table_overlay_state = TableOverlayState()
        self._ui_settings = UISettingsState()
        self._calibration_store = CalibrationSettingsStore()
        self._object_visibility_store = ObjectVisibilityStore()
        self._calibration_profile_key = self._calibration_store.current_profile_key()
        self._object_visibility_by_id: Dict[str, bool] = {}
        self._volume_callback: Callable[[float], None] | None = None
        self._last_applied_volume: float | None = None
        self._home_view: Optional[HomeUIView] = None
        self._setting_view: Optional[SettingUIView] = None
        self._calibration_view: Optional[CalibrationUIView] = None
        self._table_overlay_view: Optional[TableOverlayUIView] = None
        self._ui_input_adapter = UIGestureInputAdapter()
        self._table_menu_hold_started_at_ms: int | None = None
        self._table_menu_hold_origin_norm: tuple[float, float] | None = None

    @staticmethod
    def _box_model_center_offset() -> tuple[float, float, float]:
        # Kept as a compatibility shim for tests and external callers.
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
        # Scene commands carry absolute world_norm positions. Scaling them in
        # the renderer distorts authored layouts because distant anchors such as
        # the table root move farther than nearby props.
        return tuple(float(value) for value in position)

    @staticmethod
    def _world_norm_to_scene_scale(scale: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        x, y, z = (float(value) for value in scale)
        return (x, z, y)

    @staticmethod
    def _parse_xyz_dict(payload: dict[str, Any], *, keys: tuple[str, str, str]) -> tuple[float, float, float] | None:
        if not all(key in payload for key in keys):
            return None
        try:
            return tuple(float(payload[key]) for key in keys)
        except (TypeError, ValueError):
            return None

    def _parse_scene_object_descriptor(self, command: SceneCommand, obj_data: dict[str, Any]) -> SceneObjectDescriptor | None:
        object_id = obj_data.get("object_id")
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
            return None

        init_pos_data = obj_data.get("init_pos")
        if isinstance(init_pos_data, dict):
            init_pos = self._parse_xyz_dict(init_pos_data, keys=("x", "y", "z"))
            if init_pos is None:
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
                return None
        elif isinstance(init_pos_data, (list, tuple)) and len(init_pos_data) == 3:
            try:
                init_pos = tuple(float(v) for v in init_pos_data)
            except (TypeError, ValueError):
                init_pos = None
        else:
            init_pos = None
        if init_pos is None:
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
            return None

        init_hpr_data = obj_data.get("init_hpr")
        if isinstance(init_hpr_data, dict):
            init_hpr = self._parse_xyz_dict(init_hpr_data, keys=("h", "p", "r"))
            if init_hpr is None:
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
                return None
        elif isinstance(init_hpr_data, (list, tuple)) and len(init_hpr_data) == 3:
            try:
                init_hpr = tuple(float(v) for v in init_hpr_data)
            except (TypeError, ValueError):
                init_hpr = None
        else:
            init_hpr = None
        if init_hpr is None:
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
            return None

        shape = str(obj_data.get("shape", "cube")).lower()
        logger.info(f"解析到模型shape: {shape}，object_id: {object_id}")


        scale_data = obj_data.get("scale", {"x": 0.2, "y": 0.2, "z": 0.2})
        if isinstance(scale_data, dict):
            scale = self._parse_xyz_dict(scale_data, keys=("x", "y", "z"))
        elif isinstance(scale_data, (list, tuple)) and len(scale_data) == 3:
            try:
                scale = tuple(float(v) for v in scale_data)
            except (TypeError, ValueError):
                scale = None
        else:
            scale = None
        if scale is None or any(value <= 0.0 for value in scale):
            scale = (0.2, 0.2, 0.2)

        color_data = obj_data.get("color", {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0})
        if isinstance(color_data, dict):
            try:
                color = (
                    float(color_data["r"]),
                    float(color_data["g"]),
                    float(color_data["b"]),
                    float(color_data.get("a", 1.0)),
                )
            except (KeyError, TypeError, ValueError):
                color = (0.5, 0.5, 0.5, 1.0)
        else:
            color = (0.5, 0.5, 0.5, 1.0)

        interaction_state = str(obj_data.get("interaction_state", "idle"))
        interactable = bool(obj_data.get("interactable", True))
        collision_surface_y = obj_data.get("collision_surface_y")
        if not isinstance(collision_surface_y, (int, float)):
            collision_surface_y = None
        return SceneObjectDescriptor(
            object_id=object_id,
            init_pos=init_pos,
            init_hpr=init_hpr,
            interaction_state=interaction_state,
            shape=shape,
            scale=scale,
            color=color,
            interactable=interactable,
            collision_surface_y=float(collision_surface_y) if collision_surface_y is not None else None,
        )

    def _create_scene_object(self, descriptor: SceneObjectDescriptor) -> NodePath:
        """
        替换原有硬编码实现，改为工厂模式创建
        入参、返回值完全兼容原有代码，无需修改其他逻辑
        """
        if self._model_factory is None:
            raise RuntimeError("Model factory not initialized, call start() first")
        
        # 先做坐标转换，把world_norm的scale转成Panda3D场景scale
        scene_scale = self._world_norm_to_scene_scale(descriptor.scale)
        
        # 【修复】传入转换后的scene_scale，而不是原始的descriptor.scale
        object_np = self._model_factory.create_instance(
            shape_id=descriptor.shape,
            parent=self._scene_root,
            object_id=descriptor.object_id,
            scale=scene_scale,
            color=descriptor.color,
            interactable=descriptor.interactable
        )
        return object_np

    @staticmethod
    def _state_color_scale(
        base_color: tuple[float, float, float, float],
        state: str,
    ) -> tuple[float, float, float, float]:
        state_multipliers = {
            "idle": (1.0, 1.0, 1.0, 1.0),
            "pending_grab": (1.18, 1.05, 0.62, 0.88),
            "grabbed": (1.0, 0.58, 0.58, 0.92),
            "rotating": (0.62, 1.0, 1.0, 0.9),
        }
        multiplier = state_multipliers.get(state, state_multipliers["idle"])
        return tuple(
            max(0.0, min(1.0, float(channel) * float(scale)))
            for channel, scale in zip(base_color, multiplier)
        )

    def _apply_object_visual_state(self, object_id: str, state: str) -> None:
        obj_np = self._object_cache[object_id]
        profile = self._object_visual_profiles.get(object_id)

        if profile is None or profile.use_builtin_materials:
            obj_np.setMaterial(self._material_cache[state], 1)
            return

        clear_material = getattr(obj_np, "clearMaterial", None)
        if callable(clear_material):
            clear_material()

        set_color_scale = getattr(obj_np, "setColorScale", None)
        if callable(set_color_scale):
            set_color_scale(*self._state_color_scale(profile.base_color, state))
    
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
        self._calibration_store.load_into(self._ui_settings, self._calibration_profile_key)
        self._object_visibility_by_id = self._object_visibility_store.load()
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
                pixel2d = self._rendering_core.get_pixel2d()
                if pixel2d is not None:
                    self._home_view = HomeUIView(
                        pixel2d,
                        self._window_size,
                        self._handle_home_button_activated,
                        display_scale_provider=self._display_scale,
                    )
                    self._setting_view = SettingUIView(
                        pixel2d,
                        self._window_size,
                        self._handle_setting_button_activated,
                        display_scale_provider=self._display_scale,
                    )
                    self._calibration_view = CalibrationUIView(
                        pixel2d,
                        self._window_size,
                        self._handle_calibration_button_activated,
                        display_scale_provider=self._display_scale,
                    )
                    self._table_overlay_view = TableOverlayUIView(
                        pixel2d,
                        self._window_size,
                        self._handle_table_overlay_button_activated,
                        display_scale_provider=self._display_scale,
                    )
                    self._apply_ui_settings_to_views()
                if self._debug_stats_enabled:
                    self._data_panel = DataPanelManager(self._auto_scaling)
                self._camera_preview = CameraPreviewManager(
                    self._auto_scaling,
                    top_margin=(
                        DataPanelManager.camera_preview_top_margin()
                        if self._debug_stats_enabled
                        else CameraPreviewManager.PREVIEW_MARGIN
                    ),
                )
            else:
                logger.info(
                    "Skipping debug overlay initialization: overlay_supported=%s",
                    self._supports_debug_overlay(self._rendering_core),
                )
            
            # 实例化模型工厂，开启自动扫描
            base = self._rendering_core.get_base()
            
            # ========== 新增：把自定义模型文件夹添加到Panda3D的模型搜索路径 ==========
            from panda3d.core import Filename, getModelPath
            model_path = getModelPath()
            # 把我们的文件夹添加到搜索路径最前面
            custom_models_filename = Filename.fromOsSpecific(self._custom_models_dir)
            custom_models_filename.makeAbsolute()
            model_path.prependDirectory(custom_models_filename)
            logger.debug("Panda3D model path updated: %s", model_path)
            # ========== 搜索路径添加结束 ==========
            
            self._model_factory = ModelResourceFactory(
                loader=base.loader,
                auto_scan_dir=self._custom_models_dir
            )
            # 绑定材质缓存
            self._model_factory.set_material_cache(self._material_cache)

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
            self._virtual_hands = {
                "hand-1": self._virtual_hand,
                "hand-2": VirtualHand(
                    base=base,
                    root_np=base.render,
                    config=self._virtual_hand_config,
                ),
            }
            logger.info("Virtual hand initialized successfully with base.render as parent")
            self._register_calibration_shortcuts()
            self._apply_window_brightness()
            self._sync_view_visibility()

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

            if self._is_table_interaction_locked() and command.command_type in TABLE_INTERACTION_LOCKED_COMMAND_TYPES:
                logger.debug(
                    "Suppressing table interaction command while overlay %s is active: %s",
                    self._table_overlay_state.active_overlay.value,
                    command.command_type,
                )
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
                logger.debug("Received heartbeat command, module state: %s", self._status)
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

    def set_volume_callback(self, callback: Callable[[float], None] | None) -> None:
        self._volume_callback = callback
        if callback is None:
            self._last_applied_volume = None
            return
        self._apply_volume_setting(force=True)

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
        if self._home_view:
            self._home_view.update_layout()

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
                "active_view": self.active_view,
                "available_views": [view.value for view in RenderView],
                "active_table_overlay": self.active_table_overlay,
                "available_table_overlays": [overlay.value for overlay in TableOverlay],
                "table_interaction_locked": self._is_table_interaction_locked(),
                "table_menu_hold_ms": 0 if self._table_menu_hold_started_at_ms is None or self._last_gesture_packet is None else max(int(getattr(self._last_gesture_packet, "timestamp_ms", 0)) - self._table_menu_hold_started_at_ms, 0),
                "table_menu_cooldown_until_ms": self._table_overlay_state.trigger_cooldown_until_ms,
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
                "pending_commands_count": len(self._pending_commands),
                "ui_settings": {
                    "data_panel_enabled": self._ui_settings.data_panel_enabled,
                    "cam_preview_enabled": self._ui_settings.cam_preview_enabled,
                    "cursor_scale": self._ui_settings.cursor_scale,
                    "cursor_opacity": self._ui_settings.cursor_opacity,
                    "brightness": self._ui_settings.brightness,
                    "volume": self._ui_settings.volume,
                    "ui_cursor_scale_x": self._ui_settings.ui_cursor_scale_x,
                    "ui_cursor_scale_y": self._ui_settings.ui_cursor_scale_y,
                    "ui_cursor_offset_x": self._ui_settings.ui_cursor_offset_x,
                    "ui_cursor_offset_y": self._ui_settings.ui_cursor_offset_y,
                    "calibration_profile_key": self._calibration_profile_key,
                },
                "object_visibility": dict(sorted(self._object_visibility_by_id.items())),
            }
        )
    
    def stop(self) -> None:
        """Stop the module, release resources, and switch to STOPPED state (original logic preserved)"""
        if self._status == LIFECYCLE_STOPPED:
            logger.info("Module already stopped, no need for repeated operation")
            return None

        self._flush_pose_log_summary()
        
        # Clean up submodules
        if self._home_view:
            self._home_view.destroy()
            self._home_view = None
        if self._setting_view:
            self._setting_view.destroy()
            self._setting_view = None
        if self._calibration_view:
            self._calibration_view.destroy()
            self._calibration_view = None
        if self._table_overlay_view:
            self._table_overlay_view.destroy()
            self._table_overlay_view = None
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
        # Clean up virtual hands
        for virtual_hand in list(getattr(self, "_virtual_hands", {}).values()):
            try:
                virtual_hand.root.removeNode()
            except Exception:
                logger.exception("Failed to remove virtual hand node")
        self._virtual_hands = {}
        if hasattr(self, '_virtual_hand') and self._virtual_hand:
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
            self._update_dual_scale_status_from_pose_payload(payload)

            has_position = "position" in payload
            has_hpr = "hpr" in payload
            has_scale = "scale" in payload

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

            # 3b. Parse scale parameters (support dict{x,y,z} or 3D list/tuple).
            scale: list[float] | None = None
            if has_scale:
                scale_data = payload["scale"]
                if isinstance(scale_data, dict):
                    parsed = self._parse_xyz_dict(scale_data, keys=("x", "y", "z"))
                    if parsed is not None:
                        scale = list(parsed)
                    else:
                        logger.warning(f"set_object_pose scale dict missing required keys (ID: {command.command_id})")
                elif isinstance(scale_data, (list, tuple)) and len(scale_data) == 3:
                    try:
                        scale = [float(v) for v in scale_data]
                    except (TypeError, ValueError):
                        scale = None

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
            if not self._is_object_visible(object_id):
                logger.info("Ignoring set_object_pose for hidden object: %s", object_id)
                return
            raw_scene_pos = getattr(obj_np, "pos", (0.0, 0.0, 0.0))
            current_scene_pos = tuple(raw_scene_pos) if isinstance(raw_scene_pos, (list, tuple)) and len(raw_scene_pos) == 3 else (0.0, 0.0, 0.0)
            raw_hpr = getattr(obj_np, "hpr", (0.0, 0.0, 0.0))
            current_hpr = list(raw_hpr) if isinstance(raw_hpr, (list, tuple)) and len(raw_hpr) == 3 else [0.0, 0.0, 0.0]

            # 6. Convert absolute world coordinates directly into scene space.
            next_pos = list(self._last_world_norm_pos)
            scene_pos = current_scene_pos
            if pos_float is not None:
                next_pos = [float(v) for v in pos_float]
                scaled_pos = self._scale_world_norm_position(next_pos)
                scene_pos = self._world_norm_to_scene_pos(scaled_pos)

            clipped_hpr = current_hpr
            if hpr_float is not None:
                clipped_hpr = [float(v) for v in hpr_float]

            # 7. Update the object transform.
            if pos_float is not None:
                obj_np.setPos(*scene_pos)
            if hpr_float is not None:
                obj_np.setHpr(*clipped_hpr)
            if scale is not None and all(v > 0.0 for v in scale):
                initial_state = self._object_initial_states.get(object_id)
                template_default_scale = (
                    initial_state.template_default_scale if initial_state is not None else (1.0, 1.0, 1.0)
                )
                effective_scale = tuple(component * default for component, default in zip(scale, template_default_scale))
                scene_scale = self._world_norm_to_scene_scale(effective_scale)
                obj_np.setScale(*scene_scale)
            self._metrics.pose_updates += 1
            self._metrics.commands_applied += 1
            
            # Save coordinate data for display
            if self._data_panel and pos_float is not None:
                self._data_panel.update_coordinate_data(tuple(next_pos), scene_pos)
            if pos_float is not None:
                self._last_world_norm_pos = tuple(next_pos)
                self._last_scene_pos = scene_pos
            
            # 8. Logging
            self._log_pose_update(command, object_id=object_id, position=next_pos, rotation=clipped_hpr)
            
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
            if not self._is_object_visible(object_id):
                logger.info("Ignoring set_object_state for hidden object: %s", object_id)
                self._object_interaction_states[object_id] = "idle"
                return
            self._apply_object_visual_state(object_id, state)
            self._object_interaction_states[object_id] = state
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
            if command.object_id == "hand-1":
                # Reset scale-active flag every frame; dual-scale object pose may set it back to true.
                self._set_dual_scale_status(ratio=self._dual_scale_ratio, active=False)

            virtual_hand = self._resolve_virtual_hand(command.object_id)
            if virtual_hand is None:
                return

            visible = bool(payload.get("visible", False))
            if not visible:
                self._last_hand_points_world_by_id[command.object_id] = {}
                self._last_hand_points_world = {}
                virtual_hand.update_points(None)
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
            required_points = ("wrist", "thumb_tip", "index_tip", "anchor")
            optional_points = ("thumb_base", "index_base")
            for point_name in required_points:
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
                scene_point = self._world_norm_to_scene_pos(self._scale_world_norm_position(world_point))
                scene_points[point_name] = PandaVec3(*scene_point)
                cached_points[point_name] = tuple(world_point)

            for point_name in optional_points:
                point = points.get(point_name)
                if not isinstance(point, dict):
                    continue
                world_point = [
                    float(point["x"]),
                    float(point["y"]),
                    float(point["z"]),
                ]
                scene_point = self._world_norm_to_scene_pos(self._scale_world_norm_position(world_point))
                scene_points[point_name] = PandaVec3(*scene_point)
                cached_points[point_name] = tuple(world_point)

            self._last_hand_points_world_by_id[command.object_id] = cached_points
            self._last_hand_points_world = cached_points
            virtual_hand.update_points(scene_points)
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
                # Restore scale on the object transform root, matching live scale updates.
                obj_np.setScale(*init_state.scale)
                # Restore state without overriding visibility.
                restored_state = init_state.state if self._is_object_visible(object_id) else "idle"
                self._object_interaction_states[object_id] = restored_state
                self._apply_object_visibility(object_id)
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
                # 清空模型缓存，避免重复加载
                if self._model_factory:
                    self._model_factory.clear_cache()
                self._object_cache.clear()
                self._object_visual_profiles.clear()
                self._object_initial_states.clear()
                self._object_interaction_states.clear()
                logger.info("Duplicate init_scene received, scene cache reset")
            
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
                descriptor = self._parse_scene_object_descriptor(command, obj_data)
                if descriptor is None:
                    continue

                object_np = self._create_scene_object(descriptor)
                scene_init_pos = self._world_norm_to_scene_pos(
                    self._scale_world_norm_position(descriptor.init_pos)
                )
                object_np.setPos(*scene_init_pos)
                object_np.setHpr(*descriptor.init_hpr)

                self._object_cache[descriptor.object_id] = object_np
                self._object_visual_profiles[descriptor.object_id] = ObjectVisualProfile(
                    base_color=descriptor.color,
                    use_builtin_materials=self._model_factory.uses_builtin_materials(descriptor.shape),
                )
                effective_state = descriptor.interaction_state if descriptor.interaction_state in self._material_cache else "idle"
                self._apply_object_visual_state(descriptor.object_id, effective_state)
                template_default_scale = self._model_factory.get_template_default_scale(descriptor.shape)
                effective_init_scale = self._world_norm_to_scene_scale(
                    tuple(component * default for component, default in zip(descriptor.scale, template_default_scale))
                )
                self._object_initial_states[descriptor.object_id] = ObjectInitialState(
                    pos=scene_init_pos,
                    hpr=descriptor.init_hpr,
                    scale=effective_init_scale,
                    template_default_scale=template_default_scale,
                    state=descriptor.interaction_state if descriptor.interactable else "idle",
                )
                self._object_interaction_states[descriptor.object_id] = effective_state
                self._apply_object_visibility(descriptor.object_id)

                logger.info(
                    "init_scene executed: created object %s shape=%s pos=%s hpr=%s scale=%s interactable=%s",
                    descriptor.object_id,
                    descriptor.shape,
                    descriptor.init_pos,
                    descriptor.init_hpr,
                    descriptor.scale,
                    descriptor.interactable,
                )
            
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
            self._apply_ui_settings_to_views()
            
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

        # Same-frame commands are expected (e.g., hand-1 and hand-2 poses in one frame).
        if frame_status == "duplicate":
            logger.debug(
                "Accepting same-frame command id=%s frame_id=%s",
                command.command_id,
                command.frame_id,
            )

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
        self._view_state.set_active_view(RenderView.HOME)
        self._last_command_ts = None
        self._scene_root = None
        self._object_cache.clear()
        self._object_visual_profiles.clear()
        self._object_initial_states.clear()
        self._object_interaction_states.clear()
        self._executed_command_ids.clear()
        self._latest_frame_id = None
        self._pending_commands.clear()
        self._is_resetting = False
        self._last_pose_log_ts = None
        self._suppressed_pose_logs = 0
        self._last_gesture_packet = None
        self._last_hand_points_world = {}
        self._last_hand_points_world_by_id = {}
        self._dual_scale_ratio = 1.0
        self._dual_scale_active = False
        self._last_fps = 0.0
        self._frame_times = []
        self._last_world_norm_pos = (0.0, 0.0, 0.0)
        self._last_scene_pos = (0.0, 0.0, 0.0)
        self._reset_table_overlay_runtime_state(clear_cooldown=True)

    def _resolve_virtual_hand(self, hand_id: str) -> VirtualHand | None:
        if hand_id in self._virtual_hands:
            return self._virtual_hands[hand_id]
        if hasattr(self, "_virtual_hand") and self._virtual_hand is not None:
            if hand_id == "hand-1":
                self._virtual_hands[hand_id] = self._virtual_hand
                return self._virtual_hand
        return None

    def _set_dual_scale_status(self, *, ratio: float, active: bool) -> None:
        self._dual_scale_ratio = float(ratio)
        self._dual_scale_active = bool(active)
        if self._data_panel is not None:
            self._data_panel.update_scale_status(scale_ratio=self._dual_scale_ratio, scaling_active=self._dual_scale_active)

    def _update_dual_scale_status_from_pose_payload(self, payload: dict[str, Any]) -> None:
        debug_payload = payload.get("debug")
        if not isinstance(debug_payload, dict):
            return None
        dual_scale = debug_payload.get("dual_scale")
        if not isinstance(dual_scale, dict):
            return None
        ratio = dual_scale.get("ratio", self._dual_scale_ratio)
        active = dual_scale.get("active", False)
        if isinstance(ratio, (int, float)):
            self._set_dual_scale_status(ratio=float(ratio), active=bool(active))
        return None

    @staticmethod
    def _supports_debug_overlay(rendering_core: RenderingCoreManager) -> bool:
        get_pixel2d = getattr(rendering_core, "get_pixel2d", None)
        return callable(get_pixel2d)

    def _record_error(self, error: Dict[str, Any]) -> None:
        payload = dict(error)
        payload.setdefault("timestamp", int(time.time() * 1000))
        self._errors.append(payload)
        self._errors = self._errors[-MAX_ERROR_HISTORY:]

    def _build_calibration_preview_state(
        self,
        packet,
        ui_input,
        *,
        window_size: tuple[int, int],
    ) -> UICalibrationPreviewState:
        preview_state = UICalibrationPreviewState(
            mapped_cursor_norm=ui_input.cursor_norm,
            mapped_cursor_pixels=ui_input.cursor_pixels,
            window_size=window_size,
            pinch_state=getattr(packet, "pinch_state", None),
            visible=bool(ui_input.visible),
        )

        if packet is None or getattr(packet, "tracking_state", None) != "tracked":
            return preview_state

        midpoint_x = (float(packet.index_tip.x) + float(packet.thumb_tip.x)) * 0.5
        midpoint_y = (float(packet.index_tip.y) + float(packet.thumb_tip.y)) * 0.5
        source_cursor_norm = (
            (1.0 - midpoint_x) * 0.5,
            (1.0 - midpoint_y) * 0.5,
        )
        source_clamped = not (0.0 <= source_cursor_norm[0] <= 1.0 and 0.0 <= source_cursor_norm[1] <= 1.0)
        adjusted_cursor_norm = (
            (source_cursor_norm[0] - 0.5) * self._ui_settings.ui_cursor_scale_x + 0.5 + self._ui_settings.ui_cursor_offset_x,
            (source_cursor_norm[1] - 0.5) * self._ui_settings.ui_cursor_scale_y + 0.5 + self._ui_settings.ui_cursor_offset_y,
        )
        mapped_clamped = not (0.0 <= adjusted_cursor_norm[0] <= 1.0 and 0.0 <= adjusted_cursor_norm[1] <= 1.0)
        preview_state.camera_midpoint = (midpoint_x, midpoint_y)
        preview_state.source_cursor_norm = source_cursor_norm
        preview_state.source_cursor_pixels = (
            max(0.0, min(1.0, source_cursor_norm[0])) * window_size[0],
            max(0.0, min(1.0, source_cursor_norm[1])) * window_size[1],
        )
        preview_state.source_clamped = source_clamped
        preview_state.mapped_clamped = mapped_clamped
        return preview_state
    
    def update_gesture_data(self, packet) -> None:
        """Update gesture data"""
        self._last_gesture_packet = packet
        window_size = self._window_size()
        ui_input = self._ui_input_adapter.to_ui_input(packet, window_size=window_size)
        calibration_preview = self._build_calibration_preview_state(packet, ui_input, window_size=window_size)
        pinch_state = getattr(packet, "pinch_state", None)
        self._update_table_menu_hold_gate(packet, ui_input)
        self._sync_table_menu_hold_feedback(packet)
        if self._view_state.active_view == RenderView.HOME and self._home_view:
            self._home_view.update_layout()
            self._home_view.update_cursor(ui_input, pinch_state=pinch_state)
        elif self._view_state.active_view == RenderView.SETTING and self._setting_view:
            self._setting_view.update_layout()
            self._setting_view.update_cursor(ui_input, pinch_state=pinch_state)
        elif self._view_state.active_view == RenderView.CALIBRATION and self._calibration_view:
            self._calibration_view.update_layout()
            self._calibration_view.update_calibration_preview(calibration_preview)
            self._calibration_view.update_cursor(ui_input, pinch_state=pinch_state)
        elif (
            self._view_state.active_view == RenderView.TABLE
            and self._table_overlay_state.active_overlay != TableOverlay.NONE
            and self._table_overlay_view
        ):
            self._table_overlay_view.update_layout()
            self._table_overlay_view.update_cursor(ui_input, pinch_state=pinch_state)
    
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
