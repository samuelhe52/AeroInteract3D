from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional, Set, Any

from panda3d.core import (
    WindowProperties, Material, AmbientLight, DirectionalLight,
    PerspectiveLens, Vec4, NodePath
)
from direct.showbase.ShowBase import ShowBase

from src.contracts import SceneCommand
from src.ports import RenderOutputPort
from src.utils.runtime import (
    LIFECYCLE_INITIALIZING, LIFECYCLE_RUNNING, LIFECYCLE_DEGRADED, LIFECYCLE_STOPPED,
    build_health, classify_frame, error_entry
)

# 日志记录器（配置应在应用入口完成）
logger = logging.getLogger("rendering_service")



# 物体初始状态
class ObjectInitialState:
    def __init__(self, pos: tuple, hpr: tuple):
        self.pos = pos
        self.hpr = hpr
        self.state = "idle"


class Panda3DWindowAdapter:
    """Panda3D窗口适配器，处理窗口、相机、灯光的初始化和管理"""
    
    def __init__(self):
        self._base: Optional[ShowBase] = None
        self._is_initialized: bool = False
    
    def init_window(self, window_size: tuple = (800, 600), window_title: str = "AeroInteract3D Rendering") -> None:
        """初始化窗口"""
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
        """配置相机适配world_norm坐标系"""
        if not self._is_initialized:
            raise RuntimeError("窗口未初始化，无法配置相机")
        try:
            # 使用透视相机而不是正交相机，这样更容易看到3D物体
            lens = PerspectiveLens()
            lens.setFov(60)  # 设置视场角
            lens.setNearFar(0.1, 100.0)  # 更合理的近远裁剪面
            self._base.cam.node().setLens(lens)
            
            # 调整相机位置，从合适的角度看向物体
            self._base.cam.setPos(0.0, 3.0, 2.0)  # 从斜上方看
            self._base.cam.lookAt(0.0, 0.0, 0.0)  # 看向原点
            logger.info("Camera configured, using perspective camera for 3D scene")
        except Exception as e:
            logger.error(f"Camera configuration failed: {str(e)}")
            raise RuntimeError(f"Camera configuration failed: {str(e)}") from e
    
    def create_base_lights(self) -> None:
        """创建基础灯光"""
        if not self._is_initialized:
            raise RuntimeError("窗口未初始化，无法创建灯光")
        try:
            # 环境光
            amb_light = AmbientLight("ambient_light")
            amb_light.setColor((0.2, 0.2, 0.2, 1.0))
            amb_light_np = self._base.render.attachNewNode(amb_light)
            self._base.render.setLight(amb_light_np)
            # 方向光
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
        """重置场景"""
        if not self._is_initialized:
            raise RuntimeError("窗口未初始化，无法重置场景")
        scene_root.removeChildren()
        logger.info("Scene reset safely (window/camera/lights preserved)")


class AeroRenderingService(RenderOutputPort):
    """RenderOutputPort核心实现，处理SceneCommand流并渲染3D场景"""
    
    def __init__(self):
        super().__init__()
        # 生命周期状态
        self._status: str = LIFECYCLE_INITIALIZING
        # 错误列表
        self._errors: List[Dict[str, Any]] = []
        # 最后命令时间戳
        self._last_command_ts: int = 0
        # 窗口适配器
        self._window_adapter = Panda3DWindowAdapter()
        # 场景根节点
        self._scene_root: Optional[NodePath] = None
        # 物体缓存：object_id → NodePath
        self._object_cache: Dict[str, NodePath] = {}
        # 物体初始状态缓存（用于reset_interaction）
        self._object_initial_states: Dict[str, ObjectInitialState] = {}
        # 命令去重/过时帧控制
        self._executed_command_ids: Set[str] = set()  # 已执行的command_id（去重）
        self._latest_frame_id: Optional[int] = None   # 最新frame_id（防过时）
        # 材质缓存（状态映射）
        self._material_cache: Dict[str, Material] = self._init_materials()
        # 重置过程中的命令暂存（reset_interaction时使用）
        self._pending_commands: List[SceneCommand] = []
        # 重置标记
        self._is_resetting: bool = False
    
    def _init_materials(self) -> Dict[str, Material]:
        """初始化状态对应的材质"""
        material_cache = {}
        
        # 1. idle材质（灰色，不透明）
        idle_mat = Material()
        idle_mat.setAmbient(Vec4(0.5, 0.5, 0.5, 1.0))  # 环境光反射（alpha=1 不透明）
        idle_mat.setDiffuse(Vec4(0.5, 0.5, 0.5, 1.0))  # 漫反射（alpha=1 不透明）
        idle_mat.setSpecular(Vec4(0.1, 0.1, 0.1, 1.0)) # 高光（alpha=1 不透明）
        idle_mat.setShininess(5.0)                     # 高光强度
        material_cache["idle"] = idle_mat
        
        # 2. hover材质（蓝色，半透明）
        hover_mat = Material()
        hover_mat.setAmbient(Vec4(0.0, 0.0, 0.8, 0.7))  # 环境光反射（alpha=0.7 半透明）
        hover_mat.setDiffuse(Vec4(0.0, 0.0, 0.8, 0.7))  # 漫反射（alpha=0.7 半透明）
        hover_mat.setSpecular(Vec4(0.2, 0.2, 0.8, 0.7)) # 高光（alpha=0.7 半透明）
        hover_mat.setShininess(10.0)
        material_cache["hover"] = hover_mat
        
        # 3. grabbed材质（红色，高亮）
        grabbed_mat = Material()
        grabbed_mat.setAmbient(Vec4(0.8, 0.0, 0.0, 0.9))  # 环境光反射（alpha=0.9 略微透明）
        grabbed_mat.setDiffuse(Vec4(0.8, 0.0, 0.0, 0.9))  # 漫反射（alpha=0.9 略微透明）
        grabbed_mat.setSpecular(Vec4(0.8, 0.2, 0.2, 0.9)) # 高光（alpha=0.9 略微透明）
        grabbed_mat.setShininess(15.0)
        material_cache["grabbed"] = grabbed_mat
        
        return material_cache
    
    def start(self) -> None:
        """启动模块：初始化基础环境，状态→RUNNING/DEGRADED"""
        if self._status == LIFECYCLE_STOPPED:
            logger.warning("Module already stopped, cannot restart")
            return
        
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
            self._errors.clear()
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
            error["timestamp"] = int(time.time() * 1000)
            self._errors.append(error)
            logger.error(f"Module startup failed: {error['message']} (code: {error['code']})")
            raise RuntimeError(f"Module startup failed: {error['message']} (code: {error['code']})") from e
    
    def push(self, command: SceneCommand) -> None:
        """推送命令：核心入口，包含完整容错处理"""
        # 0. 基础容错：捕获所有格式错误/类型错误
        try:
            # 0.1 校验命令基本结构（字段存在性）
            self._validate_command_structure(command)
            
            # 0.2 Status check
            if self._status in [LIFECYCLE_INITIALIZING, LIFECYCLE_STOPPED]:
                logger.warning(f"Module in {self._status} state, ignoring command (ID: {command.command_id}")
                return
            
            # 0.3 Record last command timestamp
            self._last_command_ts = command.timestamp_ms
            
            # 0.4 DEGRADED state: record only, do not execute
            if self._status == LIFECYCLE_DEGRADED:
                logger.info(f"Module DEGRADED, recording command but not executing (ID: {command.command_id}")
                return
            
            # 0.5 During reset: queue command
            if self._is_resetting:
                self._pending_commands.append(command)
                logger.info(f"During reset, queuing command (ID: {command.command_id}), will execute after reset completes")
                return
            
            # 1. 命令有效性校验（去重+过时帧）
            if not self._validate_command_effectiveness(command):
                return
            
            # 2. 按命令类型处理
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
                logger.info(f"Received heartbeat command, module state: {self._status}")
            else:
                logger.warning(f"Unknown command type: {command_type} (ID: {command.command_id}), ignoring")
            
        except Exception as e:
            # 容错核心：仅记录错误，不崩溃
            error = error_entry(
                "rendering.command.validate.failed",
                "Command validation failed",
                recoverable=True,
                hint="Ensure the command has all required fields and correct types.",
                details={"error": str(e), "command_id": getattr(command, "command_id", "unknown")}
            )
            error["timestamp"] = int(time.time() * 1000)
            self._errors.append(error)
            # Non-fatal error → DEGRADED, do not terminate program
            if self._status == LIFECYCLE_RUNNING:
                self._status = LIFECYCLE_DEGRADED
                logger.error(f"Command processing failed, module switched to DEGRADED: {error['detail']}")
            else:
                logger.warning(f"Command processing failed: {error['detail']}")
    
    def health(self) -> Dict[str, Any]:
        """返回结构化健康状态（含日志相关信息）"""
        return build_health(
            component="rendering",
            lifecycle_state=self._status,
            errors=self._errors,
            stats={
                "last_command_ts": self._last_command_ts,
                "window_initialized": self._window_adapter.is_initialized(),
                "executed_command_count": len(self._executed_command_ids),
                "latest_frame_id": self._latest_frame_id,
                "pending_commands_count": len(self._pending_commands)
            }
        )
    
    def stop(self) -> None:
        """停止模块：释放所有资源，状态→STOPPED"""
        if self._status == LIFECYCLE_STOPPED:
            logger.info("Module already stopped, no need for repeated operation")
            return
        
        # Stop task loop, release window
        if self._window_adapter.is_initialized():
            base = self._window_adapter.get_base()
            base.task_mgr.stop()
            base.win.close()
            base.destroy()
        
        # Clear caches
        self._object_cache.clear()
        self._object_initial_states.clear()
        self._executed_command_ids.clear()
        self._pending_commands.clear()
        # Switch state
        self._status = LIFECYCLE_STOPPED
        logger.info("Rendering module stopped, all resources released")
    
    def _handle_set_object_pose(self, command: SceneCommand) -> None:
        """处理set_object_pose命令"""
        try:
            # 1. 解析命令参数
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
            
            # 5. Invalid object_id handling
            if object_id not in self._object_cache:
                error = error_entry(
                    "rendering.object.not_found",
                    "Object not found",
                    recoverable=True,
                    hint="Ensure the object ID exists in the scene.",
                    details={"object_id": object_id, "command_id": command.command_id}
                )
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
                logger.warning(f"{error['message']}: {error['details']}")
                return
            
            # 6. 坐标范围校验+自动裁剪（world_norm [-1.0,1.0]）
            clipped_pos = self._clip_coordinate(pos_float)
            clipped_hpr = self._clip_coordinate(hpr_float, rotation=True)  # 旋转无范围限制，仅校验类型
            
            # 7. 更新物体姿态（仅RUNNING状态执行）
            obj_np = self._object_cache[object_id]
            obj_np.setPos(*clipped_pos)
            obj_np.setHpr(*clipped_hpr)
            
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
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
            logger.info(f"Successfully updated object pose: ID={object_id}, position={clipped_pos}, rotation={clipped_hpr} (ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"set_object_pose processing failed (ID: {command.command_id}): {str(e)}")
            # 仅记录错误，不重新抛出异常，确保模块继续运行
    
    def _handle_set_object_state(self, command: SceneCommand) -> None:
        """处理set_object_state命令"""
        try:
            # 1. 解析命令参数
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
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
                logger.warning(f"{error['message']}: {error['details']}")
                return
            
            # 4. Update object state
            obj_np = self._object_cache[object_id]
            mat = self._material_cache[state]
            obj_np.setMaterial(mat, 1)  # 1=force replace material
            
            # 5. Logging
            logger.info(f"Successfully updated object state: ID={object_id}, interaction_state={state} (command ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"set_object_state processing failed (command ID: {command.command_id}): {str(e)}")
            # 仅记录错误，不重新抛出异常，确保模块继续运行
    
    def _handle_reset_interaction(self, command: SceneCommand) -> None:
        """处理reset_interaction命令"""
        try:
            # 1. Mark as resetting to prevent concurrency issues
            self._is_resetting = True
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
            # 仅记录错误，不重新抛出异常，确保模块继续运行
    
    def _handle_init_scene(self, command: SceneCommand) -> None:
        """处理init_scene命令"""
        try:
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
                
                # Set initial state: pos, hpr, state=idle
                cube_np.setPos(*init_pos)
                cube_np.setHpr(*init_hpr)
                cube_np.setMaterial(self._material_cache["idle"], 1)
                cube_np.setScale(0.2)  # fit world_norm
                
                # Cache object and initial state
                self._object_cache[object_id] = cube_np
                self._object_initial_states[object_id] = ObjectInitialState(pos=init_pos, hpr=init_hpr)
                
                logger.info(f"init_scene executed: created object {object_id}, initial state pos={init_pos}, hpr={init_hpr}, state=idle")
            
            # Log if no objects were created
            if not objects:
                logger.warning(f"init_scene command received with empty objects list (ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"init_scene processing failed (command ID: {command.command_id}): {str(e)}")
            # 仅记录错误，不重新抛出异常，确保模块继续运行
    
    def _validate_command_effectiveness(self, command: SceneCommand) -> bool:
        """命令有效性校验：去重+过时帧处理"""
        command_id = command.command_id
        frame_id = command.frame_id
        
        # 1. Deduplication check: ignore duplicate command_id
        if command_id in self._executed_command_ids:
            logger.warning(f"Command ID {command_id} already executed, ignoring (deduplication logic)")
            return False
        
        # 2. Outdated frame check
        if self._latest_frame_id is not None and frame_id < self._latest_frame_id:
            logger.warning(f"Command ID {command_id} frame_id={frame_id} outdated (latest={self._latest_frame_id}), ignoring")
            return False
        
        # 3. Mark as executed, update latest frame_id
        self._executed_command_ids.add(command_id)
        if self._latest_frame_id is None or frame_id > self._latest_frame_id:
            self._latest_frame_id = frame_id
            logger.debug(f"Updated latest frame_id: {self._latest_frame_id} (command ID: {command_id})")
        
        return True
    
    def _validate_command_structure(self, command: SceneCommand) -> None:
        """命令结构校验（容错核心）"""
        # 1. Check required fields
        required_fields = ["command_id", "frame_id", "timestamp_ms", "command_type", "object_id", "payload"]
        for field in required_fields:
            if not hasattr(command, field):
                raise ValueError(f"Command missing required field: {field}")
        
        # 2. Validate field types (fault tolerance: type error conversion)
        try:
            # command_id must be string
            if not isinstance(command.command_id, str):
                command.command_id = str(command.command_id)
            # frame_id must be integer
            if not isinstance(command.frame_id, int):
                command.frame_id = int(command.frame_id)
            # timestamp_ms must be integer
            if not isinstance(command.timestamp_ms, int):
                command.timestamp_ms = int(command.timestamp_ms)
            # command_type must be string
            if not isinstance(command.command_type, str):
                command.command_type = str(command.command_type)
            # object_id must be string
            if not isinstance(command.object_id, str):
                command.object_id = str(command.object_id)
            # payload must be dictionary
            if not isinstance(command.payload, dict):
                command.payload = {}
        except Exception as e:
            raise ValueError(f"Command field type error: {str(e)}")
        
        # 3. Ignore unknown payload fields (forward compatibility)
        payload_keys = command.payload.keys()
        valid_payload_keys = {
            "init_scene": ["objects"],
            "set_object_pose": ["coordinate_space", "position", "hpr"],
            "set_object_state": ["interaction_state"],
            "reset_interaction": [],
            "heartbeat": ["interaction_state"]
        }
        current_cmd_type = command.command_type
        if current_cmd_type in valid_payload_keys:
            valid_keys = valid_payload_keys[current_cmd_type]
            unknown_keys = [k for k in payload_keys if k not in valid_keys]
            if unknown_keys:
                logger.info(f"Command ID {command.command_id} contains unknown payload fields: {unknown_keys}, ignored (forward compatibility)")
    
    def _clip_coordinate(self, coord: list, rotation: bool = False) -> list:
        """坐标裁剪：超出world_norm[-1.0,1.0]自动裁剪"""
        if rotation:
            # 旋转参数仅转换为float，不裁剪
            return [float(v) for v in coord]
        # 位置参数裁剪到[-1.0,1.0]
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