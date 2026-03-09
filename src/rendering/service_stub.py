import sys
sys.path.insert(0, "/home/giraffe/CODE1/AeroInteract3D")

from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional, Set

from panda3d.core import (
    WindowProperties, Material, AmbientLight, DirectionalLight,
    PerspectiveLens, LVector3, Vec4, NodePath
)
from direct.showbase.ShowBase import ShowBase

from src.contracts import SceneCommand
from src.ports import RenderOutputPort

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rendering_service")

# 生命周期状态
LIFECYCLE_STATES = {
    "INITIALIZING": "INITIALIZING",
    "RUNNING": "RUNNING",
    "DEGRADED": "DEGRADED",
    "STOPPED": "STOPPED"
}

# 错误代码
ERROR_CODES = {
    "PANDA3D_INIT_FAILED": {
        "code": "E001",
        "message": "Panda3D初始化失败",
        "recoverable": False
    },
    "COMMAND_VALIDATE_FAILED": {
        "code": "E002",
        "message": "命令验证失败",
        "recoverable": True
    },
    "OBJECT_NOT_FOUND": {
        "code": "E003",
        "message": "对象未找到",
        "recoverable": True
    },
    "COORDINATE_OUT_OF_RANGE": {
        "code": "E004",
        "message": "坐标超出范围",
        "recoverable": True
    }
}

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
            logger.info(f"窗口已初始化（{window_size}），跳过重复创建")
            return
        try:
            window_props = WindowProperties()
            window_props.setSize(*window_size)
            window_props.setTitle(window_title)
            # 使用正确的方式设置窗口属性
            self._base = ShowBase()
            self._base.win.requestProperties(window_props)
            self._is_initialized = True
            logger.info(f"窗口初始化成功：尺寸{window_size}，标题{window_title}")
        except Exception as e:
            logger.error(f"窗口初始化失败：{str(e)}")
            raise RuntimeError(f"窗口初始化失败：{str(e)}") from e
    
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
            logger.info("相机配置完成，使用透视相机适配3D场景")
        except Exception as e:
            logger.error(f"相机配置失败：{str(e)}")
            raise RuntimeError(f"相机配置失败：{str(e)}") from e
    
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
            logger.info("基础灯光创建完成")
        except Exception as e:
            logger.error(f"灯光创建失败：{str(e)}")
            raise RuntimeError(f"灯光创建失败：{str(e)}") from e
    
    def get_base(self) -> Optional[ShowBase]:
        return self._base
    
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def reset_scene(self, scene_root: NodePath) -> None:
        """重置场景"""
        if not self._is_initialized:
            raise RuntimeError("窗口未初始化，无法重置场景")
        scene_root.removeChildren()
        logger.info("场景已安全重置（保留窗口/相机/灯光）")


class AeroRenderingService(RenderOutputPort):
    """RenderOutputPort核心实现，处理SceneCommand流并渲染3D场景"""
    
    def __init__(self):
        super().__init__()
        # 生命周期状态
        self._status: str = LIFECYCLE_STATES["INITIALIZING"]
        # 错误列表
        self._errors: List[Dict] = []
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
        self._latest_frame_id: int = 0                # 最新frame_id（防过时）
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
        if self._status == LIFECYCLE_STATES["STOPPED"]:
            logger.warning("模块已停止，无法重复启动")
            return
        
        try:
            # 初始化窗口/相机/灯光
            self._window_adapter.init_window()
            self._window_adapter.config_camera_for_world_norm()
            self._window_adapter.create_base_lights()
            # 创建场景根节点
            self._scene_root = NodePath("scene_root")
            self._scene_root.reparentTo(self._window_adapter.get_base().render)
            # 状态切换为RUNNING
            self._status = LIFECYCLE_STATES["RUNNING"]
            self._errors.clear()
            logger.info("Rendering模块启动成功，状态切换为RUNNING")
        except Exception as e:
            # 初始化失败→DEGRADED
            self._status = LIFECYCLE_STATES["DEGRADED"]
            error = ERROR_CODES["PANDA3D_INIT_FAILED"]
            error["detail"] = str(e)
            error["timestamp"] = int(time.time() * 1000)
            self._errors.append(error)
            logger.error(f"模块启动失败：{error['message']}（code: {error['code']}）")
            raise RuntimeError(f"模块启动失败：{error['message']}（code: {error['code']}）") from e
    
    def push(self, command: SceneCommand) -> None:
        """推送命令：核心入口，包含完整容错处理"""
        # 0. 基础容错：捕获所有格式错误/类型错误
        try:
            # 0.1 校验命令基本结构（字段存在性）
            self._validate_command_structure(command)
            
            # 0.2 状态校验
            if self._status in [LIFECYCLE_STATES["INITIALIZING"], LIFECYCLE_STATES["STOPPED"]]:
                logger.warning(f"模块处于{self._status}状态，忽略命令（ID: {command.command_id}")
                return
            
            # 0.3 记录最后命令时间戳
            self._last_command_ts = command.timestamp_ms
            
            # 0.4 DEGRADED状态：仅记录不执行
            if self._status == LIFECYCLE_STATES["DEGRADED"]:
                logger.info(f"模块DEGRADED，记录命令不执行（ID: {command.command_id}")
                return
            
            # 0.5 重置过程中：暂存命令
            if self._is_resetting:
                self._pending_commands.append(command)
                logger.info(f"重置过程中，暂存命令（ID: {command.command_id}），待重置完成后执行")
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
                logger.info(f"收到心跳命令，模块状态：{self._status}")
            else:
                logger.warning(f"未知命令类型：{command_type}（ID: {command.command_id}），忽略")
            
        except Exception as e:
            # 容错核心：仅记录错误，不崩溃
            error = ERROR_CODES["COMMAND_VALIDATE_FAILED"]
            error["detail"] = str(e)
            error["command_id"] = getattr(command, "command_id", "unknown")
            error["timestamp"] = int(time.time() * 1000)
            self._errors.append(error)
            # 非致命错误→DEGRADED，不终止程序
            if self._status == LIFECYCLE_STATES["RUNNING"]:
                self._status = LIFECYCLE_STATES["DEGRADED"]
                logger.error(f"命令处理失败，模块切换为DEGRADED：{error['detail']}")
            else:
                logger.warning(f"命令处理失败：{error['detail']}")
    
    def health(self) -> Dict:
        """返回结构化健康状态（含日志相关信息）"""
        return {
            "status": self._status,
            "errors": self._errors.copy(),
            "last_command_ts": self._last_command_ts,
            "window_initialized": self._window_adapter.is_initialized(),
            "executed_command_count": len(self._executed_command_ids),
            "latest_frame_id": self._latest_frame_id,
            "pending_commands_count": len(self._pending_commands)
        }
    
    def stop(self) -> None:
        """停止模块：释放所有资源，状态→STOPPED"""
        if self._status == LIFECYCLE_STATES["STOPPED"]:
            logger.info("模块已停止，无需重复操作")
            return
        
        # 停止任务循环，释放窗口
        if self._window_adapter.is_initialized():
            base = self._window_adapter.get_base()
            base.task_mgr.stop()
            base.win.close()
            base.destroy()
        
        # 清空缓存
        self._object_cache.clear()
        self._object_initial_states.clear()
        self._executed_command_ids.clear()
        self._pending_commands.clear()
        # 状态切换
        self._status = LIFECYCLE_STATES["STOPPED"]
        logger.info("Rendering模块已停止，所有资源释放完成")
    
    def _handle_set_object_pose(self, command: SceneCommand) -> None:
        """处理set_object_pose命令"""
        try:
            # 1. 解析命令参数
            object_id = command.object_id
            payload = command.payload
            
            # 2. 解析位置/旋转参数（容错：字段缺失/类型错误）
            pos = payload.get("position", [0.0, 0.0, 0.0])
            hpr = payload.get("hpr", [0.0, 0.0, 0.0])  # 旋转（heading/pitch/roll）
            
            # 3. 格式校验：必须是3维列表/元组
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                logger.warning(f"set_object_pose命令格式错误：position必须是3维列表（ID: {command.command_id}")
                return
            if not isinstance(hpr, (list, tuple)) or len(hpr) != 3:
                logger.warning(f"set_object_pose命令格式错误：hpr必须是3维列表（ID: {command.command_id}")
                return
            
            # 4. 无效object_id处理
            if object_id not in self._object_cache:
                error = ERROR_CODES["OBJECT_NOT_FOUND"]
                error["detail"] = f"物体ID {object_id} 不存在（ID: {command.command_id}）"
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
                logger.warning(f"{error['message']}：{error['detail']}")
                return
            
            # 5. 坐标范围校验+自动裁剪（world_norm [-1.0,1.0]）
            clipped_pos = self._clip_coordinate(pos)
            clipped_hpr = self._clip_coordinate(hpr, rotation=True)  # 旋转无范围限制，仅校验类型
            
            # 6. 更新物体姿态（仅RUNNING状态执行）
            obj_np = self._object_cache[object_id]
            obj_np.setPos(*clipped_pos)
            obj_np.setHpr(*clipped_hpr)
            
            # 7. 日志记录
            if clipped_pos != pos:
                logger.warning(f"坐标超出world_norm范围，已自动裁剪：原{pos} → 裁剪后{clipped_pos}（ID: {command.command_id}")
                error = ERROR_CODES["COORDINATE_OUT_OF_RANGE"]
                error["detail"] = f"object_id={object_id}，原坐标{pos}，裁剪后{clipped_pos}"
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
            logger.info(f"成功更新物体姿态：ID={object_id}，位置={clipped_pos}，旋转={clipped_hpr}（ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"set_object_pose处理失败（ID: {command.command_id}）：{str(e)}")
            raise
    
    def _handle_set_object_state(self, command: SceneCommand) -> None:
        """处理set_object_state命令"""
        try:
            # 1. 解析命令参数
            object_id = command.object_id
            payload = command.payload
            state = payload.get("state", "idle")
            
            # 2. 状态校验（仅处理idle/hover/grabbed）
            valid_states = ["idle", "hover", "grabbed"]
            if state not in valid_states:
                logger.warning(f"未知状态：{state}（ID: {command.command_id}），默认使用idle")
                state = "idle"
            
            # 3. 无效object_id处理
            if object_id not in self._object_cache:
                error = ERROR_CODES["OBJECT_NOT_FOUND"]
                error["detail"] = f"物体ID {object_id} 不存在（ID: {command.command_id}）"
                error["timestamp"] = int(time.time() * 1000)
                self._errors.append(error)
                logger.warning(f"{error['message']}：{error['detail']}")
                return
            
            # 4. 更新物体状态
            obj_np = self._object_cache[object_id]
            mat = self._material_cache[state]
            obj_np.setMaterial(mat, 1)  # 1=强制替换材质
            
            # 5. 日志记录
            logger.info(f"成功更新物体状态：ID={object_id}，状态={state}（命令ID: {command.command_id}")
            
        except Exception as e:
            logger.error(f"set_object_state处理失败（命令ID: {command.command_id}）：{str(e)}")
            raise
    
    def _handle_reset_interaction(self, command: SceneCommand) -> None:
        """处理reset_interaction命令"""
        try:
            # 1. 标记为重置中，防止并发问题
            self._is_resetting = True
            logger.info(f"开始重置交互状态（命令ID: {command.command_id}")
            
            # 2. 无初始化场景处理
            if not self._object_initial_states:
                logger.warning(f"无初始化的物体状态，跳过重置（命令ID: {command.command_id}")
                self._is_resetting = False
                return
            
            # 3. 恢复所有物体到初始状态
            for object_id, init_state in self._object_initial_states.items():
                if object_id not in self._object_cache:
                    logger.warning(f"物体ID {object_id} 不存在，跳过重置")
                    continue
                obj_np = self._object_cache[object_id]
                # 恢复位置
                obj_np.setPos(*init_state.pos)
                # 恢复旋转
                obj_np.setHpr(*init_state.hpr)
                # 恢复状态（idle）
                obj_np.setMaterial(self._material_cache[init_state.state], 1)
                logger.debug(f"重置物体 {object_id} 到初始状态：pos={init_state.pos}，hpr={init_state.hpr}，state={init_state.state}")
            
            # 4. 清空命令缓存（去重/过时帧）
            self._executed_command_ids.clear()
            self._latest_frame_id = 0
            logger.info("清空command_id/frame_id缓存，重置完成")
            
            # 5. 重置标记置为False，执行暂存命令
            self._is_resetting = False
            pending_count = len(self._pending_commands)
            if pending_count > 0:
                logger.info(f"执行重置过程中暂存的{pending_count}条命令")
                for pending_cmd in self._pending_commands:
                    self.push(pending_cmd)
                self._pending_commands.clear()
            
            # 6. 日志记录
            logger.info(f"交互状态重置完成（命令ID: {command.command_id}），模块保持RUNNING状态")
            
        except Exception as e:
            logger.error(f"reset_interaction处理失败（命令ID: {command.command_id}）：{str(e)}")
            self._is_resetting = False
            raise
    
    def _handle_init_scene(self, command: SceneCommand) -> None:
        """处理init_scene命令"""
        object_id = "interact_obj_01"
        try:
            # 重置场景
            if self._scene_root is not None and not self._scene_root.isEmpty():
                self._window_adapter.reset_scene(self._scene_root)
                self._object_cache.clear()
                self._object_initial_states.clear()
                logger.info("重复接收init_scene，已重置场景缓存")
            
            # 创建立方体
            base = self._window_adapter.get_base()
            cube_model = base.loader.loadModel("box")
            if cube_model.isEmpty():
                raise RuntimeError("加载立方体模型失败")
            
            # 创建NodePath
            cube_np = self._scene_root.attachNewNode(object_id)
            cube_model.reparentTo(cube_np)
            
            # 初始状态：pos=(0,0,0)，hpr=(0,0,0)，state=idle
            init_pos = (0.0, 0.0, 0.0)
            init_hpr = (0.0, 0.0, 0.0)
            cube_np.setPos(*init_pos)
            cube_np.setHpr(*init_hpr)
            cube_np.setMaterial(self._material_cache["idle"], 1)
            cube_np.setScale(0.2)  # 适配world_norm
            
            # 缓存物体和初始状态
            self._object_cache[object_id] = cube_np
            self._object_initial_states[object_id] = ObjectInitialState(pos=init_pos, hpr=init_hpr)
            
            logger.info(f"init_scene执行完成：创建物体{object_id}，初始状态pos={init_pos}，hpr={init_hpr}，state=idle")
            
        except Exception as e:
            logger.error(f"init_scene处理失败（命令ID: {command.command_id}）：{str(e)}")
            raise
    
    def _validate_command_effectiveness(self, command: SceneCommand) -> bool:
        """命令有效性校验：去重+过时帧处理"""
        command_id = command.command_id
        frame_id = command.frame_id
        
        # 1. 去重校验：重复command_id直接忽略
        if command_id in self._executed_command_ids:
            logger.warning(f"命令ID {command_id} 已执行，忽略（去重逻辑）")
            return False
        
        # 2. 过时帧校验：frame_id < 最新frame_id 忽略
        if frame_id < self._latest_frame_id:
            logger.warning(f"命令ID {command_id} frame_id={frame_id} 过时（最新={self._latest_frame_id}），忽略")
            return False
        
        # 3. 标记为已执行，更新最新frame_id
        self._executed_command_ids.add(command_id)
        if frame_id > self._latest_frame_id:
            self._latest_frame_id = frame_id
            logger.debug(f"更新最新frame_id：{self._latest_frame_id}（命令ID: {command_id}）")
        
        return True
    
    def _validate_command_structure(self, command: SceneCommand) -> None:
        """命令结构校验（容错核心）"""
        # 1. 检查必填字段
        required_fields = ["command_id", "frame_id", "timestamp_ms", "command_type", "object_id", "payload"]
        for field in required_fields:
            if not hasattr(command, field):
                raise ValueError(f"命令缺少必填字段：{field}")
        
        # 2. 校验字段类型（容错：类型错误转换）
        try:
            # command_id必须是字符串
            if not isinstance(command.command_id, str):
                command.command_id = str(command.command_id)
            # frame_id必须是整数
            if not isinstance(command.frame_id, int):
                command.frame_id = int(command.frame_id)
            # timestamp_ms必须是整数
            if not isinstance(command.timestamp_ms, int):
                command.timestamp_ms = int(command.timestamp_ms)
            # command_type必须是字符串
            if not isinstance(command.command_type, str):
                command.command_type = str(command.command_type)
            # object_id必须是字符串
            if not isinstance(command.object_id, str):
                command.object_id = str(command.object_id)
            # payload必须是字典
            if not isinstance(command.payload, dict):
                command.payload = {}
        except Exception as e:
            raise ValueError(f"命令字段类型错误：{str(e)}")
        
        # 3. 忽略未知payload字段（前向兼容）
        payload_keys = command.payload.keys()
        valid_payload_keys = {
            "init_scene": ["scene_config"],
            "set_object_pose": ["coordinate_space", "position", "hpr"],
            "set_object_state": ["state"],
            "reset_interaction": [],
            "heartbeat": ["ping"]
        }
        current_cmd_type = command.command_type
        if current_cmd_type in valid_payload_keys:
            valid_keys = valid_payload_keys[current_cmd_type]
            unknown_keys = [k for k in payload_keys if k not in valid_keys]
            if unknown_keys:
                logger.info(f"命令ID {command.command_id} 包含未知payload字段：{unknown_keys}，已忽略（前向兼容）")
    
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


# 导出RenderOutputServiceStub作为向后兼容
class RenderOutputServiceStub(RenderOutputPort):
    def start(self) -> None:
        return None

    def push(self, command: SceneCommand) -> None:
        return None

    def health(self) -> dict:
        return {"status": "not_implemented"}

    def stop(self) -> None:
        return None