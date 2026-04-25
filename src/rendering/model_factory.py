from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

from panda3d.core import Filename, Loader, Material, NodePath


logger = logging.getLogger("rendering_service")


@dataclass(slots=True)
class ModelTemplate:
    """模型模板元数据，注册时定义，一次注册永久复用"""

    shape_id: str
    model_path: str
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    two_sided: bool = False
    preload: bool = False
    use_builtin_materials: bool = True


class ModelResourceFactory:
    """
    通用模型资源工厂
    核心能力：注册一次模型，全场景复用，新增模型仅需一行register代码
    """

    def __init__(self, loader: Loader, auto_scan_dir: str | None = None):
        self._loader = loader
        self._template_registry: Dict[str, ModelTemplate] = {}
        self._model_cache: Dict[str, NodePath] = {}
        self._material_cache: Dict[str, Material] = {}
        self._init_builtin_templates()

        if auto_scan_dir:
            self.auto_scan_and_register(auto_scan_dir)

    def _init_builtin_templates(self):
        """初始化内置基础形状，100%兼容现有代码的shape约定"""
        self.register_template(
            ModelTemplate(
                shape_id="cube",
                model_path="models/box",
                center_offset=(-0.5, -0.5, -0.5),
            )
        )
        self.register_template(
            ModelTemplate(
                shape_id="tile",
                model_path="models/box",
                center_offset=(-0.5, -0.5, -0.5),
            )
        )
        self.register_template(
            ModelTemplate(
                shape_id="pillar",
                model_path="models/cylinder",
                center_offset=(0.0, 0.0, -0.5),
            )
        )
        self.register_template(
            ModelTemplate(
                shape_id="plane",
                model_path="models/plane",
                center_offset=(0.0, 0.0, 0.0),
            )
        )
        self.register_template(
            ModelTemplate(
                shape_id="sphere",
                model_path="models/sphere",
                center_offset=(0.0, 0.0, 0.0),
            )
        )
        self.register_template(
            ModelTemplate(
                shape_id="cylinder",
                model_path="models/cylinder",
                center_offset=(0.0, 0.0, -0.5),
            )
        )
        logger.info("Built-in model templates initialized")

    def set_material_cache(self, material_cache: Dict[str, Material]) -> None:
        """
        【重要】绑定RenderingServiceImpl的材质缓存
        必须在工厂初始化后、创建实例前调用
        """
        self._material_cache = material_cache
        logger.info("材质缓存已绑定到模型工厂")

    def auto_scan_and_register(self, models_dir: str) -> None:
        """
        自动扫描指定文件夹，注册所有 glb/egg/bam 模型
        文件名（不含后缀）自动作为 shape_id
        """
        logger.debug(
            "Auto-scanning custom model directory: path=%s abs=%s exists=%s",
            models_dir,
            os.path.abspath(models_dir),
            os.path.isdir(models_dir),
        )

        if not os.path.isdir(models_dir):
            logger.warning(f"自动扫描文件夹不存在：{models_dir}，跳过自动注册")
            return

        supported_formats = (".glb", ".egg", ".bam")
        registered_count = 0

        for filename in os.listdir(models_dir):
            file_path = os.path.join(models_dir, filename)
            if not os.path.isfile(file_path):
                continue

            if filename.lower().endswith(supported_formats):
                shape_id = os.path.splitext(filename)[0]
                self.register_template(
                    ModelTemplate(
                        shape_id=shape_id,
                        model_path=file_path,
                        center_offset=(0.0, 0.0, 0.0),
                        use_builtin_materials=False,
                    )
                )
                registered_count += 1

        logger.info(f"自动扫描完成，共注册 {registered_count} 个自定义模型")

    def _resolve_template(self, shape_id: str) -> ModelTemplate:
        template = self._template_registry.get(shape_id)
        if template is not None:
            return template

        logger.error(f"Shape ID {shape_id} not registered, fallback to cube")
        return self._template_registry["cube"]

    def register_template(self, template: ModelTemplate) -> None:
        """
        注册新模型模板，【新增自定义模型的核心入口，仅需这一行】
        示例：factory.register_template(ModelTemplate(shape_id="teapot", model_path="models/teapot.glb"))
        """
        if template.shape_id in self._template_registry:
            logger.warning(f"Shape ID {template.shape_id} already registered, overwriting")

        self._template_registry[template.shape_id] = template
        if template.preload:
            self._load_model_template(template.shape_id)
        logger.info(f"Model template registered: {template.shape_id}")

    def _load_model_template(self, shape_id: str) -> Optional[NodePath]:
        """内部方法：懒加载模型模板，仅第一次使用时加载"""
        if shape_id in self._model_cache:
            return self._model_cache[shape_id]

        template = self._resolve_template(shape_id)
        try:
            is_custom_model = os.path.isabs(template.model_path) or template.model_path.startswith("\\\\")

            if is_custom_model:
                model_filename = Filename.fromOsSpecific(template.model_path)
                model_filename.makeAbsolute()
                logger.debug(
                    "Loading custom model: path=%s exists=%s",
                    model_filename,
                    model_filename.exists(),
                )
                model = self._loader.loadModel(model_filename)
            else:
                logger.debug("Loading built-in model: path=%s", template.model_path)
                model = self._loader.loadModel(template.model_path)

            if model.isEmpty():
                raise RuntimeError(f"模型文件 {template.model_path} 无效或为空")

            if template.use_builtin_materials:
                model.setTextureOff(1)
            model.setPos(*template.center_offset)
            if template.two_sided:
                model.setTwoSided(True)

            self._model_cache[shape_id] = model
            logger.info(f"Model template loaded and cached: {shape_id}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {shape_id}: {str(e)}, fallback to cube", exc_info=True)
            return self._load_model_template("cube")

    def uses_builtin_materials(self, shape_id: str) -> bool:
        return self._resolve_template(shape_id).use_builtin_materials

    def create_instance(
        self,
        shape_id: str,
        parent: NodePath,
        object_id: str,
        scale: tuple[float, float, float],
        color: tuple[float, float, float, float],
        interactable: bool,
    ) -> NodePath:
        """
        创建模型实例，自动绑定内置四态材质
        """
        template_model = self._load_model_template(shape_id)
        if template_model is None:
            raise RuntimeError(f"模型 {shape_id} 模板加载失败")
        template = self._resolve_template(shape_id)

        object_np = parent.attachNewNode(object_id)
        visual_np = object_np.attachNewNode(f"{object_id}_visual")

        template_model.copyTo(visual_np)

        object_np.setScale(*scale)
        object_np.setColorScale(*color)
        object_np.setTransparency(1)

        if template.use_builtin_materials and self._material_cache and "idle" in self._material_cache:
            object_np.setMaterial(self._material_cache["idle"], 1)

        object_np.setTag("shape", shape_id)
        object_np.setTag("interactable", "1" if interactable else "0")

        logger.debug(f"模型实例创建成功：{object_id}, shape={shape_id}，材质已绑定")
        return object_np

    def clear_cache(self) -> None:
        """清空模型缓存，场景重置时调用，兼容现有reset逻辑"""
        for model in self._model_cache.values():
            model.removeNode()
        self._model_cache.clear()
        logger.info("Model cache cleared")