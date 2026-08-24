from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from panda3d.core import Filename, Loader, Material, NodePath


logger = logging.getLogger("rendering_service")


@dataclass(slots=True)
class ModelTemplate:
    """模型模板元数据，注册时定义，一次注册永久复用"""

    shape_id: str
    model_path: str
    display_name: str | None = None
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    import_hpr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    two_sided: bool = False
    preload: bool = False
    use_builtin_materials: bool = True


class ModelResourceFactory:
    """
    通用模型资源工厂
    核心能力：注册一次模型，全场景复用，新增模型仅需一行register代码
    """

    _SUPPORTED_CUSTOM_MODEL_FORMATS = (".glb", ".egg", ".bam")
    _SIDECAR_SUFFIX = ".model.json"

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

        registered_count = 0
        scanned_files_by_shape: Dict[str, list[str]] = {}
        has_glb_model = False

        for filename in sorted(os.listdir(models_dir)):
            file_path = os.path.join(models_dir, filename)
            if not os.path.isfile(file_path):
                continue

            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in self._SUPPORTED_CUSTOM_MODEL_FORMATS:
                continue

            if file_ext == ".glb":
                has_glb_model = True

            shape_id = self._shape_id_from_model_path(file_path)
            scanned_files_by_shape.setdefault(shape_id, []).append(file_path)

        if has_glb_model:
            self._ensure_gltf_loader_available(models_dir)

        conflicting_shape_ids = {
            shape_id: file_paths
            for shape_id, file_paths in scanned_files_by_shape.items()
            if len(file_paths) > 1
        }
        for shape_id, file_paths in conflicting_shape_ids.items():
            logger.error(
                "Custom model shape_id conflict: %s -> %s. This shape_id will not be registered.",
                shape_id,
                file_paths,
            )

        for shape_id, file_paths in scanned_files_by_shape.items():
            if shape_id in conflicting_shape_ids:
                continue
            if shape_id in self._template_registry:
                logger.error(
                    "Custom model shape_id conflict with reserved or existing template: %s -> %s. This shape_id will not be registered.",
                    shape_id,
                    file_paths[0],
                )
                continue

            self.register_template(self._build_template_from_scanned_model(file_paths[0]))
            registered_count += 1

        logger.info(f"自动扫描完成，共注册 {registered_count} 个自定义模型")

    @staticmethod
    def _ensure_gltf_loader_available(models_dir: str) -> None:
        try:
            importlib.import_module("gltf")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GLB custom models require the 'panda3d-gltf' package. "
                f"Detected .glb assets under {models_dir}, but Panda3D would otherwise fall back "
                "to the Assimp loader, which uses a different up-axis and causes machine-dependent "
                "90-degree import rotations. Install project dependencies with 'uv sync' "
                "(or install 'panda3d-gltf') instead of compensating with sidecar import_hpr flips."
            ) from exc

    @classmethod
    def _shape_id_from_model_path(cls, model_file_path: str) -> str:
        return os.path.splitext(os.path.basename(model_file_path))[0].lower()

    @classmethod
    def _sidecar_path_for_model(cls, model_file_path: str) -> str:
        model_root, _ = os.path.splitext(model_file_path)
        return f"{model_root}{cls._SIDECAR_SUFFIX}"

    @staticmethod
    def _parse_optional_vec3(
        value: Any,
        *,
        field_name: str,
        model_file_path: str,
    ) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            logger.warning("Ignoring invalid %s for custom model %s: %r", field_name, model_file_path, value)
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s for custom model %s: %r", field_name, model_file_path, value)
            return None

    @classmethod
    def _parse_optional_positive_vec3(
        cls,
        value: Any,
        *,
        field_name: str,
        model_file_path: str,
    ) -> tuple[float, float, float] | None:
        parsed = cls._parse_optional_vec3(value, field_name=field_name, model_file_path=model_file_path)
        if parsed is None:
            return None
        if any(component <= 0.0 for component in parsed):
            logger.warning("Ignoring invalid %s for custom model %s: %r", field_name, model_file_path, value)
            return None
        return parsed

    @staticmethod
    def _parse_optional_bool(value: Any, *, field_name: str, model_file_path: str) -> bool | None:
        if isinstance(value, bool):
            return value
        logger.warning("Ignoring invalid %s for custom model %s: %r", field_name, model_file_path, value)
        return None

    @staticmethod
    def _parse_optional_str(value: Any, *, field_name: str, model_file_path: str) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        logger.warning("Ignoring invalid %s for custom model %s: %r", field_name, model_file_path, value)
        return None

    def _load_sidecar_config(self, model_file_path: str) -> dict[str, Any]:
        sidecar_path = self._sidecar_path_for_model(model_file_path)
        if not os.path.isfile(sidecar_path):
            return {}

        try:
            with open(sidecar_path, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse model sidecar for %s: %s", model_file_path, exc)
            return {}

        if not isinstance(payload, dict):
            logger.warning("Ignoring non-object model sidecar for %s: %r", model_file_path, payload)
            return {}

        config: dict[str, Any] = {}
        if "display_name" in payload:
            parsed_value = self._parse_optional_str(
                payload["display_name"],
                field_name="display_name",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["display_name"] = parsed_value
        if "center_offset" in payload:
            parsed_value = self._parse_optional_vec3(
                payload["center_offset"],
                field_name="center_offset",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["center_offset"] = parsed_value
        if "import_hpr" in payload:
            parsed_value = self._parse_optional_vec3(
                payload["import_hpr"],
                field_name="import_hpr",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["import_hpr"] = parsed_value
        if "default_scale" in payload:
            parsed_value = self._parse_optional_positive_vec3(
                payload["default_scale"],
                field_name="default_scale",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["default_scale"] = parsed_value
        if "two_sided" in payload:
            parsed_value = self._parse_optional_bool(
                payload["two_sided"],
                field_name="two_sided",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["two_sided"] = parsed_value
        if "use_builtin_materials" in payload:
            parsed_value = self._parse_optional_bool(
                payload["use_builtin_materials"],
                field_name="use_builtin_materials",
                model_file_path=model_file_path,
            )
            if parsed_value is not None:
                config["use_builtin_materials"] = parsed_value

        logger.info("Loaded custom model sidecar for %s with fields: %s", model_file_path, sorted(config.keys()))
        return config

    def _build_template_from_scanned_model(self, model_file_path: str) -> ModelTemplate:
        sidecar_config = self._load_sidecar_config(model_file_path)
        return ModelTemplate(
            shape_id=self._shape_id_from_model_path(model_file_path),
            model_path=model_file_path,
            display_name=sidecar_config.get("display_name"),
            center_offset=sidecar_config.get("center_offset", (0.0, 0.0, 0.0)),
            import_hpr=sidecar_config.get("import_hpr", (0.0, 0.0, 0.0)),
            default_scale=sidecar_config.get("default_scale", (1.0, 1.0, 1.0)),
            two_sided=sidecar_config.get("two_sided", False),
            use_builtin_materials=sidecar_config.get("use_builtin_materials", False),
        )

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
                    "Loading custom model without Panda3D cache: path=%s exists=%s",
                    model_filename,
                    model_filename.exists(),
                )
                model = self._loader.loadModel(model_filename, noCache=True)
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
            if shape_id == "cube":
                logger.error("Failed to load built-in cube fallback: %s", e, exc_info=True)
                return None
            logger.error(f"Failed to load model {shape_id}: {str(e)}, fallback to cube", exc_info=True)
            return self._load_model_template("cube")

    def uses_builtin_materials(self, shape_id: str) -> bool:
        return self._resolve_template(shape_id).use_builtin_materials

    def get_display_name(self, shape_id: str) -> str | None:
        display_name = self._resolve_template(shape_id).display_name
        if isinstance(display_name, str) and display_name.strip():
            return display_name.strip()
        return None

    def get_template_default_scale(self, shape_id: str) -> tuple[float, float, float]:
        return self._resolve_template(shape_id).default_scale

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
        visual_np.setHpr(*template.import_hpr)

        template_model.copyTo(visual_np)

        effective_scale = tuple(component * default for component, default in zip(scale, template.default_scale))
        object_np.setScale(*effective_scale)
        object_np.setColorScale(*color)
        if color[3] < 1.0:
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
