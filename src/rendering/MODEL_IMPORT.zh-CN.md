# 自定义模型导入

本文档说明渲染模块当前已经实现的自定义模型导入流程。

它只覆盖当前已经落地的行为：

- 零配置模型发现
- 可选 sidecar JSON 覆盖
- 重名冲突处理
- 内置模型名保留规则

## 快速开始

要使用一个自定义模型：

1. 把模型文件放进 `assets/custom_models`
2. 在场景对象配置里，把文件名去后缀后的值写到 `shape`
3. 如果需要模板级精调，再添加 sidecar JSON 文件

例如：

- 模型文件：`assets/custom_models/teapot.glb`
- 对应 `shape_id`：`teapot`

场景对象示例：

```python
{
    "object_id": "demo_teapot",
    "shape": "teapot",
    "init_pos": {"x": 0.0, "y": -0.08, "z": 0.18},
    "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
    "scale": {"x": 0.2, "y": 0.2, "z": 0.2},
    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "interactable": True,
}
```

## 支持的模型格式

当前自动扫描只支持 `assets/custom_models` 顶层目录中的以下格式：

- `.glb`
- `.egg`
- `.bam`

当前版本不会递归扫描子目录。

## 零配置导入

如果你只把模型文件放进 `assets/custom_models`，系统仍然会自动注册它。

当前自动扫描模型的默认行为：

- `shape_id` = 文件名去后缀后转成小写
- `display_name` = 与 `shape_id` 相同
- `center_offset` = `(0.0, 0.0, 0.0)`
- `default_scale` = `(1.0, 1.0, 1.0)`
- `two_sided` = `false`
- `use_builtin_materials` = `false`

这条路径适合最快速地把模型接进系统。

## 可选 sidecar JSON

如果你希望在不改 Python 代码的情况下调整模型模板参数，可以在模型文件旁边放一个 sidecar JSON。

命名规则：

- 模型文件：`teapot.glb`
- sidecar 文件：`teapot.model.json`

sidecar 文件是可选的。

如果 sidecar 不存在，系统会直接走零配置导入路径。

## 当前支持的 sidecar 字段

当前实现支持以下字段。

### `display_name`

- 类型：字符串
- 是否可选：是
- 默认值：`shape_id`

用于给日志或后续 UI 显示一个更友好的名字。

### `default_scale`

- 类型：长度为 3 的正数数组
- 是否可选：是
- 默认值：`[1.0, 1.0, 1.0]`

这个值会在实例创建时乘到实例缩放上。

### `center_offset`

- 类型：长度为 3 的数字数组
- 是否可选：是
- 默认值：`[0.0, 0.0, 0.0]`

用于修正模型的锚点或视觉中心偏移。

### `two_sided`

- 类型：布尔值
- 是否可选：是
- 默认值：`false`

适用于薄片模型或单面面片需要双面可见的情况。

### `use_builtin_materials`

- 类型：布尔值
- 是否可选：是
- 默认值：`false`

当值为 `false` 时，导入模型默认保留自身材质和贴图表现。

## Sidecar 示例

最小示例：

```json
{
  "default_scale": [0.18, 0.18, 0.18],
  "center_offset": [0.0, 0.0, -0.08]
}
```

当前实现支持的完整示例：

```json
{
  "display_name": "Orange Pyramid",
  "default_scale": [0.15, 0.15, 0.15],
  "center_offset": [0.0, 0.0, 0.0],
  "two_sided": false,
  "use_builtin_materials": false
}
```

## 非法 sidecar 的处理方式

当前实现采用宽容解析。

- sidecar 缺失：忽略
- JSON 非法：只记 warning，模型仍然注册
- 单个字段值非法：忽略该字段，其他合法字段继续生效
- 未知字段：忽略

这样可以避免因为一个配置错误导致整个模型导入失败。

## 重名与冲突规则

当前版本不会自动解决重名。

规则如下：

- `shape_id` 由文件名去后缀后转小写得到
- 如果两个自定义模型文件映射到同一个 `shape_id`，则该 `shape_id` 直接拒绝注册
- 如果自定义模型与内置模型名冲突，也直接拒绝注册
- 冲突的 `shape_id` 不会被注册

例如：

- `teapot.glb` 和 `teapot.egg` 会互相冲突
- `cube.glb` 会和内置 `cube` 冲突

当前内置保留名包括：

- `cube`
- `tile`
- `pillar`
- `plane`
- `sphere`
- `cylinder`

出现冲突时，请直接重命名模型文件后再重试。

## 场景对象配置 与 模型 sidecar 的边界

这两层配置负责不同事情。

模型 sidecar 用于模板级默认参数：

- 模型显示名
- 默认缩放
- 中心偏移
- 双面渲染
- 材质策略

场景对象配置用于实例级行为：

- `object_id`
- `shape`
- `init_pos`
- `init_hpr`
- 实例 `scale`
- `color`
- `interactable`

不要把 sidecar JSON 当成场景对象配置的替代品。

## 常见问题

### 模型已经放进文件夹，但没有显示

请检查：

- 文件是否真的位于 `assets/custom_models`
- 文件后缀是否受支持
- `shape` 是否等于文件名去后缀后的值
- 是否与其他自定义模型或内置模型名发生冲突

### 模型太大或太小

可以调整下面任一项：

- 场景对象里的 `scale`
- sidecar 里的 `default_scale`

如果是模板级默认大小，优先用 `default_scale`；如果只是某一个实例要调整，优先用场景对象的 `scale`。

### 模型中心或锚点看起来不对

请在 sidecar JSON 里增加 `center_offset`。

### 模型只有一面可见

请在 sidecar JSON 里把 `two_sided` 设为 `true`。

### 模型没有保留原本的材质观感

请确认 sidecar 中的 `use_builtin_materials` 为 `false`，或者不写该字段并使用导入模型的默认行为。