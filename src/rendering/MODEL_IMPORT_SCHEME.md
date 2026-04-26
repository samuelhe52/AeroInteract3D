# 模型导入方案草案

下面这份方案按 3 个层级来设计，核心目标是不让用户一上来就被配置压住，同时又给后续精调留出干净扩展位。

## 总体原则

- 不写 sidecar JSON 时，模型也必须能直接被扫描和使用。
- 写了 sidecar JSON 时，只覆盖用户明确指定的字段，其他仍走默认值。
- 文档只讲“用户需要关心的字段”，不要把内部实现细节全暴露出去。
- 第一版先优先支持“导入成功并可见”，第二版再补“导入后看起来正确”。

## Level 0：零配置模式

这是最轻的用户路径。

用户只需要做两件事：
1. 把模型文件放进 `assets/custom_models`
2. 在场景对象里把 `shape` 写成文件名去后缀后的值

例如放入：
- `teapot.glb`

场景里写：

```python
{
    "object_id": "my_teapot",
    "shape": "teapot",
    "init_pos": {"x": 0.0, "y": -0.08, "z": 0.18},
    "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
    "scale": {"x": 0.2, "y": 0.2, "z": 0.2},
    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "interactable": True,
}
```

系统默认行为建议：
- 自动扫描 `glb`、`egg`、`bam`
- `shape_id = 文件名去后缀`
- `use_builtin_materials = false`
- `center_offset = (0, 0, 0)`
- `two_sided = false`
- `default_scale = (1, 1, 1)`

适合人群：
- 只是想快速把模型接进来看看能不能跑
- 愿意先在场景对象里手调位置和缩放
- 不关心模型模板级元数据

这一档的优点是门槛最低。  
缺点也明确：模型常常“能出来，但不一定好看”。

## Level 1：最小 sidecar JSON 方案

这是我最推荐的第一版精调方案。字段少，但已经能解决 80% 的用户痛点。

命名规则建议：
- `teapot.glb`
- `teapot.model.json`

推荐暴露字段只有 5 个：

```json
{
  "display_name": "Teapot",
  "default_scale": [0.2, 0.2, 0.2],
  "center_offset": [0.0, 0.0, -0.1],
  "two_sided": false,
  "use_builtin_materials": false
}
```

字段含义：
- `display_name`
作用：给日志、调试面板、未来 UI 列表展示一个更友好的名字  
必要性：低，但很有用

- `default_scale`
作用：给模型一个推荐缩放，减少用户在场景里反复试尺寸  
必要性：高

- `center_offset`
作用：修正模型锚点或重心偏移  
必要性：高

- `two_sided`
作用：解决薄片模型、单面面片被背面裁掉的问题  
必要性：中高

- `use_builtin_materials`
作用：决定是否吃系统内置交互材质，而不是保留模型原材质  
必要性：中

这一档的特点是：
- 用户不需要懂内部 `ModelTemplate`
- 用户只需要知道 5 个最常见问题怎么调
- JSON 仍然足够短，不会有“配置文件比模型还复杂”的感觉

我认为这档是最适合作为第一阶段落地的。

## Level 2：增强 sidecar JSON 方案

这一档是面向更认真使用导入功能的用户，适合后续扩展，但不建议第一版一次性全放出去。

可以支持的字段扩展到下面这些：

```json
{
  "display_name": "Teapot",
  "default_scale": [0.2, 0.2, 0.2],
  "center_offset": [0.0, 0.0, -0.1],
  "two_sided": false,
  "use_builtin_materials": false,
  "preload": false,
  "interactable": true,
  "default_color": [1.0, 1.0, 1.0, 1.0],
  "default_hpr": [0.0, 0.0, 0.0],
  "tags": {
    "category": "prop",
    "source": "custom"
  }
}
```

这里面我建议分成两类。

第一类，比较值得保留：
- `preload`
- `default_color`
- `default_hpr`

第二类，我建议慎重：
- `interactable`
- `tags`

原因是第二类开始接近“场景对象级配置”，而不是“模型模板级配置”。  
一旦把这类字段放进模型 sidecar，后面很容易让“模板配置”和“场景实例配置”边界变混。

所以就算支持，也建议在文档里明确：

- sidecar JSON 负责“模型默认模板”
- 场景对象配置负责“某个实例放在哪里、能不能交互、初始状态是什么”

## Level 3：高级方案，不建议第一版就做

如果以后模型导入变成高频能力，还可以继续扩到更完整的 schema，比如：

```json
{
  "display_name": "Teapot",
  "aliases": ["tea_pot", "pot_tea"],
  "default_scale": [0.2, 0.2, 0.2],
  "center_offset": [0.0, 0.0, -0.1],
  "two_sided": false,
  "use_builtin_materials": false,
  "preload": false,
  "preview": {
    "thumbnail": "teapot.png"
  },
  "interaction": {
    "preferred_radius": 0.18
  },
  "validation": {
    "expected_up_axis": "z"
  }
}
```

这类能力的代价是：
- schema 复杂度明显上升
- 用户更容易困惑“这个字段到底是模板还是实例”
- 代码验证、报错、文档维护成本都会变高

所以这档适合以后真的要把“模型导入”做成一个稳定用户功能时再上，不适合作为第一刀。

## 推荐落地路径

如果我们现在要真正开始做，我建议就按这个路线：

1. 先实现 Level 0 + Level 1
结果是：
- 不写 JSON 也能直接用
- 写 JSON 就能解决常见精调问题

2. Level 1 第一版只支持 5 个字段
就是：
- `display_name`
- `default_scale`
- `center_offset`
- `two_sided`
- `use_builtin_materials`

3. 同时补一个简短的说明文档
建议文档内容结构非常简单：
- 支持格式
- 如何零配置导入
- sidecar JSON 文件如何命名
- 5 个字段各自做什么
- 一个最小示例
- 一个常见问题区

## 文档草案结构

文件名我建议后面可以叫：
- `src/rendering/MODEL_IMPORT.md`
或者
- `src/rendering/README.custom-models.md`

文档目录建议是：

1. Quick Start  
2. Zero-config import  
3. Optional sidecar JSON  
4. Supported fields  
5. Examples  
6. Common issues

## 最小 JSON 示例

```json
{
  "default_scale": [0.18, 0.18, 0.18],
  "center_offset": [0.0, 0.0, -0.08]
}
```

## 完整推荐示例

```json
{
  "display_name": "Orange Pyramid",
  "default_scale": [0.15, 0.15, 0.15],
  "center_offset": [0.0, 0.0, 0.0],
  "two_sided": false,
  "use_builtin_materials": false
}
```

## 我对字段边界的建议

为了后面不失控，建议现在就定这条规则：

- 模型 sidecar 只描述“这个模型本身的默认表现”
- 场景对象配置只描述“这个对象实例在当前场景中的摆放和行为”

这样后面不会把所有内容都堆进 JSON。

## 最终建议

第一版直接做下面这个组合最稳：

- 零配置导入保留
- sidecar JSON 可选
- sidecar JSON 先只开放 5 个字段
- 补一份 md 模板文档
- 暂时不做更复杂的 schema

这样用户侧的体验会很清楚：

- 想省事：只放模型文件
- 想精调：再加一个很小的 JSON
- 想知道怎么写：看文档复制模板

如果你愿意，下一步我可以继续把这份方案直接收敛成“可实施版本”：
1. 先帮你定最终 JSON schema
2. 再帮你定文档模板
3. 然后开始改 `ModelResourceFactory` 和相关测试

## 可实施版本 V1

这一节开始把上面的思路收敛成一个可以直接落地的第一版方案。

目标不是一步把模型导入做成完整产品，而是先把下面这件事做顺：

- 不写 sidecar JSON 也能导入并使用模型
- 需要精调时只写一个很小的 sidecar JSON
- 用户不需要为了常见模型问题去改 Python 代码

### V1 范围

第一版只做下面这些：

1. 保留零配置导入
2. 新增可选 sidecar JSON 读取
3. sidecar JSON 只支持 5 个字段
4. 启动时输出更清晰的扫描与注册结果
5. 补一份面向用户的简短说明文档

第一版不做下面这些：

- 递归扫描子目录
- 图形化模型管理界面
- 缩略图或预览图
- 别名系统
- 复杂 schema 校验器
- 模型导入向导

## 首版 sidecar JSON schema

### 文件命名规则

模型文件：
- `teapot.glb`

对应 sidecar：
- `teapot.model.json`

规则：
- sidecar 文件必须和模型文件同目录
- sidecar 文件必须和模型文件同名，仅后缀改为 `.model.json`
- sidecar 缺失时不报错，直接走零配置导入

### 重名与冲突策略

V1 不尝试“自动解决”重名，而是采用“显式报冲突并拒绝注册”的策略。

规则如下：

- `shape_id` 由模型文件名去后缀后再转小写得到
- 同一扫描目录中，如果两个模型文件映射到同一个 `shape_id`，则视为冲突
- 如果扫描出的 `shape_id` 与内置模型名冲突，也视为冲突
- 冲突项不注册，不进行静默覆盖
- 只有当 `shape_id` 唯一且有效时，才会继续查找和读取对应的 `.model.json`

例如下面这些情况都应直接报冲突：

- `teapot.glb` 与 `teapot.egg`
- `cube.glb` 与内置 `cube`

日志需要明确列出：

- 冲突的 `shape_id`
- 涉及的文件路径
- 该 `shape_id` 已被跳过注册
- 用户应通过重命名来解决冲突

V1 不支持通过 sidecar JSON 提供别名，也不支持通过配置文件覆盖 `shape_id`。

### 首版字段定义

V1 只支持这 5 个字段：

```json
{
  "display_name": "Teapot",
  "default_scale": [0.2, 0.2, 0.2],
  "center_offset": [0.0, 0.0, -0.1],
  "two_sided": false,
  "use_builtin_materials": false
}
```

字段约束建议如下：

- `display_name`
  - 类型：字符串
  - 必填：否
  - 默认值：文件名去后缀后的 `shape_id`

- `default_scale`
  - 类型：长度为 3 的数字数组
  - 必填：否
  - 默认值：`[1.0, 1.0, 1.0]`
  - 约束：每个值都必须大于 0

- `center_offset`
  - 类型：长度为 3 的数字数组
  - 必填：否
  - 默认值：`[0.0, 0.0, 0.0]`

- `two_sided`
  - 类型：布尔值
  - 必填：否
  - 默认值：`false`

- `use_builtin_materials`
  - 类型：布尔值
  - 必填：否
  - 默认值：`false`

### V1 解析策略

建议采用宽容解析：

- 文件不存在：忽略，不报错
- 文件存在但 JSON 非法：跳过该 sidecar，并记录 warning
- 某个字段非法：只忽略该字段，其他合法字段继续生效
- 所有未提供字段都回退到默认值

这样可以避免用户因为一个配置写错，导致整个模型直接不可用。

## 用户文档模板草案

后续建议新增一份真正给用户看的文档，例如：
- `src/rendering/MODEL_IMPORT.md`

这份文档建议尽量短，只保留用户真正会用到的内容。

### 建议目录

1. What this feature does
2. Quick start
3. Zero-config import
4. Optional sidecar JSON
5. Supported fields
6. Examples
7. Common issues

### 建议文案骨架

#### 1. What this feature does

说明系统支持把 `glb`、`egg`、`bam` 模型放进 `assets/custom_models`，并通过 `shape` 名称在场景中使用。

#### 2. Quick start

三步说明：

1. 把模型文件放进 `assets/custom_models`
2. 用文件名去后缀作为 `shape`
3. 在场景对象配置中引用它

#### 3. Zero-config import

提供一个最短示例：

```python
{
    "object_id": "demo_model",
    "shape": "teapot",
    "init_pos": {"x": 0.0, "y": -0.08, "z": 0.18},
    "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
    "scale": {"x": 0.2, "y": 0.2, "z": 0.2},
    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "interactable": true
}
```

#### 4. Optional sidecar JSON

解释：如果模型中心、默认大小或双面渲染不合适，可以添加同名 `.model.json` 文件。

#### 5. Supported fields

把 5 个字段逐一解释清楚，保持和 schema 一致。

#### 6. Examples

建议至少放两个例子：

- 最小 sidecar 示例
- 完整推荐 sidecar 示例

#### 7. Common issues

建议覆盖这几个常见问题：

- 模型放进去但没有显示
- `shape` 名写错
- 模型太大或太小
- 模型位置看起来不对
- 模型只有一面可见
- 模型颜色和原材质不一致

## ModelResourceFactory 改造清单

下面这部分对应真正的代码实施任务。

### 1. 新增 sidecar 发现逻辑

在自动扫描模型文件时：

- 先识别模型主文件
- 再按同名规则查找 `.model.json`
- 如果存在则尝试读取并解析

建议新增一个内部辅助方法，例如：

- `_load_sidecar_config(model_file_path: str) -> dict[str, Any]`

职责：
- 找到 sidecar
- 读取 JSON
- 做宽容解析
- 返回一个标准化后的配置字典

### 2. 新增 sidecar 到 ModelTemplate 的映射逻辑

建议新增一个内部辅助方法，例如：

- `_build_template_from_scanned_model(model_file_path: str) -> ModelTemplate`

职责：
- 从文件名推导 `shape_id`
- 读取 sidecar 配置
- 按默认值填充缺省字段
- 构造最终 `ModelTemplate`

### 3. 在 ModelTemplate 上补充用户可见名称字段

如果后续需要在日志、调试面板、对象列表中展示更友好的名字，建议给 `ModelTemplate` 增加：

- `display_name: str | None = None`

第一版不是强依赖，但现在加上边界会更清楚。

### 4. 优化扫描日志输出

现在的日志主要是“扫描了几个模型”。

建议改成至少能输出：

- 扫描目录路径
- 发现了哪些模型文件
- 哪些模型成功注册
- 哪些 sidecar 解析失败
- 最终的 `shape_id`
- 是否使用了 sidecar 覆盖值

目标是让用户能用日志直接判断：
“模型有没有被系统识别，以及识别成了什么配置。”

### 5. 保持零配置路径不被破坏

这一点必须明确：

- 没有 sidecar 时，行为必须与现在兼容
- sidecar 是增强能力，不是门槛

## 场景接入层的处理建议

第一版先不改 scene object 的基本接入方式。

也就是说，用户依然通过场景对象配置去决定：

- `object_id`
- `shape`
- `init_pos`
- `init_hpr`
- `scale`
- `color`
- `interactable`

但要明确边界：

- 模型 sidecar 决定“模型默认模板”
- 场景对象配置决定“这个实例在当前场景中的摆放和行为”

第一版不要把这两层混起来，否则后面会很难维护。

## 首版验收标准

### 用户体验验收

1. 用户只放一个模型文件，不写 JSON，也能正常被扫描并通过 `shape` 使用
2. 用户写一个很小的 sidecar JSON，就能修正常见问题
3. 用户不需要去改 rendering 侧 Python 注册代码
4. 用户能从日志中看懂模型是否注册成功

### 功能验收

1. 支持 `glb`、`egg`、`bam`
2. sidecar 缺失时不报错
3. sidecar 非法时只 warning，不影响模型主文件导入
4. 合法字段能覆盖默认模板值
5. 不合法字段不会拖垮整个模型注册流程

## 测试清单

建议至少补这些测试。

### 单元测试

1. 自动扫描模型文件时，缺失 sidecar 仍能正常注册
2. sidecar 存在且合法时，字段能正确映射到 `ModelTemplate`
3. sidecar 中单个字段非法时，其他字段仍可生效
4. sidecar 整体 JSON 非法时，模型仍能按零配置路径注册
5. `shape_id` 仍然默认等于文件名去后缀

### 集成测试

1. 通过 `shape` 使用自动扫描模型时，实例能正常创建
2. 带 `default_scale` 的模型在未手动覆盖时能体现模板默认值
3. 带 `center_offset` 的模型能应用偏移
4. `two_sided` 和 `use_builtin_materials` 能影响实例创建行为

## 推荐实施顺序

建议按下面顺序落地：

1. 在 `ModelResourceFactory` 中加入 sidecar 读取与模板构建逻辑
2. 保持现有自动扫描入口不变
3. 增加宽容日志与 warning
4. 补最小测试集合
5. 最后再写给用户看的 `MODEL_IMPORT.md`

这个顺序的好处是：

- 先把能力做出来
- 再把行为稳定下来
- 最后再固化成用户文档

## V1 最终结论

第一版不追求“模型导入一键化”，而是追求“模型导入不再需要改 rendering 注册代码，并且允许用一个很小的 JSON 做精调”。

这是一条成本最低、后续最好扩展、也最符合当前项目阶段的路线。