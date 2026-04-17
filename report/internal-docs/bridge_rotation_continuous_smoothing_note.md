# Bridge Rotation 连续化与稳定化改动说明

本文档记录本次围绕 `gesture -> bridge` 旋转链路做的实现改动，重点说明：

- 原实现为什么会出现“15 度一跳”的离散体验
- 现在连续旋转是如何实现的
- 新增了哪些稳定化 / smoothing 逻辑
- 为什么后来又补了一个“整只手握拳也能进入 rotation mode”的修复
- 相关文件和测试在哪里

## 一句话结论

这次改动把 rotation 通道从“基于 slot 的离散步进”改成了“连续角度累积 + 轻量平滑 + 主轴抑制次轴扰动”。

也就是说：

- Bridge 仍然消费 `deg_x / deg_y / deg_z`
- 但这些角度不再是每跨过一个阈值就跳 `15°`
- 而是 gesture 侧每帧输出连续变化的角度
- 同时在 gesture 侧加入了主轴抑制和 delta smoothing，减少“明明主要在 x 轴运动，结果 y 轴也乱跳”的问题

后续又补了一个 mode gate 修复：rotation mode 的进入不再只依赖 fingertip spread 足够小，紧凑握拳也可以进入。

## 1. 原实现的问题

原来的 rotation 逻辑在 `src/gesture/temporal.py` 中是典型的“累积到阈值再发一步”设计：

- 先取 pinch midpoint 的 `x / y / z` 帧间位移
- 乘以 gain 后转成 `delta_x_deg / delta_y_deg / delta_z_deg`
- 再把每个轴的 delta 累积到各自 buffer
- 只有 buffer 超过阈值时，才把对应轴的 slot 加一或减一
- 输出角度时再用 `slot * ROT_SLOT_STEP_DEG`

由于 `ROT_SLOT_COUNT = 24`，所以：

- `ROT_SLOT_STEP_DEG = 360 / 24 = 15°`

这就直接导致两个问题：

1. 用户实际手势在持续变化，但输出角度只有跨过整段阈值后才会更新。
2. 一次更新就是 `15°`，所以视觉上明显是“卡顿式步进”，不是连续旋转。

另外，旧逻辑虽然有一些基础抗抖处理：

- deadzone
- opposite-direction jitter suppression
- `rotating` gate hysteresis

但它没有真正对输出角度做连续平滑，所以实际感受仍然比较粗糙。

## 2. 本次实现做了什么

核心改动在：

- `src/gesture/constants.py`
- `src/gesture/temporal.py`
- `tests/test_gesture_temporal.py`

### 2.1 从 slot 驱动改成连续角度累积

现在 gesture reducer 内部新增了连续角度状态：

- `_rotation_angle_x_deg`
- `_rotation_angle_y_deg`
- `_rotation_angle_z_deg`

每帧处理流程变成：

1. 计算 pinch midpoint 的帧间位移
2. 通过原来的 gain / deadzone 逻辑得到每轴 `delta_deg`
3. 先做稳定化处理
4. 再做 delta smoothing
5. 把平滑后的 delta 直接累加到 `_rotation_angle_*_deg`
6. 通过 `deg_x / deg_y / deg_z` 直接输出给 Bridge

也就是说，真正驱动 Bridge 的角度现在是连续值，而不是 slot 值。

### 2.2 保留 slot 字段，但改为兼容性派生值

虽然内部已经不是 slot 驱动了，但 debug payload 里仍然保留：

- `slot`
- `slot_x`
- `slot_y`
- `slot_z`
- `slot_count`

原因是：

- 现有 debug overlay 和一些测试还会读取这些字段
- 直接删除会扩大改动面

现在这些 slot 不再决定角度，只是根据连续角度做兼容性映射：

- `slot = int((deg % 360) / ROT_SLOT_STEP_DEG)`

所以它们现在只是 debug/兼容层字段，不再是控制逻辑本体。

## 3. 新增的 smoothing / 稳定化逻辑

### 3.1 主轴抑制次轴扰动

新增常量：

- `ROT_DOMINANT_AXIS_RATIO = 1.75`
- `ROT_SECONDARY_AXIS_ATTENUATION = 0.12`

实现位置：

- `src/gesture/temporal.py::_stabilize_rotation_deltas(...)`

逻辑是：

- 先看当前帧三个轴的 `|delta|` 谁最大
- 如果最大轴明显强于次大轴（达到 ratio 门槛），就认为当前动作有明确主轴
- 对非主轴分量做衰减，只保留 `12%`

这样当用户主要沿 x 轴运动时：

- `x` 仍然主导旋转
- 小幅 `y / z` 扰动不会等量混进来

注意这里不是“硬性只保留最大轴”，而是“有明显主轴时强烈压制次轴”。

这样做比“每帧只留最大分量”更稳一些，原因是：

- 不会在两个轴很接近时频繁切主轴
- 保留一点次轴量，更自然，不会太生硬

### 3.2 每轴 delta smoothing

新增常量：

- `ROT_CONTINUOUS_DELTA_ALPHA = 0.40`

实现位置：

- `src/gesture/temporal.py::_smooth_rotation_delta(...)`

逻辑是对每个轴的 delta 做一层轻量 EMA/lerp 型平滑：

- `smoothed = lerp(previous, current, alpha)`

这层 smoothing 的作用不是“慢吞吞拖影”，而是：

- 吸掉单帧小毛刺
- 降低角速度突变感
- 让连续旋转更顺

同时如果 delta 已经接近 noise floor，则直接归零，不保留无意义残余。

### 3.3 gate 不再阻塞角度更新

旧逻辑里，`rotating` gate 既承担“状态判断”，又间接影响是否发出有效旋转步进。

现在 `ROT_GATE_FRAMES / ROT_GATE_RELEASE_FRAMES` 仍然保留，但用途变成：

- 只表达当前 rotation channel 是否处于稳定 active 状态
- 用于 `rotating=True/False` 的状态显示和 hysteresis

连续角度本身不再等 gate 完全打开才开始变化。

这点很重要，因为它避免了：

- 用户已经开始稳定运动
- 但系统还在等若干帧
- 所以前几帧完全没旋转反馈

现在即使 `rotating` 还没翻到 `True`，连续角度也可以已经开始小幅变化。

## 4. rotation mode 进入回归，以及后续修复

连续旋转改完后，出现了一个实际使用上的回归：

- 用户整只手握拳收紧时，rotation mode 不容易进入

问题不在 Bridge，而在 gesture 侧的 mode gate 手势判定。

### 4.1 原因

原来的“grab”判定基本只看：

- fingertip spread 是否小于 `ROT_HAND_GRAB_TIP_SPREAD_MAX`

这对某些标准姿态可以工作，但对真实“整只手握拳”不够鲁棒。

因为实际握拳时可能出现：

- fingertip spread 不算特别小
- 但 fingertips 已经整体靠近 wrist，显然应该算“抓握”

结果就是：

- `is_grabbed` 判不到
- `grab -> open` 的 mode toggle 序列走不通
- rotation mode 无法进入

### 4.2 修复方式

新增常量：

- `ROT_HAND_GRAB_TIP_WRIST_MAX = 0.30`

修改位置：

- `src/gesture/temporal.py::_is_hand_grabbed(...)`
- `src/gesture/temporal.py::_is_hand_open(...)`

现在 grab 判定变成：

- `spread <= spread_threshold`
- 或者 `max_tip_to_wrist_distance <= wrist_threshold`

也就是允许“紧凑握拳”作为 grab。

同时为了避免同一帧同时被识别成 open，又把 open 判定改成了：

- `spread > spread_threshold`
- 且 `max_tip_to_wrist_distance > wrist_threshold`

这样 grab / open 就重新恢复为互斥关系，不会出现一个紧凑握拳同时满足两边的问题。

## 5. 相关文件

### 主要实现文件

- `src/gesture/constants.py`
- `src/gesture/temporal.py`

### 调试显示

- `src/rendering/debug/cam_preview.py`

这里顺手把 preview 文案改了。因为 grab gate 不再只是看 spread，所以现在预览面板显示：

- `spread`
- `max`（tip 到 wrist 的最大归一化距离）

否则 UI 还写着“grab < 0.270”，会误导调试。

### 测试

- `tests/test_gesture_temporal.py`
- `tests/test_bridge_service.py`
- `tests/test_gesture_service.py`
- `tests/test_main.py`

## 6. 这次新增 / 调整的测试点

### 手势时序测试

`tests/test_gesture_temporal.py` 里主要补了这些方向：

- 连续旋转确实会产生小于 `15°` 的增量，而不是只会整步跳变
- `rotating` gate 还没完全打开时，角度也已经可以开始变化
- 主轴运动时，次轴扰动会被明显压低
- 紧凑握拳 + 张开手的序列也可以成功切换 `rotation mode`

### Bridge 回归验证

Bridge 侧测试没有大改，因为：

- Bridge 仍然消费 `deg_x / deg_y / deg_z`
- 这次主要改变的是 gesture 输出这些角度的方式

现有 bridge tests 通过，说明 contract 没被破坏。

## 7. 当前行为总结

现在的 rotation 行为可以概括为：

1. 用户做 `grab -> open` 序列，进入 `rotation mode`
2. 在 mode active 下，pinch midpoint 的连续位移会被转换为每轴连续角度增量
3. 如果有明确主轴，则次轴分量被强烈抑制
4. 每轴增量再经过一层轻量 smoothing
5. gesture 输出连续 `deg_x / deg_y / deg_z`
6. Bridge 继续把这些角度相对参考姿态转换成对象的 `hpr`

相较于旧版本，用户侧的直接体感变化应该是：

- 不再是明显的 `15°` 一格一格跳
- 小抖动不那么容易串到错误轴
- rotation mode 更容易被真实握拳手势触发

## 8. 仍然保留的限制

这次改动解决的是“连续性”和“基础稳定性”，不是最终版 3D rotation 方案，因此仍然有一些明确限制：

- 当前 rotation 仍然基于 pinch midpoint 的相对位移，不是基于真正的手掌姿态 / 法向量 / 局部坐标系
- 主轴判定是启发式，不是严格姿态分解
- smoothing 是轻量 EMA 风格，不是 One Euro / Kalman 这类更复杂滤波器
- slot 字段虽然还在，但已经不是底层控制逻辑，后续如果要继续清理，可以把依赖 slot 的 debug/test 一并重构掉

## 9. 如果后续还要继续改

如果之后还要继续提升 rotation 体验，优先级建议如下：

1. 先实际录屏验证当前 alpha / dominant-axis ratio 是否合适
2. 如果还觉得次轴串扰明显，再调：
   - `ROT_DOMINANT_AXIS_RATIO`
   - `ROT_SECONDARY_AXIS_ATTENUATION`
3. 如果还觉得角速度毛刺明显，再调：
   - `ROT_CONTINUOUS_DELTA_ALPHA`
4. 如果 mode 进入/退出还不够稳，再继续调：
   - `ROT_HAND_GRAB_TIP_SPREAD_MAX`
   - `ROT_HAND_GRAB_TIP_WRIST_MAX`

如果未来想进一步升级，而不是只调参，那么更自然的下一步会是：

- 把 rotation 输入从“pinch midpoint 位移”升级成“手姿态驱动”的 rotation 估计

但那会是另外一个层级的改动，不属于这次修复的范围。
