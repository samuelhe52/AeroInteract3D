# AeroInteract3D

[English Version](README.md)

AeroInteract3D 是一款由普通单目摄像头手势驱动的 3D 交互应用。你可以在桌面场景中用手势操控物体。

## 功能概览

当前版本支持：

- 单手悬停、拾取、移动和释放物体
- 双手缩放支持的物体
- 在包含多个道具的 3D 桌面场景中操作
- 在应用窗口内查看实时摄像头画面
- 打开内置设置和校准视图
- 通过配置文件调整本地运行默认参数

实时处理管线还支持摄像头连续读取失败后的自动恢复、相邻道具间更稳定的悬停选择，以及场景安全范围内的双手缩放。

## 快速上手

环境要求：

- Python `3.12`
- `uv`
- 可用的摄像头
- 与 Panda3D 兼容的 macOS 图形支持

安装依赖：

```bash
make setup
```

运行应用：

```bash
make run
```

如需使用不同摄像头或修改运行时设置：

```bash
make run -- --camera-index 1
```

## 本地配置

如需设置本地持久化默认值，可从模板创建根目录下的 `.run.yaml`：

```bash
cp .run.example.yaml .run.yaml
```

该文件用于保存偏好的摄像头索引、画面镜像、分辨率、帧率等运行时选项。

## 主要命令

```bash
make run
make preview
make test
make lint
```

`make preview` 启动仅手势实时预览。`make run` 启动完整的 3D 应用。

## 项目结构

- [`main.py`](/Users/samuelhe/projects/AeroInteract3D/main.py)：应用入口
- [`Makefile`](/Users/samuelhe/projects/AeroInteract3D/Makefile)：常用命令
- [`DEVELOPMENT.md`](/Users/samuelhe/projects/AeroInteract3D/DEVELOPMENT.md)：开发环境配置
- [`assets/custom_models/`](/Users/samuelhe/projects/AeroInteract3D/assets/custom_models)：场景模型
- [`src/`](/Users/samuelhe/projects/AeroInteract3D/src)：应用代码
- [`tests/`](/Users/samuelhe/projects/AeroInteract3D/tests)：测试

## 注意事项

- 主应用在 Panda3D 窗口内显示摄像头预览。
- 摄像头采集会请求低延迟单帧缓冲，并在连续读取失败后自动重新打开。
- 非法的运行参数会在摄像头或渲染器初始化前被拒绝。
- 校准和 UI 设置保存在 `~/.config/AeroInteract3D/calibration_profiles.json`（除非设置了 `XDG_CONFIG_HOME`）。
- 开发详情请参阅 [DEVELOPMENT.md](DEVELOPMENT.md)。
