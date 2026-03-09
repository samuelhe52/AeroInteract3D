from __future__ import annotations

"""项目模块间的抽象端口定义。

这个文件只定义“对外公开必须实现什么方法”，不负责具体实现。

可以把它理解成接口合同：
- Gesture 模块负责产出 `GesturePacket`
- Bridge 模块负责把 `GesturePacket` 转成 `SceneCommand`
- Rendering 模块负责消费 `SceneCommand`

说明：
- 这里保留 4 个公开生命周期函数：`start()`、`poll()`/`push()`/`process()`、`health()`、`stop()`。
- 诸如 `_read_frame()`、`_detect_hand()`、`_build_packet()` 这样的内部小函数，应该写在具体实现类里，
    例如 `src/gesture/service_stub.py`，而不是放在端口接口中。
"""

from abc import ABC, abstractmethod

from src.contracts import GesturePacket, SceneCommand


class GestureInputPort(ABC):
    """手势输入端口。

    这是 gesture 模块对外暴露的最小接口。

    典型职责分工：
    - `start()`：做启动准备，比如初始化状态、资源和缓存
    - `poll()`：处理一帧输入，返回一个 `GesturePacket` 或 `None`
    - `health()`：返回当前健康状态、错误信息和运行指标
    - `stop()`：释放资源并停止服务

    具体实现中可以继续拆出私有小函数，例如：
    - `_setup_backend()`
    - `_read_frame()`
    - `_detect_hand()`
    - `_compute_pinch_state()`
    - `_build_packet()`
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def poll(self) -> GesturePacket | None: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...


class RenderOutputPort(ABC):
    """渲染输出端口。

    这是 rendering 模块对外暴露的最小接口。

    典型职责分工：
    - `start()`：初始化渲染环境或消费通道
    - `push()`：接收一条 `SceneCommand` 并执行或缓存
    - `health()`：返回当前渲染模块状态
    - `stop()`：释放渲染资源并停止服务
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def push(self, command: SceneCommand) -> None: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...


class BridgeService(ABC):
    """桥接服务端口。

    这是 bridge 模块对外暴露的最小接口。

    典型职责分工：
    - `start()`：初始化桥接状态机和命令发送准备
    - `process()`：把一个 `GesturePacket` 转成 0 到多条 `SceneCommand`
    - `health()`：返回桥接模块状态和异常信息
    - `stop()`：清理状态并停止服务
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def process(self, packet: GesturePacket) -> list[SceneCommand]: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...
