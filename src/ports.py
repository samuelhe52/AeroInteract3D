from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.contracts import GesturePacket, SceneCommand


class GestureInputPort(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def poll(self) -> GesturePacket | None: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...


class RenderOutputPort(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def push(self, command: SceneCommand) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def update_gesture_data(self, packet: Optional[GesturePacket]) -> None:
        """更新手势数据到渲染窗口的实时面板"""
        ...


class BridgeService(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def process(self, packet: GesturePacket) -> list[SceneCommand]: ...

    @abstractmethod
    def health(self) -> dict: ...

    @abstractmethod
    def stop(self) -> None: ...