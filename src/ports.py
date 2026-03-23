from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

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


class DebugFrameSource(ABC):
    @abstractmethod
    def get_camera_data(self) -> tuple[Any | None, Any | None]:
        """Return the latest camera frame and raw observation for debug rendering."""
        ...


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
        """Update gesture data to the rendering window's real-time panel."""
        ...

    @abstractmethod
    def update_camera_frame(
        self,
        frame: Any,
        observation: Any | None = None,
        packet: Optional[GesturePacket] = None,
    ) -> None:
        """Update camera preview data used by the rendering debug overlay."""
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