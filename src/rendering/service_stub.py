from __future__ import annotations

from src.contracts import SceneCommand
from src.ports import RenderOutputPort


class RenderOutputServiceStub(RenderOutputPort):
    def start(self) -> None:
        return None

    def push(self, command: SceneCommand) -> None:
        return None

    def health(self) -> dict:
        return {"status": "not_implemented"}

    def stop(self) -> None:
        return None
