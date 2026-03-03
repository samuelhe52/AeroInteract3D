from __future__ import annotations

from src.contracts import GesturePacket, SceneCommand
from src.ports import BridgeService


class BridgeServiceStub(BridgeService):
    def start(self) -> None:
        return None

    def process(self, packet: GesturePacket) -> list[SceneCommand]:
        return []

    def health(self) -> dict:
        return {"status": "not_implemented"}

    def stop(self) -> None:
        return None
