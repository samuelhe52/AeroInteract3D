from __future__ import annotations

from src.contracts import GesturePacket
from src.ports import GestureInputPort


class GestureInputServiceStub(GestureInputPort):
    def start(self) -> None:
        return None

    def poll(self) -> GesturePacket | None:
        return None

    def health(self) -> dict:
        return {"status": "not_implemented"}

    def stop(self) -> None:
        return None
