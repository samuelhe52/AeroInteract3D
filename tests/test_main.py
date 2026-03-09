from __future__ import annotations

import main as app_main

from src.contracts import GesturePacket, SceneCommand, Vec3
from src.ports import BridgeService, GestureInputPort, RenderOutputPort


class RunningGestureInput(GestureInputPort):
    def start(self) -> None:
        return None

    def poll(self) -> GesturePacket | None:
        return None

    def health(self) -> dict:
        return {"status": "RUNNING"}

    def stop(self) -> None:
        return None


class RunningBridge(BridgeService):
    def start(self) -> None:
        return None

    def process(self, packet: GesturePacket) -> list[SceneCommand]:
        return []

    def health(self) -> dict:
        return {"status": "RUNNING"}

    def stop(self) -> None:
        return None


class RunningRender(RenderOutputPort):
    def start(self) -> None:
        return None

    def push(self, command: SceneCommand) -> None:
        return None

    def health(self) -> dict:
        return {"status": "RUNNING"}

    def stop(self) -> None:
        return None


class PlaceholderBridge(RunningBridge):
    def health(self) -> dict:
        return {"status": "not_implemented"}


def test_initialize_requires_all_components_to_be_running() -> None:
    app = app_main.App(
        app_main.AppConfig(),
        RunningGestureInput(),
        PlaceholderBridge(),
        RunningRender(),
    )

    try:
        app.initialize()
    except RuntimeError as exc:
        assert "bridge=not_implemented" in str(exc)
        assert app.lifecycle_state == app_main.LIFECYCLE_DEGRADED
    else:
        raise AssertionError("initialize() should fail when a component is not ready")
    finally:
        app.shutdown()


def test_initialize_succeeds_when_all_components_are_running() -> None:
    app = app_main.App(
        app_main.AppConfig(),
        RunningGestureInput(),
        RunningBridge(),
        RunningRender(),
    )

    try:
        app.initialize()
        assert app.lifecycle_state == app_main.LIFECYCLE_RUNNING
    finally:
        app.shutdown()


def test_main_returns_failure_for_placeholder_services() -> None:
    assert app_main.main([]) == 1