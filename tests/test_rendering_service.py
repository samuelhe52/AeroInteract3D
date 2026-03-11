from __future__ import annotations

from src.contracts import SceneCommand
from src.rendering import service as rendering_service
from src.rendering.service import RenderingServiceImpl
from src.utils.runtime import LIFECYCLE_DEGRADED, LIFECYCLE_RUNNING, LIFECYCLE_STOPPED


def make_command(
    *,
    command_id: str = "cmd-1",
    frame_id: int = 1,
    timestamp_ms: int = 100,
    command_type: str = "heartbeat",
    object_id: str = "primary_cube",
    payload: dict | None = None,
) -> SceneCommand:
    return SceneCommand(
        contract_version="0.1.0",
        command_id=command_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        command_type=command_type,
        object_id=object_id,
        payload={} if payload is None else payload,
    )


class FakeTaskManager:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeWindow:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeBase:
    def __init__(self) -> None:
        self.render = object()
        self.task_mgr = FakeTaskManager()
        self.win = FakeWindow()
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class FakeWindowAdapter:
    def __init__(self) -> None:
        self._base = FakeBase()
        self._is_initialized = False

    def init_window(self, window_size: tuple = (800, 600), window_title: str = "AeroInteract3D Rendering") -> None:
        self._is_initialized = True

    def config_camera_for_world_norm(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("window must be initialized")

    def create_base_lights(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("window must be initialized")

    def get_base(self) -> FakeBase:
        return self._base

    def is_initialized(self) -> bool:
        return self._is_initialized

    def reset_scene(self, scene_root: object) -> None:
        return None


class FakeNodePath:
    def __init__(self, name: str) -> None:
        self.name = name
        self.parent = None

    def reparentTo(self, parent: object) -> None:
        self.parent = parent

    def removeChildren(self) -> None:
        return None

    def isEmpty(self) -> bool:
        return False


def test_rendering_start_resets_state_and_can_restart(monkeypatch) -> None:
    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service._errors = [{"code": "stale"}]
    service._last_command_ts = 999
    service._executed_command_ids.add("old-command")
    service._pending_commands.append(make_command())

    service.start()

    assert service.health()["lifecycle_state"] == LIFECYCLE_RUNNING
    assert service.health()["errors"] == []
    assert service._last_command_ts is None
    assert service._executed_command_ids == set()
    assert service._pending_commands == []

    service.stop()

    assert service.health()["lifecycle_state"] == LIFECYCLE_STOPPED

    service.start()

    assert service.health()["lifecycle_state"] == LIFECYCLE_RUNNING


def test_rendering_validation_does_not_mutate_invalid_command() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    command = SceneCommand(
        contract_version="0.1.0",
        command_id="cmd-invalid",
        frame_id="7",  # type: ignore[arg-type]
        timestamp_ms=100,
        command_type="heartbeat",
        object_id="primary_cube",
        payload=[],  # type: ignore[arg-type]
    )

    service.push(command)

    assert command.frame_id == "7"
    assert command.payload == []
    assert service.health()["lifecycle_state"] == LIFECYCLE_DEGRADED
    assert [error["code"] for error in service.health()["errors"]] == [
        "scene.frame_id.invalid",
        "scene.payload.invalid",
    ]


def test_rendering_error_history_is_bounded() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING

    for index in range(12):
        command = SceneCommand(
            contract_version="0.1.0",
            command_id=f"cmd-{index}",
            frame_id=index,
            timestamp_ms=100 + index,
            command_type="heartbeat",
            object_id="primary_cube",
            payload=[],  # type: ignore[arg-type]
        )
        service.push(command)

    health = service.health()

    assert health["lifecycle_state"] == LIFECYCLE_DEGRADED
    assert len(health["errors"]) == 10
    assert all(error["code"] == "scene.payload.invalid" for error in health["errors"])


def test_rendering_health_exposes_structured_metrics() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING

    service.push(make_command(command_id="heartbeat-1", frame_id=1, command_type="heartbeat"))
    service.push(make_command(command_id="heartbeat-1", frame_id=1, command_type="heartbeat"))
    service.push(make_command(command_id="heartbeat-2", frame_id=0, command_type="heartbeat"))
    service.push(make_command(command_id="invalid-1", frame_id=2, command_type="heartbeat", payload=[]))  # type: ignore[arg-type]

    stats = service.health()["stats"]

    assert stats["commands_seen"] == 4
    assert stats["commands_applied"] == 1
    assert stats["heartbeats_received"] == 1
    assert stats["duplicate_commands"] == 1
    assert stats["stale_commands"] == 1
    assert stats["rejected_commands"] == 1