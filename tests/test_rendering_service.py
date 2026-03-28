from __future__ import annotations

from src.contracts import GesturePacket, SceneCommand, Vec3
from src.rendering.debug.data_panel import DataPanelManager
from src.rendering.rendering_core import RenderingCoreManager
from src.rendering import service as rendering_service
from src.rendering.service import ObjectInitialState, RenderingServiceImpl
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
        contract_version="2.0.0",
        command_id=command_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        command_type=command_type,
        object_id=object_id,
        payload={} if payload is None else payload,
    )


def make_packet_with_rotation() -> GesturePacket:
    return GesturePacket(
        contract_version="2.0.0",
        frame_id=7,
        timestamp_ms=112,
        hand_id="hand-right",
        tracking_state="tracked",
        confidence=0.93,
        pinch_state="pinched",
        index_tip=Vec3(0.4, 0.2, 0.1),
        thumb_tip=Vec3(0.3, 0.2, 0.1),
        wrist=Vec3(0.2, 0.1, 0.0),
        coordinate_space="camera_norm",
        pinch_distance=0.031,
        debug={
            "rotation": {
                "enabled": True,
                "rotating": True,
                "slot": 4,
                "slot_count": 18,
                "slot_x": 2,
                "slot_y": 3,
                "slot_z": 4,
                "deg_x": 40.0,
                "deg_y": 60.0,
                "deg_z": 80.0,
                "gate_count": 3,
                "source": "equivalent_xyz",
                "mode_name": "ROTATE_ENABLED",
            }
        },
    )


class FakeTaskManager:
    def __init__(self) -> None:
        self.stopped = False
        self.steps = 0

    def stop(self) -> None:
        self.stopped = True

    def step(self) -> None:
        self.steps += 1


class FakeWindow:
    def __init__(self) -> None:
        self.closed = False
        self.width = 1600
        self.height = 900

    def close(self) -> None:
        self.closed = True

    def getXSize(self) -> int:
        return self.width

    def getYSize(self) -> int:
        return self.height

    def requestProperties(self, props) -> None:
        get_x_size = getattr(props, "getXSize", None)
        get_y_size = getattr(props, "getYSize", None)
        if callable(get_x_size) and callable(get_y_size):
            self.width = int(get_x_size())
            self.height = int(get_y_size())


class FakeRenderRoot:
    def __init__(self) -> None:
        self.children: list[object] = []

    def attachNewNode(self, node) -> "FakeNodePath":
        child = FakeNodePath(getattr(node, "name", "child"))
        child.parent = self
        self.children.append(child)
        return child


class FakeBase:
    def __init__(self) -> None:
        self.render = FakeRenderRoot()
        self.taskMgr = FakeTaskManager()
        self.win = FakeWindow()
        self.destroyed = False
        self.accepted_events: dict[str, object] = {}

    def accept(self, event_name: str, callback) -> None:
        self.accepted_events[event_name] = callback

    def destroy(self) -> None:
        self.destroyed = True

    def userExit(self) -> None:
        self.destroyed = True


class FakeWindowAdapter:
    def __init__(self) -> None:
        self._base = FakeBase()
        self._is_initialized = False
        self.quit_callback = None

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

    def step(self) -> None:
        self._base.taskMgr.step()

    def set_quit_handler(self, callback) -> None:
        self.quit_callback = callback


class FakeNodePath:
    def __init__(self, name: str) -> None:
        self.name = name
        self.parent = None

    def reparentTo(self, parent: object) -> None:
        self.parent = parent

    def attachNewNode(self, node) -> "FakeNodePath":
        child = FakeNodePath(getattr(node, "name", "child"))
        child.parent = self
        return child

    def removeChildren(self) -> None:
        return None

    def removeNode(self) -> None:
        return None

    def hide(self) -> None:
        return None

    def show(self) -> None:
        return None

    def setPos(self, *values: float) -> None:
        self.pos = values

    def setScale(self, value: float) -> None:
        self.scale = value

    def setTransparency(self, mode) -> None:
        self.transparency = mode

    def isEmpty(self) -> bool:
        return False


class FakeObjectNode:
    def __init__(self) -> None:
        self.pos = None
        self.hpr = None
        self.material = None
        self.scale = None

    def setPos(self, *values: float) -> None:
        self.pos = values

    def setHpr(self, *values: float) -> None:
        self.hpr = values

    def setMaterial(self, material: object, priority: int) -> None:
        self.material = (material, priority)

    def setScale(self, value: float) -> None:
        self.scale = value


class FakeVirtualHand:
    def __init__(self, *args, **kwargs) -> None:
        self.last_points = None
        self.root = FakeNodePath("virtual_hand")

    def update_points(self, points) -> None:
        self.last_points = points


def test_rendering_start_resets_state_and_can_restart(monkeypatch) -> None:
    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)
    monkeypatch.setattr(rendering_service, "VirtualHand", FakeVirtualHand)

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


def test_data_panel_formats_rotation_lines_from_packet_debug() -> None:
    packet = make_packet_with_rotation()

    lines = DataPanelManager._rotation_lines(packet)

    assert lines == (
        "rot: ROTATE_ENABLED rot/live g03",
        "xyz: +40.0 +60.0 +80.0",
    )


def test_rendering_validation_does_not_mutate_invalid_command() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    command = SceneCommand(
        contract_version="2.0.0",
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
            contract_version="2.0.0",
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


def test_rendering_records_structured_errors_for_recoverable_command_format_issues() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING

    service.push(
        make_command(
            command_id="pose-invalid-1",
            frame_id=5,
            command_type="set_object_pose",
            payload={"position": "bad-position", "hpr": [0.0, 0.0, 0.0]},
        )
    )

    health = service.health()

    assert health["errors"][-1]["code"] == "rendering.set_object_pose.position.invalid_type"
    assert "timestamp" in health["errors"][-1]


def test_rendering_pose_logging_is_debounced(caplog) -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    service._object_cache["primary_cube"] = FakeObjectNode()

    with caplog.at_level("INFO", logger="rendering_service"):
        service.push(
            make_command(
                command_id="pose-1",
                frame_id=1,
                timestamp_ms=1_000,
                command_type="set_object_pose",
                payload={"position": [0.1, 0.2, 0.3], "hpr": [0.0, 0.0, 0.0]},
            )
        )
        service.push(
            make_command(
                command_id="pose-2",
                frame_id=2,
                timestamp_ms=1_100,
                command_type="set_object_pose",
                payload={"position": [0.2, 0.3, 0.4], "hpr": [0.0, 0.0, 0.0]},
            )
        )
        service.push(
            make_command(
                command_id="pose-3",
                frame_id=3,
                timestamp_ms=2_100,
                command_type="set_object_pose",
                payload={"position": [0.3, 0.4, 0.5], "hpr": [0.0, 0.0, 0.0]},
            )
        )

    pose_logs = [record.message for record in caplog.records if "Updated object pose" in record.message]

    assert len(pose_logs) == 2
    assert "suppressed_updates=1" not in pose_logs[0]
    assert "suppressed_updates=1" in pose_logs[1]


def test_rendering_maps_contract_world_norm_axes_to_panda_axes() -> None:
    service = RenderingServiceImpl()

    scene_pos = service._world_norm_to_scene_pos((0.25, 0.6, -0.4))

    assert scene_pos == (0.25, -0.4, 0.6)


def test_rendering_applies_pose_updates_with_axis_remap() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["primary_cube"] = obj

    service.push(
        make_command(
            command_id="pose-remap-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={"position": {"x": 0.2, "y": 0.7, "z": -0.3}, "hpr": [0.0, 0.0, 0.0]},
        )
    )

    assert obj.pos == (0.2, -0.3, 0.7)
    assert obj.hpr == (0.0, 0.0, 0.0)


def test_rendering_applies_position_sensitivity_to_pose_updates() -> None:
    service = RenderingServiceImpl(position_sensitivity=1.5)
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["primary_cube"] = obj

    service.push(
        make_command(
            command_id="pose-scale-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={"position": {"x": 0.2, "y": 0.7, "z": -0.3}, "hpr": [0.0, 0.0, 0.0]},
        )
    )

    assert obj.pos == (0.30000000000000004, -0.44999999999999996, 1.0499999999999998)


def test_rendering_applies_pending_grab_material_state() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["primary_cube"] = obj

    service.push(
        make_command(
            command_id="state-pending-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_state",
            payload={"interaction_state": "pending_grab"},
        )
    )

    assert obj.material is not None


def test_rendering_updates_virtual_hand_from_scene_command() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    service._virtual_hand = FakeVirtualHand()

    service.push(
        make_command(
            command_id="hand-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_hand_pose",
            object_id="hand-1",
            payload={
                "coordinate_space": "world_norm",
                "visible": True,
                "points": {
                    "wrist": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "thumb_tip": {"x": 0.1, "y": 0.0, "z": 0.0},
                    "index_tip": {"x": -0.1, "y": 0.0, "z": 0.0},
                    "anchor": {"x": 0.0, "y": 0.05, "z": 0.0},
                },
            },
        )
    )

    assert service._virtual_hand.last_points is not None
    assert service.health()["stats"]["hand_pose_updates"] == 1


def test_rendering_reset_restores_cached_scene_pose() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["primary_cube"] = obj
    service._object_initial_states["primary_cube"] = ObjectInitialState(
        pos=(0.1, -0.2, 0.4),
        hpr=(1.0, 2.0, 3.0),
    )

    service.push(make_command(command_id="reset-1", frame_id=1, command_type="reset_interaction"))

    assert obj.pos == (0.1, -0.2, 0.4)
    assert obj.hpr == (1.0, 2.0, 3.0)


def test_rendering_step_advances_panda3d_task_manager(monkeypatch) -> None:
    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)
    monkeypatch.setattr(rendering_service, "VirtualHand", FakeVirtualHand)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()

    service.step()

    stats = service.health()["stats"]
    assert service._window_adapter.get_base().taskMgr.steps == 1
    assert stats["render_steps"] == 1


def test_rendering_flushes_suppressed_pose_logs_on_stop(caplog) -> None:
    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service._status = LIFECYCLE_RUNNING
    service._object_cache["primary_cube"] = FakeObjectNode()
    service._window_adapter = FakeWindowAdapter()

    with caplog.at_level("INFO", logger="rendering_service"):
        service.push(
            make_command(
                command_id="pose-1",
                frame_id=1,
                timestamp_ms=1_000,
                command_type="set_object_pose",
                payload={"position": [0.1, 0.2, 0.3], "hpr": [0.0, 0.0, 0.0]},
            )
        )
        service.push(
            make_command(
                command_id="pose-2",
                frame_id=2,
                timestamp_ms=1_050,
                command_type="set_object_pose",
                payload={"position": [0.2, 0.3, 0.4], "hpr": [0.0, 0.0, 0.0]},
            )
        )
        service.stop()

    assert any(
        record.message == "Suppressed 1 repetitive pose update log entries"
        for record in caplog.records
    )


def test_rendering_core_window_size_uses_screen_scale_and_aspect_ratio() -> None:
    width, height = RenderingCoreManager.compute_window_size(screen_size=(1920, 1200))

    assert width == 1536
    assert height == 864


def test_rendering_core_window_size_falls_back_when_display_is_invalid() -> None:
    width, height = RenderingCoreManager.compute_window_size(screen_size=(0, 0))

    assert (width, height) == RenderingCoreManager.reference_window_size()


def test_rendering_core_aspect_lock_prefers_width_when_width_changes_more() -> None:
    target = RenderingCoreManager.compute_aspect_locked_size(
        (1200, 720),
        previous_size=(1000, 560),
    )

    assert target == (1200, 675)


def test_rendering_core_aspect_lock_prefers_height_when_height_changes_more() -> None:
    target = RenderingCoreManager.compute_aspect_locked_size(
        (1100, 700),
        previous_size=(1100, 560),
    )

    assert target == (1244, 700)


def test_rendering_service_registers_quit_callback_with_window_adapter(monkeypatch) -> None:
    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)
    monkeypatch.setattr(rendering_service, "VirtualHand", FakeVirtualHand)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.set_quit_callback(lambda: None)
    service.start()

    assert service._window_adapter.quit_callback is not None


def test_rendering_core_registers_quit_shortcuts() -> None:
    manager = RenderingCoreManager()
    fake_base = FakeBase()
    manager._base = fake_base

    manager.set_quit_handler(lambda: None)

    assert set(fake_base.accepted_events) >= {"meta-w", "meta-q", "control-w", "control-q"}
