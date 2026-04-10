from __future__ import annotations

import logging

import pytest

from src.contracts import GesturePacket, SceneCommand, Vec3
from src.rendering.debug.data_panel import DataPanelManager
from src.rendering.rendering_core import RenderingCoreManager
from src.rendering import service as rendering_service
from src.rendering.service import ObjectInitialState, RenderingServiceImpl
from src.rendering.ui.input_adapter import UIGestureInputAdapter
from src.rendering.ui.interaction import UIButtonBounds, UIButtonInteractionController
from src.rendering.ui.state import UIInputState, UISettingsState
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
                "mode_active": True,
                "grab_detected": False,
                "open_detected": True,
            }
        },
    )


def make_packet_with_menu_candidate(
    *,
    frame_id: int = 9,
    timestamp_ms: int = 100,
    grab_detected: bool = True,
    mode_active: bool = False,
    cursor: tuple[float, float] = (0.35, 0.2),
) -> GesturePacket:
    midpoint_x, midpoint_y = cursor
    return GesturePacket(
        contract_version="2.0.0",
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        hand_id="hand-right",
        tracking_state="tracked",
        confidence=0.93,
        pinch_state="open",
        index_tip=Vec3(midpoint_x + 0.05, midpoint_y, 0.1),
        thumb_tip=Vec3(midpoint_x - 0.05, midpoint_y, 0.1),
        wrist=Vec3(0.2, 0.1, 0.0),
        coordinate_space="camera_norm",
        pinch_distance=0.04,
        debug={
            "rotation": {
                "enabled": False,
                "rotating": False,
                "slot": 0,
                "slot_count": 18,
                "slot_x": 0,
                "slot_y": 0,
                "slot_z": 0,
                "deg_x": 0.0,
                "deg_y": 0.0,
                "deg_z": 0.0,
                "gate_count": 0,
                "source": "equivalent_xyz",
                "mode_name": "MOVE_ONLY",
                "mode_active": mode_active,
                "grab_detected": grab_detected,
                "open_detected": False,
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


class FakeLoader:
    def loadModel(self, model_path):
        return FakeLoadedModel(model_path)


class FakeLoadedModel:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.texture_off = False
        self.pos = None
        self.two_sided = False

    def isEmpty(self) -> bool:
        return False

    def setTextureOff(self, priority: int) -> None:
        self.texture_off = True

    def setPos(self, *values: float) -> None:
        self.pos = values

    def setTwoSided(self, enabled: bool) -> None:
        self.two_sided = enabled

    def copyTo(self, parent):
        child = FakeNodePath(getattr(self.model_path, "cStr", lambda: str(self.model_path))())
        child.parent = parent
        return child

    def removeNode(self) -> None:
        return None


class FakeBase:
    def __init__(self) -> None:
        self.render = FakeRenderRoot()
        self.pixel2d = FakeNodePath("pixel2d")
        self.loader = FakeLoader()
        self.taskMgr = FakeTaskManager()
        self.win = FakeWindow()
        self.destroyed = False
        self.accepted_events: dict[str, object] = {}
        self.background_color = None

    def accept(self, event_name: str, callback, extra_args=None) -> None:
        if extra_args:
            self.accepted_events[event_name] = lambda: callback(*extra_args)
            return
        self.accepted_events[event_name] = callback

    def destroy(self) -> None:
        self.destroyed = True

    def userExit(self) -> None:
        self.destroyed = True

    def setBackgroundColor(self, *values: float) -> None:
        self.background_color = values


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

    def get_pixel2d(self):
        if not self._is_initialized:
            return None
        return self._base.pixel2d

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
        self.hidden = False
        self.color_scale = None
        self.material = None
        self.material_cleared = False

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
        self.hidden = True

    def show(self) -> None:
        self.hidden = False

    def setPos(self, *values: float) -> None:
        self.pos = values

    def setScale(self, *values: float) -> None:
        self.scale = values[0] if len(values) == 1 else values

    def setHpr(self, *values: float) -> None:
        self.hpr = values

    def setTransparency(self, mode) -> None:
        self.transparency = mode

    def setColorScale(self, *values: float) -> None:
        self.color_scale = values

    def setMaterial(self, material: object, priority: int) -> None:
        self.material = (material, priority)

    def clearMaterial(self) -> None:
        self.material = None
        self.material_cleared = True

    def setTag(self, key: str, value: str) -> None:
        if not hasattr(self, "tags"):
            self.tags = {}
        self.tags[key] = value

    def isEmpty(self) -> bool:
        return False


class FakeObjectNode:
    def __init__(self) -> None:
        self.pos = None
        self.hpr = None
        self.material = None
        self.scale = None
        self.color_scale = None
        self.material_cleared = False
        self.hidden = False

    def setPos(self, *values: float) -> None:
        self.pos = values

    def setHpr(self, *values: float) -> None:
        self.hpr = values

    def setMaterial(self, material: object, priority: int) -> None:
        self.material = (material, priority)

    def setScale(self, value: float) -> None:
        self.scale = value

    def setColorScale(self, *values: float) -> None:
        self.color_scale = values

    def clearMaterial(self) -> None:
        self.material = None
        self.material_cleared = True

    def hide(self) -> None:
        self.hidden = True

    def show(self) -> None:
        self.hidden = False


class FakeVirtualHand:
    def __init__(self, *args, **kwargs) -> None:
        self.last_points = None
        self.root = FakeNodePath("virtual_hand")

    def update_points(self, points) -> None:
        self.last_points = points


class FakeVisibilityController:
    def __init__(self) -> None:
        self.visible = True
        self.panel_visible = True
        self.indicator_visible = True

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def set_panel_visible(self, visible: bool) -> None:
        self.panel_visible = visible
        self.visible = visible

    def set_indicator_visible(self, visible: bool) -> None:
        self.indicator_visible = visible


class FakeHomeView:
    def __init__(self, pixel2d, window_size_provider, on_button_activated=None) -> None:
        self.pixel2d = pixel2d
        self.window_size_provider = window_size_provider
        self.on_button_activated = on_button_activated
        self.visible = True
        self.destroyed = False
        self.layout_updates = 0
        self.last_window_size = window_size_provider()
        self.cursor_state = None
        self.last_pinch_state = None
        self.last_settings = None

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def update_layout(self, force: bool = False) -> None:
        self.layout_updates += 1
        self.last_window_size = self.window_size_provider()

    def update_cursor(self, state, pinch_state=None) -> None:
        self.cursor_state = state
        self.last_pinch_state = pinch_state

    def set_ui_settings(self, settings) -> None:
        self.last_settings = (settings.cursor_scale, settings.cursor_opacity)

    def destroy(self) -> None:
        self.destroyed = True


class FakeSettingView:
    def __init__(self, pixel2d, window_size_provider, on_button_activated=None) -> None:
        self.pixel2d = pixel2d
        self.window_size_provider = window_size_provider
        self.on_button_activated = on_button_activated
        self.visible = False
        self.destroyed = False
        self.layout_updates = 0
        self.last_window_size = window_size_provider()
        self.cursor_state = None
        self.last_pinch_state = None
        self.last_settings = None
        self.object_visibility_summary = None
        self.calibration_preview_state = None

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def update_layout(self, force: bool = False) -> None:
        self.layout_updates += 1
        self.last_window_size = self.window_size_provider()

    def update_cursor(self, state, pinch_state=None) -> None:
        self.cursor_state = state
        self.last_pinch_state = pinch_state

    def set_ui_settings(self, settings) -> None:
        self.last_settings = (settings.cursor_scale, settings.cursor_opacity)

    def set_object_visibility_summary(self, total_count: int, hidden_count: int) -> None:
        self.object_visibility_summary = (total_count, hidden_count)

    def update_calibration_preview(self, state) -> None:
        self.calibration_preview_state = state

    def destroy(self) -> None:
        self.destroyed = True


class FakeCalibrationView:
    def __init__(self, pixel2d, window_size_provider, on_button_activated=None) -> None:
        self.pixel2d = pixel2d
        self.window_size_provider = window_size_provider
        self.on_button_activated = on_button_activated
        self.visible = False
        self.destroyed = False
        self.layout_updates = 0
        self.last_window_size = window_size_provider()
        self.cursor_state = None
        self.last_pinch_state = None
        self.last_settings = None
        self.calibration_preview_state = None
        self.selected_parameter_key = "ui_cursor_scale_x"
        self.adjustments: list[int] = []
        self.selection_steps: list[int] = []

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def update_layout(self, force: bool = False) -> None:
        self.layout_updates += 1
        self.last_window_size = self.window_size_provider()

    def update_cursor(self, state, pinch_state=None) -> None:
        self.cursor_state = state
        self.last_pinch_state = pinch_state

    def set_ui_settings(self, settings) -> None:
        self.last_settings = (
            settings.ui_cursor_scale_x,
            settings.ui_cursor_scale_y,
            settings.ui_cursor_offset_x,
            settings.ui_cursor_offset_y,
        )

    def update_calibration_preview(self, state) -> None:
        self.calibration_preview_state = state

    def select_next_parameter(self, step: int = 1) -> str:
        keys = [
            "ui_cursor_scale_x",
            "ui_cursor_scale_y",
            "ui_cursor_offset_x",
            "ui_cursor_offset_y",
        ]
        index = keys.index(self.selected_parameter_key)
        self.selected_parameter_key = keys[(index + step) % len(keys)]
        self.selection_steps.append(step)
        return self.selected_parameter_key

    def adjust_selected_parameter(self, step_count: int):
        self.adjustments.append(step_count)
        if self.on_button_activated is None:
            return None
        current_values = {
            "ui_cursor_scale_x": 1.0,
            "ui_cursor_scale_y": 1.0,
            "ui_cursor_offset_x": 0.0,
            "ui_cursor_offset_y": 0.0,
        }
        if self.last_settings is not None:
            current_values = {
                "ui_cursor_scale_x": self.last_settings[0],
                "ui_cursor_scale_y": self.last_settings[1],
                "ui_cursor_offset_x": self.last_settings[2],
                "ui_cursor_offset_y": self.last_settings[3],
            }
        step_size = 0.01
        next_value = current_values[self.selected_parameter_key] + step_size * float(step_count)
        self.on_button_activated(f"set_{self.selected_parameter_key}:{next_value}")
        return next_value

    def destroy(self) -> None:
        self.destroyed = True


class FakeTableOverlayView:
    def __init__(self, pixel2d, window_size_provider, on_button_activated=None) -> None:
        self.pixel2d = pixel2d
        self.window_size_provider = window_size_provider
        self.on_button_activated = on_button_activated
        self.visible = False
        self.destroyed = False
        self.layout_updates = 0
        self.last_window_size = window_size_provider()
        self.cursor_state = None
        self.last_pinch_state = None
        self.last_settings = None
        self.last_overlay = "none"
        self.last_object_items = []

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def set_overlay(self, overlay) -> None:
        self.last_overlay = str(overlay)

    def update_layout(self, force: bool = False) -> None:
        self.layout_updates += 1
        self.last_window_size = self.window_size_provider()

    def update_cursor(self, state, pinch_state=None) -> None:
        self.cursor_state = state
        self.last_pinch_state = pinch_state

    def set_ui_settings(self, settings) -> None:
        self.last_settings = (settings.cursor_scale, settings.cursor_opacity)

    def set_object_visibility_items(self, items) -> None:
        self.last_object_items = list(items)

    def destroy(self) -> None:
        self.destroyed = True


def patch_ui_views(monkeypatch) -> None:
    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)
    monkeypatch.setattr(rendering_service, "VirtualHand", FakeVirtualHand)
    monkeypatch.setattr(rendering_service, "HomeUIView", FakeHomeView)
    monkeypatch.setattr(rendering_service, "SettingUIView", FakeSettingView)
    monkeypatch.setattr(rendering_service, "CalibrationUIView", FakeCalibrationView)
    monkeypatch.setattr(rendering_service, "TableOverlayUIView", FakeTableOverlayView)


def test_rendering_start_resets_state_and_can_restart(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service._errors = [{"code": "stale"}]
    service._last_command_ts = 999
    service._executed_command_ids.add("old-command")
    service._pending_commands.append(make_command())

    service.start()

    assert service.health()["lifecycle_state"] == LIFECYCLE_RUNNING
    assert service.active_view == "home"


def test_rendering_start_keeps_camera_preview_when_debug_stats_disabled(monkeypatch) -> None:
    patch_ui_views(monkeypatch)
    created_components: list[tuple[str, object]] = []

    class FakeOverlayWindowAdapter(FakeWindowAdapter):
        def get_pixel2d(self):
            return FakeNodePath("pixel2d")

    class FakeAutoScalingManager:
        def __init__(self, rendering_core) -> None:
            self._rendering_core = rendering_core

        def set_scale_callback(self, callback) -> None:
            self.callback = callback

        def get_ui_scale(self) -> float:
            return 1.0

    class FakeDataPanel:
        @classmethod
        def camera_preview_top_margin(cls) -> int:
            return 120

        def __init__(self, auto_scaling) -> None:
            created_components.append(("data_panel", auto_scaling))

        def destroy(self) -> None:
            return None

    class FakeCameraPreview:
        PREVIEW_MARGIN = 12

        def __init__(self, auto_scaling, *, top_margin: int) -> None:
            created_components.append(("camera_preview", top_margin))

        def destroy(self) -> None:
            return None

        def set_ui_scale(self, scale: float) -> None:
            return None

        def set_visible(self, visible: bool) -> None:
            return None

    monkeypatch.setattr(rendering_service, "NodePath", FakeNodePath)
    monkeypatch.setattr(rendering_service, "VirtualHand", FakeVirtualHand)
    monkeypatch.setattr(rendering_service, "AutoScalingManager", FakeAutoScalingManager)
    monkeypatch.setattr(rendering_service, "DataPanelManager", FakeDataPanel)
    monkeypatch.setattr(rendering_service, "CameraPreviewManager", FakeCameraPreview)

    service = RenderingServiceImpl(
        window_adapter_factory=FakeOverlayWindowAdapter,
        debug_stats_enabled=False,
    )

    service.start()

    assert service.health()["lifecycle_state"] == LIFECYCLE_RUNNING
    assert created_components == [("camera_preview", 12)]
    assert service.health()["errors"] == []
    assert service._last_command_ts is None
    assert service._executed_command_ids == set()
    assert service._pending_commands == []

    service.stop()

    assert service.health()["lifecycle_state"] == LIFECYCLE_STOPPED

    service.start()

    assert service.health()["lifecycle_state"] == LIFECYCLE_RUNNING
    assert service.active_view == "home"


def test_rendering_defaults_to_home_view_in_health_stats() -> None:
    service = RenderingServiceImpl()

    stats = service.health()["stats"]

    assert stats["active_view"] == "home"
    assert stats["available_views"] == ["home", "table", "setting", "calibration"]
    assert stats["active_table_overlay"] == "none"
    assert stats["available_table_overlays"] == ["none", "menu", "option"]


def test_rendering_view_switch_toggles_table_visibility() -> None:
    service = RenderingServiceImpl()
    service._scene_root = FakeNodePath("scene_root")
    service._data_panel = FakeVisibilityController()
    service._camera_preview = FakeVisibilityController()

    service.set_active_view("table")

    assert service.active_view == "table"
    assert service._scene_root.hidden is False
    assert service._data_panel.visible is True
    assert service._camera_preview.visible is True

    service.set_active_view("setting")

    assert service.active_view == "setting"
    assert service._scene_root.hidden is True
    assert service._data_panel.visible is False
    assert service._camera_preview.visible is False


def test_rendering_leaving_table_clears_active_overlay() -> None:
    service = RenderingServiceImpl()
    service._scene_root = FakeNodePath("scene_root")
    service._data_panel = FakeVisibilityController()
    service._camera_preview = FakeVisibilityController()

    service.set_active_view("table")
    service.set_active_table_overlay("menu", timestamp_ms=123)

    assert service.active_table_overlay == "menu"

    service.set_active_view("setting")

    assert service.active_table_overlay == "none"


def test_rendering_overlay_locks_table_interaction_commands() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    service._scene_root = FakeNodePath("scene_root")
    service._data_panel = FakeVisibilityController()
    service._camera_preview = FakeVisibilityController()
    obj = FakeObjectNode()
    obj.pos = (0.1, 0.2, 0.3)
    service._object_cache["primary_cube"] = obj
    service._object_interaction_states["primary_cube"] = "idle"
    service.set_active_view("table")
    service.set_active_table_overlay("menu", timestamp_ms=100)

    service.push(
        make_command(
            command_id="locked-pose-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={"position": {"x": 0.7, "y": 0.8, "z": 0.9}},
        )
    )
    service.push(
        make_command(
            command_id="locked-state-1",
            frame_id=2,
            timestamp_ms=101,
            command_type="set_object_state",
            payload={"interaction_state": "grabbed"},
        )
    )

    assert obj.pos == (0.1, 0.2, 0.3)
    assert service._object_interaction_states["primary_cube"] == "idle"


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


def test_rendering_heartbeat_logging_is_debug_only(caplog) -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING

    with caplog.at_level(logging.INFO, logger="rendering_service"):
        service.push(make_command(command_id="heartbeat-info-1", frame_id=1, command_type="heartbeat"))

    assert "Received heartbeat command, module state" not in caplog.text


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


def test_rendering_core_world_norm_camera_pose_uses_front_view() -> None:
    camera_pos, look_at = RenderingCoreManager.camera_pose_for_world_norm()

    assert camera_pos == pytest.approx((0.0, 5.0, 1.34))
    assert look_at == (0.0, 0.0, 0.0)


def test_rendering_centers_box_model_under_transform_pivot() -> None:
    assert RenderingServiceImpl._box_model_center_offset() == (-0.5, -0.5, -0.5)


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


def test_rendering_preserves_position_for_rotation_only_pose_updates() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    obj.pos = (0.2, -0.3, 0.7)
    obj.hpr = (1.0, 2.0, 3.0)
    service._object_cache["primary_cube"] = obj
    service._last_world_norm_pos = (0.2, 0.7, -0.3)
    service._last_scene_pos = obj.pos

    service.push(
        make_command(
            command_id="pose-rotate-only-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={"hpr": {"h": 10.0, "p": 20.0, "r": 30.0}},
        )
    )

    assert obj.pos == (0.2, -0.3, 0.7)
    assert obj.hpr == (10.0, 20.0, 30.0)


def test_rendering_preserves_hpr_for_position_only_pose_updates() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    obj.pos = (0.0, 0.0, 0.0)
    obj.hpr = (4.0, 5.0, 6.0)
    service._object_cache["primary_cube"] = obj

    service.push(
        make_command(
            command_id="pose-position-only-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={"position": {"x": 0.2, "y": 0.7, "z": -0.3}},
        )
    )

    assert obj.pos == (0.2, -0.3, 0.7)
    assert obj.hpr == (4.0, 5.0, 6.0)


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


def test_auto_scanned_custom_models_preserve_authored_materials(tmp_path) -> None:
    custom_model = tmp_path / "teapot.egg"
    custom_model.write_text("placeholder", encoding="utf-8")

    factory = rendering_service.ModelResourceFactory(loader=None, auto_scan_dir=str(tmp_path))

    assert factory.uses_builtin_materials("teapot") is False


def test_rendering_uses_color_scale_for_custom_model_states() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["custom_model"] = obj
    service._object_visual_profiles["custom_model"] = rendering_service.ObjectVisualProfile(
        base_color=(1.0, 1.0, 1.0, 1.0),
        use_builtin_materials=False,
    )

    service.push(
        make_command(
            command_id="state-custom-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_state",
            object_id="custom_model",
            payload={"interaction_state": "grabbed"},
        )
    )

    assert obj.material is None
    assert obj.material_cleared is True
    assert obj.color_scale == pytest.approx((1.0, 0.58, 0.58, 0.92))


def test_rendering_applies_rotating_material_state() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    obj = FakeObjectNode()
    service._object_cache["primary_cube"] = obj

    service.push(
        make_command(
            command_id="state-rotating-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_state",
            payload={"interaction_state": "rotating"},
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


def test_rendering_routes_hand_pose_to_both_virtual_hands() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    service._virtual_hands = {
        "hand-1": FakeVirtualHand(),
        "hand-2": FakeVirtualHand(),
    }

    base_payload = {
        "coordinate_space": "world_norm",
        "visible": True,
        "points": {
            "wrist": {"x": 0.0, "y": 0.0, "z": 0.0},
            "thumb_tip": {"x": 0.1, "y": 0.0, "z": 0.0},
            "index_tip": {"x": -0.1, "y": 0.0, "z": 0.0},
            "anchor": {"x": 0.0, "y": 0.05, "z": 0.0},
        },
    }

    service.push(
        make_command(
            command_id="hand-1",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_hand_pose",
            object_id="hand-1",
            payload=base_payload,
        )
    )
    service.push(
        make_command(
            command_id="hand-2",
            frame_id=2,
            timestamp_ms=120,
            command_type="set_hand_pose",
            object_id="hand-2",
            payload=base_payload,
        )
    )

    assert service._virtual_hands["hand-1"].last_points is not None
    assert service._virtual_hands["hand-2"].last_points is not None


def test_rendering_tracks_dual_scale_status_from_pose_payload() -> None:
    service = RenderingServiceImpl()
    service._status = LIFECYCLE_RUNNING
    service._object_cache["primary_cube"] = FakeObjectNode()

    service.push(
        make_command(
            command_id="pose-scale-debug",
            frame_id=1,
            timestamp_ms=100,
            command_type="set_object_pose",
            payload={
                "coordinate_space": "world_norm",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "debug": {"dual_scale": {"active": True, "ratio": 1.42, "distance_xy": 0.12}},
            },
        )
    )

    assert service._dual_scale_active is True
    assert service._dual_scale_ratio == pytest.approx(1.42)


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
    patch_ui_views(monkeypatch)

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
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.set_quit_callback(lambda: None)
    service.start()

    assert service._window_adapter.quit_callback is not None


def test_rendering_initializes_home_view_on_start(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)

    service.start()

    assert isinstance(service._home_view, FakeHomeView)
    assert isinstance(service._table_overlay_view, FakeTableOverlayView)
    assert service._home_view.visible is True
    assert service._home_view.last_window_size == (1600, 900)


def test_rendering_view_switch_updates_home_view_visibility() -> None:
    service = RenderingServiceImpl()
    service._home_view = FakeHomeView(pixel2d=None, window_size_provider=lambda: (1600, 900))
    service._scene_root = FakeNodePath("scene_root")
    service._data_panel = FakeVisibilityController()
    service._camera_preview = FakeVisibilityController()

    service.set_active_view("table")

    assert service._home_view.visible is False

    service.set_active_view("home")

    assert service._home_view.visible is True


def test_rendering_step_updates_home_view_layout_for_window_size_changes(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service._window_adapter.get_base().win.width = 1280
    service._window_adapter.get_base().win.height = 720

    service.step()

    assert service._home_view.last_window_size == (1280, 720)
    assert service._home_view.layout_updates >= 1


def test_ui_gesture_input_adapter_maps_midpoint_to_pixels() -> None:
    adapter = UIGestureInputAdapter()

    ui_state = adapter.to_ui_input(make_packet_with_rotation(), window_size=(1600, 900))

    assert ui_state.visible is True
    assert ui_state.cursor_norm == pytest.approx((0.325, 0.4))
    assert ui_state.cursor_pixels == pytest.approx((520.0, 360.0))


def test_ui_gesture_input_adapter_hides_cursor_when_tracking_is_lost() -> None:
    adapter = UIGestureInputAdapter()
    packet = make_packet_with_rotation()
    packet.tracking_state = "temporarily_lost"

    ui_state = adapter.to_ui_input(packet, window_size=(1600, 900))

    assert ui_state.visible is False
    assert ui_state.cursor_pixels == pytest.approx((800.0, 450.0))


def test_ui_button_interaction_controller_activates_on_release_inside() -> None:
    controller = UIButtonInteractionController()
    bounds = [UIButtonBounds(100, 100, 300, 200)]

    hover = controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="open",
        button_bounds=bounds,
    )
    pressed = controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="pinched",
        button_bounds=bounds,
    )
    released = controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="open",
        button_bounds=bounds,
    )

    assert hover.hovered_index == 0
    assert hover.activated_index is None
    assert pressed.pressed_index == 0
    assert released.activated_index == 0


def test_ui_button_interaction_controller_cancels_release_outside() -> None:
    controller = UIButtonInteractionController()
    bounds = [UIButtonBounds(100, 100, 300, 200)]

    controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="pinched",
        button_bounds=bounds,
    )
    dragged = controller.update(
        UIInputState(cursor_pixels=(400, 260), visible=True),
        pinch_state="pinched",
        button_bounds=bounds,
    )
    released = controller.update(
        UIInputState(cursor_pixels=(400, 260), visible=True),
        pinch_state="open",
        button_bounds=bounds,
    )

    assert dragged.pressed_index == 0
    assert dragged.hovered_index is None
    assert released.activated_index is None


def test_ui_button_interaction_controller_activates_after_release_candidate_then_open() -> None:
    controller = UIButtonInteractionController()
    bounds = [UIButtonBounds(100, 100, 300, 200)]

    controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="pinched",
        button_bounds=bounds,
    )
    releasing = controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="release_candidate",
        button_bounds=bounds,
    )
    released = controller.update(
        UIInputState(cursor_pixels=(150, 150), visible=True),
        pinch_state="open",
        button_bounds=bounds,
    )

    assert releasing.pressed_index == 0
    assert released.activated_index == 0


def test_ui_settings_state_clamps_values() -> None:
    settings = UISettingsState()

    settings.adjust_cursor_scale(200)
    settings.adjust_cursor_opacity(-200)
    settings.set_brightness(-25)
    settings.set_volume(160)
    settings.set_ui_cursor_scale_x(2.0)
    settings.set_ui_cursor_offset_y(-1.0)

    assert settings.cursor_scale == 2.0
    assert settings.cursor_opacity == 0.2
    assert settings.brightness == 0.0
    assert settings.brightness_scale == pytest.approx(0.2)
    assert settings.volume == 100.0
    assert settings.ui_cursor_scale_x == 1.5
    assert settings.ui_cursor_offset_y == -0.25


def test_rendering_updates_home_cursor_from_gesture_packet(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()

    service.update_gesture_data(make_packet_with_rotation())

    assert service._home_view.cursor_state is not None
    assert service._home_view.cursor_state.visible is True
    assert service._home_view.cursor_state.cursor_pixels == pytest.approx((520.0, 360.0))
    assert service._home_view.last_pinch_state == "pinched"


def test_rendering_updates_home_layout_before_cursor_mapping(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service._window_adapter.get_base().win.width = 2648
    service._window_adapter.get_base().win.height = 1490

    service.update_gesture_data(make_packet_with_rotation())

    assert service._home_view.last_window_size == (2648, 1490)
    assert service._home_view.layout_updates >= 1


def test_rendering_menu_opens_after_three_second_grab_hold(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")

    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=1, timestamp_ms=1_000))
    assert service.active_table_overlay == "none"

    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=2, timestamp_ms=3_999))
    assert service.active_table_overlay == "none"

    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=3, timestamp_ms=4_000))
    assert service.active_table_overlay == "menu"
    assert service.health()["stats"]["table_interaction_locked"] is True
    assert service._table_overlay_view.visible is True
    assert service._table_overlay_view.last_overlay == "menu"


def test_rendering_menu_opens_despite_cached_grabbed_object_state(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service._object_interaction_states["primary_cube"] = "grabbed"

    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=1, timestamp_ms=1_000))
    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=2, timestamp_ms=4_000))

    assert service.active_table_overlay == "menu"


def test_rendering_opening_overlay_clears_object_interaction_states(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    object_node = FakeObjectNode()
    service._object_cache["primary_cube"] = object_node
    service._object_interaction_states["primary_cube"] = "grabbed"

    service.set_active_table_overlay("menu", timestamp_ms=4_000)

    assert service._object_interaction_states["primary_cube"] == "idle"
    assert object_node.material is not None


def test_rendering_routes_ui_input_to_table_overlay_view(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.set_active_table_overlay("menu", timestamp_ms=4_000)

    service.update_gesture_data(make_packet_with_rotation())

    assert service._table_overlay_view.cursor_state is not None
    assert service._table_overlay_view.last_pinch_state == "pinched"
    assert service._table_overlay_view.last_overlay == "menu"


def test_rendering_table_overlay_buttons_drive_navigation(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.set_active_table_overlay("menu", timestamp_ms=4_000)

    service._table_overlay_view.on_button_activated("open_option")
    assert service.active_table_overlay == "option"

    service._table_overlay_view.on_button_activated("back_to_menu")
    assert service.active_table_overlay == "menu"

    service._table_overlay_view.on_button_activated("back_home")
    assert service.active_view == "home"
    assert service.active_table_overlay == "none"


def test_rendering_option_overlay_buttons_adjust_table_settings(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.set_active_table_overlay("option", timestamp_ms=4_000)
    service._data_panel = FakeVisibilityController()

    service._table_overlay_view.on_button_activated("toggle_data_panel")
    service._table_overlay_view.on_button_activated("toggle_cam_preview")
    service._table_overlay_view.on_button_activated("set_brightness:95")
    service._table_overlay_view.on_button_activated("set_volume:55")

    assert service._ui_settings.data_panel_enabled is False
    assert service._ui_settings.cam_preview_enabled is False
    assert service._ui_settings.brightness == 95.0
    assert service._ui_settings.volume == 55.0
    assert service._data_panel.panel_visible is False
    assert service._data_panel.indicator_visible is True


def test_rendering_option_overlay_toggles_object_visibility(monkeypatch, tmp_path) -> None:
    patch_ui_views(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.push(
        make_command(
            command_id="init-scene-visibility-1",
            command_type="init_scene",
            payload={
                "objects": [
                    {
                        "object_id": "primary_cube",
                        "init_pos": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    },
                    {
                        "object_id": "secondary_cube",
                        "init_pos": {"x": -0.1, "y": 0.0, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    },
                ]
            },
        )
    )
    service.set_active_table_overlay("option", timestamp_ms=4_000)

    service._table_overlay_view.on_button_activated("toggle_object_visibility:primary_cube")

    assert service._object_cache["primary_cube"].hidden is True
    assert service.health()["stats"]["object_visibility"] == {"primary_cube": False}
    assert service._table_overlay_view.last_object_items[0]["visible"] is False


def test_rendering_hidden_object_ignores_pose_and_state_updates(monkeypatch, tmp_path) -> None:
    patch_ui_views(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.push(
        make_command(
            command_id="init-scene-hidden-1",
            command_type="init_scene",
            payload={
                "objects": [
                    {
                        "object_id": "primary_cube",
                        "init_pos": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    }
                ]
            },
        )
    )

    initial_pos = service._object_cache["primary_cube"].pos
    service._set_object_visibility("primary_cube", False, persist=True)

    service.push(
        make_command(
            command_id="hidden-pose-1",
            command_type="set_object_pose",
            object_id="primary_cube",
            payload={"position": {"x": 0.9, "y": 0.9, "z": 0.9}},
        )
    )
    service.push(
        make_command(
            command_id="hidden-state-1",
            command_type="set_object_state",
            object_id="primary_cube",
            payload={"interaction_state": "grabbed"},
        )
    )

    assert service._object_cache["primary_cube"].pos == initial_pos
    assert service._object_interaction_states["primary_cube"] == "idle"
    assert service._object_cache["primary_cube"].hidden is True


def test_rendering_persists_object_visibility_across_restart(monkeypatch, tmp_path) -> None:
    patch_ui_views(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    first_service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    first_service.start()
    first_service.set_active_view("table")
    first_service.push(
        make_command(
            command_id="init-scene-persist-1",
            command_type="init_scene",
            payload={
                "objects": [
                    {
                        "object_id": "primary_cube",
                        "init_pos": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    }
                ]
            },
        )
    )
    first_service._set_object_visibility("primary_cube", False, persist=True)
    first_service.stop()

    second_service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    second_service.start()
    second_service.set_active_view("table")
    second_service.push(
        make_command(
            command_id="init-scene-persist-2",
            command_type="init_scene",
            payload={
                "objects": [
                    {
                        "object_id": "primary_cube",
                        "init_pos": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    }
                ]
            },
        )
    )

    assert second_service._object_cache["primary_cube"].hidden is True
    assert second_service.health()["stats"]["object_visibility"] == {"primary_cube": False}


def test_rendering_setting_view_receives_object_visibility_summary(monkeypatch, tmp_path) -> None:
    patch_ui_views(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")
    service.push(
        make_command(
            command_id="init-scene-summary-1",
            command_type="init_scene",
            payload={
                "objects": [
                    {
                        "object_id": "primary_cube",
                        "init_pos": {"x": 0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    },
                    {
                        "object_id": "secondary_cube",
                        "init_pos": {"x": -0.1, "y": 0.2, "z": 0.3},
                        "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
                        "shape": "cube",
                    },
                ]
            },
        )
    )

    assert service._setting_view.object_visibility_summary == (2, 0)

    service._set_object_visibility("secondary_cube", False, persist=True)

    assert service._setting_view.object_visibility_summary == (2, 1)


def test_rendering_menu_gate_ignores_rotation_mode_active(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")

    service.update_gesture_data(
        make_packet_with_menu_candidate(frame_id=1, timestamp_ms=1_000, grab_detected=True, mode_active=True)
    )
    service.update_gesture_data(
        make_packet_with_menu_candidate(frame_id=2, timestamp_ms=5_000, grab_detected=True, mode_active=True)
    )

    assert service.active_table_overlay == "none"


def test_rendering_menu_gate_respects_reopen_cooldown(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("table")

    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=1, timestamp_ms=1_000))
    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=2, timestamp_ms=4_000))
    assert service.active_table_overlay == "menu"

    service.set_active_table_overlay("none", timestamp_ms=4_010)
    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=3, timestamp_ms=4_100))
    service.update_gesture_data(make_packet_with_menu_candidate(frame_id=4, timestamp_ms=7_050))

    assert service.active_table_overlay == "none"


def test_rendering_home_button_callback_switches_view(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()

    service._home_view.on_button_activated("table")
    assert service.active_view == "table"

    service._home_view.on_button_activated("setting")
    assert service.active_view == "setting"


def test_rendering_setting_button_updates_ui_settings(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("setting")

    service._setting_view.on_button_activated("set_cursor_scale:1.37")
    service._setting_view.on_button_activated("set_cursor_opacity:0.61")
    service._setting_view.on_button_activated("set_brightness:48")
    service._setting_view.on_button_activated("set_volume:73")

    assert service.health()["stats"]["ui_settings"] == {
        "data_panel_enabled": True,
        "cam_preview_enabled": True,
        "cursor_scale": 1.37,
        "cursor_opacity": 0.61,
        "brightness": 48.0,
        "volume": 73.0,
        "ui_cursor_scale_x": 1.0,
        "ui_cursor_scale_y": 1.0,
        "ui_cursor_offset_x": 0.0,
        "ui_cursor_offset_y": 0.0,
        "calibration_profile_key": service._calibration_profile_key,
    }
    assert service._home_view.last_settings == (1.37, 0.61)
    assert service._setting_view.last_settings == (1.37, 0.61)


def test_rendering_brightness_and_volume_apply_to_window_roots(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    volume_updates: list[float] = []
    service.set_volume_callback(volume_updates.append)
    service.set_active_view("setting")

    service._setting_view.on_button_activated("set_brightness:25")
    service._setting_view.on_button_activated("set_volume:73")

    assert service._scene_root.color_scale == pytest.approx((0.4, 0.4, 0.4, 1.0))
    assert service._window_adapter.get_base().pixel2d.color_scale == pytest.approx((0.4, 0.4, 0.4, 1.0))
    assert service._window_adapter.get_base().background_color == pytest.approx((0.4, 0.4, 0.4, 1.0))
    assert volume_updates == [50.0, 73.0]


def test_rendering_setting_button_can_return_home(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("setting")

    service._setting_view.on_button_activated("back_home")

    assert service.active_view == "home"


def test_rendering_routes_ui_input_to_setting_view(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("setting")

    service.update_gesture_data(make_packet_with_rotation())

    assert service._setting_view.cursor_state is not None
    assert service._setting_view.last_pinch_state == "pinched"
    assert service._home_view.cursor_state is None


def test_rendering_routes_calibration_preview_to_calibration_view(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("calibration")

    service.update_gesture_data(make_packet_with_rotation())

    preview_state = service._calibration_view.calibration_preview_state
    assert preview_state is not None
    assert preview_state.visible is True
    assert preview_state.camera_midpoint == pytest.approx((0.35, 0.2))
    assert preview_state.source_cursor_norm == pytest.approx((0.325, 0.4))
    assert preview_state.source_cursor_pixels == pytest.approx((520.0, 360.0))
    assert preview_state.mapped_cursor_norm == pytest.approx((0.325, 0.4))
    assert preview_state.mapped_cursor_pixels == pytest.approx((520.0, 360.0))


def test_rendering_calibration_settings_affect_ui_cursor_mapping(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("calibration")
    service._calibration_view.on_button_activated("set_ui_cursor_scale_x:1.20")
    service._calibration_view.on_button_activated("set_ui_cursor_offset_x:0.10")

    service.update_gesture_data(make_packet_with_rotation())

    assert service._calibration_view.cursor_state.cursor_norm == pytest.approx((0.39, 0.4))
    assert service._calibration_view.cursor_state.cursor_pixels == pytest.approx((624.0, 360.0))
    preview_state = service._calibration_view.calibration_preview_state
    assert preview_state.source_cursor_norm == pytest.approx((0.325, 0.4))
    assert preview_state.mapped_cursor_norm == pytest.approx((0.39, 0.4))


def test_rendering_setting_button_opens_calibration(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("setting")

    service._setting_view.on_button_activated("open_calibration")

    assert service.active_view == "calibration"
    assert service._calibration_view.visible is True


def test_rendering_registers_calibration_shortcuts(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()

    accepted_events = service._window_adapter.get_base().accepted_events
    assert set(accepted_events) >= {
        "f2",
        "escape",
        "tab",
        "shift-tab",
        "arrow_left",
        "arrow_right",
        "arrow_up",
        "arrow_down",
        "shift-arrow_left",
        "shift-arrow_right",
        "shift-arrow_up",
        "shift-arrow_down",
        "r",
        "enter",
    }


def test_rendering_f2_opens_calibration_and_escape_returns(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    accepted_events = service._window_adapter.get_base().accepted_events

    accepted_events["f2"]()
    assert service.active_view == "calibration"

    accepted_events["escape"]()
    assert service.active_view == "setting"


def test_rendering_keyboard_adjusts_selected_calibration_parameter(monkeypatch) -> None:
    patch_ui_views(monkeypatch)

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    accepted_events = service._window_adapter.get_base().accepted_events
    accepted_events["f2"]()

    accepted_events["shift-tab"]()
    assert service._calibration_view.selected_parameter_key == "ui_cursor_offset_y"

    accepted_events["shift-arrow_left"]()
    assert service.health()["stats"]["ui_settings"]["ui_cursor_offset_y"] == -0.1

    accepted_events["r"]()
    assert service.health()["stats"]["ui_settings"]["ui_cursor_offset_y"] == 0.0


def test_rendering_persists_calibration_settings_per_device(monkeypatch, tmp_path) -> None:
    patch_ui_views(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    service.start()
    service.set_active_view("calibration")
    service._calibration_view.on_button_activated("set_ui_cursor_scale_x:1.24")
    service._calibration_view.on_button_activated("set_ui_cursor_offset_y:-0.08")
    service.stop()

    second_service = RenderingServiceImpl(window_adapter_factory=FakeWindowAdapter)
    second_service.start()

    ui_settings = second_service.health()["stats"]["ui_settings"]
    assert ui_settings["ui_cursor_scale_x"] == 1.24
    assert ui_settings["ui_cursor_offset_y"] == -0.08


def test_rendering_core_registers_quit_shortcuts() -> None:
    manager = RenderingCoreManager()
    fake_base = FakeBase()
    manager._base = fake_base

    manager.set_quit_handler(lambda: None)

    assert set(fake_base.accepted_events) >= {"meta-w", "meta-q", "control-w", "control-q"}
