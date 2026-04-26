from __future__ import annotations

import logging
from typing import Any

from .rendering_core import MAIN_MENU_BACKGROUND_COLOR
from .ui import RenderView, TableOverlay
from .ui.display_metrics import clamp_display_scale, logical_size_from_physical


logger = logging.getLogger("rendering_service")

TABLE_MENU_HOLD_MS = 3000
TABLE_MENU_COOLDOWN_MS = 200
TABLE_MENU_MAX_CURSOR_DRIFT_NORM = 0.08
TABLE_OPTION_STEP = 5.0


class RenderingServiceUIMixin:
    def _handle_scale_change(self, scale: float) -> None:
        """Handle UI scale changes"""
        if self._data_panel:
            self._data_panel.set_ui_scale(scale)
        if self._camera_preview:
            self._camera_preview.set_ui_scale(scale)
        if self._home_view:
            self._home_view.update_layout(force=True)
        if self._setting_view:
            self._setting_view.update_layout(force=True)
        if self._calibration_view:
            self._calibration_view.update_layout(force=True)
        if self._table_overlay_view:
            self._table_overlay_view.update_layout(force=True)

    def _display_scale(self) -> float:
        if self._rendering_core is None:
            return 1.0
        display_scale = getattr(self._rendering_core, "display_scale", None)
        if callable(display_scale):
            return clamp_display_scale(display_scale())
        return 1.0

    def _handle_table_overlay_button_activated(self, action: str) -> None:
        if action == "return_to_table":
            self.set_active_table_overlay(TableOverlay.NONE)
            return
        if action == "open_option":
            self.set_active_table_overlay(TableOverlay.OPTION)
            return
        if action == "back_to_menu":
            self.set_active_table_overlay(TableOverlay.MENU)
            return
        if action == "back_home":
            self.set_active_view(RenderView.HOME)
            return
        if action == "toggle_data_panel":
            self._ui_settings.data_panel_enabled = not self._ui_settings.data_panel_enabled
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action == "toggle_cam_preview":
            self._ui_settings.cam_preview_enabled = not self._ui_settings.cam_preview_enabled
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action.startswith("toggle_object_visibility:"):
            object_id = action.split(":", 1)[1].strip()
            if object_id:
                self._set_object_visibility(object_id, not self._is_object_visible(object_id), persist=True)
            return
        if action == "decrease_brightness":
            self._ui_settings.set_brightness(self._ui_settings.brightness - TABLE_OPTION_STEP)
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action == "increase_brightness":
            self._ui_settings.set_brightness(self._ui_settings.brightness + TABLE_OPTION_STEP)
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action == "decrease_volume":
            self._ui_settings.set_volume(self._ui_settings.volume - TABLE_OPTION_STEP)
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action == "increase_volume":
            self._ui_settings.set_volume(self._ui_settings.volume + TABLE_OPTION_STEP)
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if self._apply_setting_action(action):
            return
        logger.warning("Unknown table overlay button action ignored: %s", action)

    def _register_calibration_shortcuts(self) -> None:
        if self._rendering_core is None:
            return
        base = self._rendering_core.get_base()
        accept = getattr(base, "accept", None) if base is not None else None
        if not callable(accept):
            return
        accept("f2", self._open_calibration_shortcut)
        accept("escape", self._exit_calibration_shortcut)
        accept("tab", self._focus_next_calibration_parameter)
        accept("shift-tab", self._focus_previous_calibration_parameter)
        accept("arrow_left", self._adjust_calibration_parameter, [-1])
        accept("arrow_down", self._adjust_calibration_parameter, [-1])
        accept("arrow_right", self._adjust_calibration_parameter, [1])
        accept("arrow_up", self._adjust_calibration_parameter, [1])
        accept("shift-arrow_left", self._adjust_calibration_parameter, [-10])
        accept("shift-arrow_down", self._adjust_calibration_parameter, [-10])
        accept("shift-arrow_right", self._adjust_calibration_parameter, [10])
        accept("shift-arrow_up", self._adjust_calibration_parameter, [10])
        accept("r", self._reset_calibration_shortcut)
        accept("enter", self._confirm_calibration_shortcut)

    def _window_size(self) -> tuple[int, int]:
        if self._rendering_core is None:
            return (1600, 900)

        base = self._rendering_core.get_base()
        win = getattr(base, "win", None) if base is not None else None
        if win is None:
            return (1600, 900)

        get_x_size = getattr(win, "getXSize", None)
        get_y_size = getattr(win, "getYSize", None)
        if not callable(get_x_size) or not callable(get_y_size):
            return (1600, 900)

        return logical_size_from_physical(
            (int(get_x_size()), int(get_y_size())),
            self._display_scale(),
        )

    @property
    def active_view(self) -> str:
        return self._view_state.active_view.value

    @property
    def active_table_overlay(self) -> str:
        return self._table_overlay_state.active_overlay.value

    def set_active_view(self, view: RenderView | str) -> str:
        next_view = self._view_state.set_active_view(view)
        if next_view != RenderView.TABLE:
            self._reset_table_overlay_runtime_state(clear_cooldown=True)
        self._sync_view_visibility()
        self._sync_table_menu_hold_feedback(self._last_gesture_packet)
        logger.info("Rendering view switched to %s", next_view.value)
        return next_view.value

    def set_active_table_overlay(self, overlay: TableOverlay | str, *, timestamp_ms: int | None = None) -> str:
        next_overlay = TableOverlay(overlay)
        if self._view_state.active_view != RenderView.TABLE and next_overlay != TableOverlay.NONE:
            logger.warning("Ignoring table overlay change outside table view: %s", next_overlay.value)
            return self._table_overlay_state.active_overlay.value

        current_overlay = self._table_overlay_state.active_overlay
        if current_overlay == next_overlay:
            return current_overlay.value

        if next_overlay != TableOverlay.NONE:
            self._clear_table_object_interaction_states()

        self._table_overlay_state.set_active_overlay(next_overlay, opened_at_ms=timestamp_ms)
        self._clear_table_menu_hold()
        self._sync_view_visibility()
        self._sync_table_menu_hold_feedback(self._last_gesture_packet)
        logger.info("Table overlay switched to %s", next_overlay.value)
        return next_overlay.value

    def _handle_home_button_activated(self, action: str) -> None:
        if action == RenderView.TABLE.value:
            self.set_active_view(RenderView.TABLE)
            return
        if action == RenderView.SETTING.value:
            self.set_active_view(RenderView.SETTING)
            return
        logger.warning("Unknown home button action ignored: %s", action)

    def _handle_setting_button_activated(self, action: str) -> None:
        if action == "back_home":
            self.set_active_view(RenderView.HOME)
            return
        if action == "open_calibration":
            self.set_active_view(RenderView.CALIBRATION)
            return
        if action == "data_panel_toggle":
            self._ui_settings.data_panel_enabled = not self._ui_settings.data_panel_enabled
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if action == "cam_preview_toggle":
            self._ui_settings.cam_preview_enabled = not self._ui_settings.cam_preview_enabled
            self._save_ui_settings()
            self._apply_ui_settings_to_views()
            self._sync_view_visibility()
            return
        if self._apply_setting_action(action):
            return
        logger.warning("Unknown setting button action ignored: %s", action)

    def _handle_calibration_button_activated(self, action: str) -> None:
        if action == "back_setting":
            self.set_active_view(RenderView.SETTING)
            return
        if action == "reset_calibration":
            self._reset_calibration_settings()
            return
        if self._apply_setting_action(action):
            return
        logger.warning("Unknown calibration button action ignored: %s", action)

    def _reset_calibration_settings(self) -> None:
        self._ui_settings.set_ui_cursor_scale_x(1.0)
        self._ui_settings.set_ui_cursor_scale_y(1.0)
        self._ui_settings.set_ui_cursor_offset_x(0.0)
        self._ui_settings.set_ui_cursor_offset_y(0.0)
        self._save_calibration_settings()
        self._apply_ui_settings_to_views()
        self._sync_view_visibility()

    def _save_calibration_settings(self) -> None:
        self._calibration_store.save_from(self._ui_settings, self._calibration_profile_key)

    def _save_ui_settings(self) -> None:
        self._calibration_store.save_from(self._ui_settings, self._calibration_profile_key)

    def _apply_setting_action(self, action: str) -> bool:
        if not action.startswith("set_") or ":" not in action:
            return False
        setting_key, raw_value = action.split(":", 1)
        setting_key = setting_key[4:]
        try:
            parsed_value = float(raw_value)
        except ValueError:
            logger.warning("Invalid setting action value ignored: %s", action)
            return True
        if setting_key == "cursor_scale":
            self._ui_settings.set_cursor_scale(parsed_value)
        elif setting_key == "cursor_opacity":
            self._ui_settings.set_cursor_opacity(parsed_value)
        elif setting_key == "brightness":
            self._ui_settings.set_brightness(parsed_value)
        elif setting_key == "volume":
            self._ui_settings.set_volume(parsed_value)
        elif setting_key == "ui_cursor_scale_x":
            self._ui_settings.set_ui_cursor_scale_x(parsed_value)
        elif setting_key == "ui_cursor_scale_y":
            self._ui_settings.set_ui_cursor_scale_y(parsed_value)
        elif setting_key == "ui_cursor_offset_x":
            self._ui_settings.set_ui_cursor_offset_x(parsed_value)
        elif setting_key == "ui_cursor_offset_y":
            self._ui_settings.set_ui_cursor_offset_y(parsed_value)
        else:
            logger.warning("Unknown setting slider action ignored: %s", action)
            return True
        self._save_ui_settings()
        self._apply_ui_settings_to_views()
        self._sync_view_visibility()
        return True

    def _open_calibration_shortcut(self) -> None:
        self.set_active_view(RenderView.CALIBRATION)

    def _exit_calibration_shortcut(self) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION:
            self.set_active_view(RenderView.SETTING)

    def _confirm_calibration_shortcut(self) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION:
            self.set_active_view(RenderView.SETTING)

    def _focus_next_calibration_parameter(self) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION and self._calibration_view is not None:
            self._calibration_view.select_next_parameter(1)

    def _focus_previous_calibration_parameter(self) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION and self._calibration_view is not None:
            self._calibration_view.select_next_parameter(-1)

    def _adjust_calibration_parameter(self, step_count: int) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION and self._calibration_view is not None:
            self._calibration_view.adjust_selected_parameter(step_count)

    def _reset_calibration_shortcut(self) -> None:
        if self._view_state.active_view == RenderView.CALIBRATION:
            self._reset_calibration_settings()

    def _apply_ui_settings_to_views(self) -> None:
        self._ui_input_adapter.set_calibration_settings(self._ui_settings)
        if self._home_view:
            self._home_view.set_ui_settings(self._ui_settings)
        if self._setting_view:
            self._setting_view.set_ui_settings(self._ui_settings)
            visibility_summary = self._table_object_visibility_summary()
            set_object_visibility_summary = getattr(self._setting_view, "set_object_visibility_summary", None)
            if callable(set_object_visibility_summary):
                set_object_visibility_summary(
                    visibility_summary["total_count"],
                    visibility_summary["hidden_count"],
                )
        if self._calibration_view:
            self._calibration_view.set_ui_settings(self._ui_settings)
        if self._table_overlay_view:
            self._table_overlay_view.set_ui_settings(self._ui_settings)
            self._table_overlay_view.set_object_visibility_items(self._table_object_visibility_items())
        self._apply_window_brightness()
        self._apply_volume_setting()

    def _table_object_visibility_items(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for object_id in self._object_cache:
            display_label = None
            object_node = self._object_cache.get(object_id)
            model_factory = getattr(self, "_model_factory", None)
            if object_node is not None and model_factory is not None:
                shape_id = ""
                get_tag = getattr(object_node, "getTag", None)
                if callable(get_tag):
                    shape_id = str(get_tag("shape") or "").strip().lower()
                elif isinstance(getattr(object_node, "tags", None), dict):
                    shape_id = str(object_node.tags.get("shape", "")).strip().lower()
                if shape_id:
                    get_display_name = getattr(model_factory, "get_display_name", None)
                    if callable(get_display_name):
                        display_label = get_display_name(shape_id)
            items.append(
                {
                    "object_id": object_id,
                    "label": display_label or object_id.replace("_", " "),
                    "visible": self._is_object_visible(object_id),
                }
            )
        return items

    def _table_object_visibility_summary(self) -> dict[str, int]:
        total_count = len(self._object_cache)
        hidden_count = sum(1 for object_id in self._object_cache if not self._is_object_visible(object_id))
        return {
            "total_count": total_count,
            "hidden_count": hidden_count,
        }

    def _is_object_visible(self, object_id: str) -> bool:
        return self._object_visibility_by_id.get(object_id, True)

    def _save_object_visibility_settings(self) -> None:
        self._object_visibility_store.save(self._object_visibility_by_id)

    def _apply_object_visibility(self, object_id: str) -> None:
        obj_np = self._object_cache.get(object_id)
        if obj_np is None:
            return
        visible = self._is_object_visible(object_id)
        if visible:
            show = getattr(obj_np, "show", None)
            if callable(show):
                show()
            state = self._object_interaction_states.get(object_id, "idle")
            self._apply_object_visual_state(object_id, state)
            return

        hide = getattr(obj_np, "hide", None)
        if callable(hide):
            hide()
        self._object_interaction_states[object_id] = "idle"
        clear_material = getattr(obj_np, "clearMaterial", None)
        if callable(clear_material):
            clear_material()

    def _set_object_visibility(self, object_id: str, visible: bool, *, persist: bool) -> None:
        self._object_visibility_by_id[object_id] = bool(visible)
        self._apply_object_visibility(object_id)
        if persist:
            self._save_object_visibility_settings()
        if self._setting_view:
            visibility_summary = self._table_object_visibility_summary()
            set_object_visibility_summary = getattr(self._setting_view, "set_object_visibility_summary", None)
            if callable(set_object_visibility_summary):
                set_object_visibility_summary(
                    visibility_summary["total_count"],
                    visibility_summary["hidden_count"],
                )
        if self._table_overlay_view:
            self._table_overlay_view.set_object_visibility_items(self._table_object_visibility_items())

    def _sync_view_visibility(self) -> None:
        home_visible = self._view_state.active_view == RenderView.HOME
        setting_visible = self._view_state.active_view == RenderView.SETTING
        calibration_visible = self._view_state.active_view == RenderView.CALIBRATION
        table_visible = self._view_state.active_view == RenderView.TABLE

        if self._home_view:
            self._home_view.set_visible(home_visible)
        if self._setting_view:
            self._setting_view.set_visible(setting_visible)
        if self._calibration_view:
            self._calibration_view.set_visible(calibration_visible)
        if self._table_overlay_view:
            self._table_overlay_view.set_overlay(self._table_overlay_state.active_overlay)
            self._table_overlay_view.set_visible(table_visible and self._table_overlay_state.active_overlay != TableOverlay.NONE)

        if self._scene_root is not None and not self._scene_root.isEmpty():
            self._scene_root.show() if table_visible else self._scene_root.hide()

        if self._data_panel:
            set_panel_visible = getattr(self._data_panel, "set_panel_visible", None)
            set_indicator_visible = getattr(self._data_panel, "set_indicator_visible", None)
            if callable(set_panel_visible):
                set_panel_visible(table_visible and self._ui_settings.data_panel_enabled)
            else:
                self._data_panel.set_visible(table_visible and self._ui_settings.data_panel_enabled)
            if callable(set_indicator_visible):
                set_indicator_visible(table_visible)

        if self._camera_preview:
            self._camera_preview.set_visible(table_visible and self._ui_settings.cam_preview_enabled)

    def _clear_table_menu_hold(self) -> None:
        self._table_menu_hold_started_at_ms = None
        self._table_menu_hold_origin_norm = None

    def _sync_table_menu_hold_feedback(self, packet) -> None:
        if self._data_panel is None:
            return
        update_menu_hold_progress = getattr(self._data_panel, "update_menu_hold_progress", None)
        if not callable(update_menu_hold_progress):
            return
        overlay_active = (
            self._view_state.active_view == RenderView.TABLE
            and self._table_overlay_state.active_overlay != TableOverlay.NONE
        )
        candidate_active = (
            self._view_state.active_view == RenderView.TABLE
            and self._table_overlay_state.active_overlay == TableOverlay.NONE
            and self._table_menu_hold_started_at_ms is not None
            and packet is not None
        )
        hold_ms = 0
        if candidate_active:
            hold_ms = max(int(getattr(packet, "timestamp_ms", 0) or 0) - self._table_menu_hold_started_at_ms, 0)
        update_menu_hold_progress(
            hold_ms,
            candidate_active=candidate_active,
            overlay_active=overlay_active,
        )

    def _clear_table_object_interaction_states(self) -> None:
        for object_id, state in list(self._object_interaction_states.items()):
            if state == "idle":
                continue
            if object_id not in self._object_cache:
                self._object_interaction_states[object_id] = "idle"
                continue
            self._apply_object_visual_state(object_id, "idle")
            self._object_interaction_states[object_id] = "idle"

    def _reset_table_overlay_runtime_state(self, *, clear_cooldown: bool) -> None:
        self._table_overlay_state.set_active_overlay(TableOverlay.NONE)
        if clear_cooldown:
            self._table_overlay_state.trigger_cooldown_until_ms = 0
        self._clear_table_menu_hold()

    def _is_table_interaction_locked(self) -> bool:
        return self._view_state.active_view == RenderView.TABLE and self._table_overlay_state.active_overlay != TableOverlay.NONE

    def _rotation_debug_payload(self, packet) -> dict[str, Any] | None:
        debug_payload = getattr(packet, "debug", None)
        if not isinstance(debug_payload, dict):
            return None
        rotation_payload = debug_payload.get("rotation")
        if not isinstance(rotation_payload, dict):
            return None
        return rotation_payload

    def _table_menu_candidate_active(self, packet, ui_input) -> bool:
        if packet is None or getattr(packet, "tracking_state", None) != "tracked" or not bool(ui_input.visible):
            return False

        rotation_payload = self._rotation_debug_payload(packet)
        if rotation_payload is None:
            return False
        if not bool(rotation_payload.get("grab_detected", False)):
            return False
        if bool(rotation_payload.get("mode_active", False)):
            return False
        return True

    @staticmethod
    def _cursor_norm_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        delta_x = float(left[0]) - float(right[0])
        delta_y = float(left[1]) - float(right[1])
        return (delta_x * delta_x + delta_y * delta_y) ** 0.5

    def _update_table_menu_hold_gate(self, packet, ui_input) -> None:
        if self._view_state.active_view != RenderView.TABLE:
            self._clear_table_menu_hold()
            return

        if self._table_overlay_state.active_overlay != TableOverlay.NONE:
            self._clear_table_menu_hold()
            return

        if packet is None:
            self._clear_table_menu_hold()
            return

        timestamp_ms = int(getattr(packet, "timestamp_ms", 0) or 0)
        if timestamp_ms < self._table_overlay_state.trigger_cooldown_until_ms:
            self._clear_table_menu_hold()
            return

        if not self._table_menu_candidate_active(packet, ui_input):
            self._clear_table_menu_hold()
            return

        if self._table_menu_hold_started_at_ms is None or self._table_menu_hold_origin_norm is None:
            self._table_menu_hold_started_at_ms = timestamp_ms
            self._table_menu_hold_origin_norm = ui_input.cursor_norm
            return

        if self._cursor_norm_distance(ui_input.cursor_norm, self._table_menu_hold_origin_norm) > TABLE_MENU_MAX_CURSOR_DRIFT_NORM:
            self._table_menu_hold_started_at_ms = timestamp_ms
            self._table_menu_hold_origin_norm = ui_input.cursor_norm
            return

        if timestamp_ms - self._table_menu_hold_started_at_ms < TABLE_MENU_HOLD_MS:
            return

        self._table_overlay_state.trigger_cooldown_until_ms = timestamp_ms + TABLE_MENU_COOLDOWN_MS
        self.set_active_table_overlay(TableOverlay.MENU, timestamp_ms=timestamp_ms)

    def _apply_window_brightness(self) -> None:
        brightness = self._ui_settings.brightness_scale

        if self._scene_root is not None and not self._scene_root.isEmpty():
            set_color_scale = getattr(self._scene_root, "setColorScale", None)
            if callable(set_color_scale):
                set_color_scale(brightness, brightness, brightness, 1.0)

        virtual_hand_root = getattr(getattr(self, "_virtual_hand", None), "root", None)
        if virtual_hand_root is not None:
            set_color_scale = getattr(virtual_hand_root, "setColorScale", None)
            if callable(set_color_scale):
                set_color_scale(brightness, brightness, brightness, 1.0)

        if self._rendering_core is None:
            return

        base = self._rendering_core.get_base()
        if base is None:
            return

        set_background = getattr(base, "setBackgroundColor", None)
        if callable(set_background):
            set_background(*MAIN_MENU_BACKGROUND_COLOR, 1.0)

        pixel2d = getattr(base, "pixel2d", None)
        set_color_scale = getattr(pixel2d, "setColorScale", None)
        if callable(set_color_scale):
            set_color_scale(brightness, brightness, brightness, 1.0)

    def _apply_volume_setting(self, *, force: bool = False) -> None:
        if self._volume_callback is None:
            self._last_applied_volume = None
            return

        next_volume = float(self._ui_settings.volume)
        if not force and self._last_applied_volume == next_volume:
            return

        self._volume_callback(next_volume)
        self._last_applied_volume = next_volume
