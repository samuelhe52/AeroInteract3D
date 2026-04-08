from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from .ui.state import UISettingsState


class CalibrationSettingsStore:
    def __init__(self, file_path: Path | None = None) -> None:
        self._file_path = file_path or self._default_file_path()

    @staticmethod
    def _default_file_path() -> Path:
        config_root = os.environ.get("XDG_CONFIG_HOME")
        if config_root:
            return Path(config_root) / "AeroInteract3D" / "calibration_profiles.json"
        return Path.home() / ".config" / "AeroInteract3D" / "calibration_profiles.json"

    @staticmethod
    def current_profile_key() -> str:
        host_name = platform.node().strip() or "unknown-host"
        system_name = platform.system().strip().lower() or "unknown-os"
        return f"{system_name}:{host_name}"

    def load_into(self, settings: UISettingsState, profile_key: str | None = None) -> bool:
        payload = self._read_payload()
        if payload is None:
            return False
        key = profile_key or self.current_profile_key()
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            return False
        profile = profiles.get(key)
        if not isinstance(profile, dict):
            return False
        try:
            settings.set_ui_cursor_scale_x(float(profile.get("ui_cursor_scale_x", 1.0)))
            settings.set_ui_cursor_scale_y(float(profile.get("ui_cursor_scale_y", 1.0)))
            settings.set_ui_cursor_offset_x(float(profile.get("ui_cursor_offset_x", 0.0)))
            settings.set_ui_cursor_offset_y(float(profile.get("ui_cursor_offset_y", 0.0)))
        except (TypeError, ValueError):
            return False
        return True

    def save_from(self, settings: UISettingsState, profile_key: str | None = None) -> None:
        key = profile_key or self.current_profile_key()
        payload = self._read_payload() or {"version": 1, "profiles": {}}
        profiles = payload.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            payload["profiles"] = {}
            profiles = payload["profiles"]
        profiles[key] = {
            "ui_cursor_scale_x": settings.ui_cursor_scale_x,
            "ui_cursor_scale_y": settings.ui_cursor_scale_y,
            "ui_cursor_offset_x": settings.ui_cursor_offset_x,
            "ui_cursor_offset_y": settings.ui_cursor_offset_y,
        }
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_payload(self) -> dict | None:
        if not self._file_path.exists():
            return None
        try:
            raw = self._file_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
