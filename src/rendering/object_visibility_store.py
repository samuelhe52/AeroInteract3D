from __future__ import annotations

import json
import os
from pathlib import Path


class ObjectVisibilityStore:
    def __init__(self, file_path: Path | None = None) -> None:
        self._file_path = file_path or self._default_file_path()

    @staticmethod
    def _default_file_path() -> Path:
        config_root = os.environ.get("XDG_CONFIG_HOME")
        if config_root:
            return Path(config_root) / "AeroInteract3D" / "object_visibility.json"
        return Path.home() / ".config" / "AeroInteract3D" / "object_visibility.json"

    def load(self) -> dict[str, bool]:
        payload = self._read_payload()
        if payload is None:
            return {}
        values = payload.get("visible_by_object_id")
        if not isinstance(values, dict):
            return {}
        return {str(object_id): bool(visible) for object_id, visible in values.items()}

    def save(self, visible_by_object_id: dict[str, bool]) -> None:
        payload = {
            "version": 1,
            "visible_by_object_id": {
                str(object_id): bool(visible)
                for object_id, visible in sorted(visible_by_object_id.items())
            },
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