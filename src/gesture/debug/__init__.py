from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from src.gesture.debug.live_preview import main as live_preview_main

    return live_preview_main(argv)


__all__ = ["main"]