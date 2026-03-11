from __future__ import annotations


def main() -> int:
    from src.gesture.debug.live_preview import main as live_preview_main

    return live_preview_main()


__all__ = ["main"]