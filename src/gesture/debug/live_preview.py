from __future__ import annotations

import sys


if __package__ in {None, ""}:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.gesture.service_impl import (
    GesturePreviewConfig,
    build_config,
    build_preview_config,
    build_service,
    main,
    parse_args,
    setup_logging,
)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))