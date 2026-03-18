# Developer Setup (uv)

This project uses **uv** as the canonical Python environment and dependency manager.
Use the root `Makefile` as the primary command surface for day-to-day development.

## Prerequisites

- Python `3.12`
- `uv` installed

## First-time setup

From the repository root:

```bash
make setup
```

This creates/updates `.venv` and installs:

- runtime dependencies from `[project.dependencies]`
- development tooling from `dev`

## Daily workflow

- Sync after dependency changes:

```bash
make sync
```

- Run the app:

```bash
make run
```

- Run the app with forwarded CLI args:

```bash
make run -- --target-fps 30 --camera-index 1
```

- Use machine-local default run options:

```bash
cp .run.example.yaml .run.yaml
```

- Run the live gesture preview:

```bash
make preview
```

Or directly:

```bash
uv run python -m src.gesture.debug
```

- Run tests:

```bash
make test
```

- Run lint:

```bash
make lint
```

- Regenerate lockfile after dependency changes:

```bash
make lock
```

- See all commands:

```bash
make help
```

Equivalent direct uv commands remain valid if needed.

## Runtime configuration

The main app entrypoint in `main.py` currently supports:

- `--run-config`
- `--no-run-config`
- `--camera-index`
- `--target-fps`
- `--frame-width`
- `--frame-height`

Current defaults are `30 FPS` and `1280x960` requested capture resolution.

If `.run.yaml` exists at the repository root, `make run` and direct `uv run python main.py`
invocations automatically use it as the default source for these options. Command-line flags
still win over file values. The real `.run.yaml` is gitignored; start from `.run.example.yaml`
and adjust it per machine. Use `--no-run-config` to ignore the local file for one run.
Use `--no-debug-stats` or `--no-aggressive-release-guard` when you need to override a boolean
enabled in the local file. The main entrypoint no longer supports the old `live_preview` key or a
separate OpenCV preview window.

The gesture live preview uses the same capture configuration shape through `GesturePreviewConfig` in `src/gesture/debug/live_preview.py`.

## Dependency management rules

- Add runtime dependency:

```bash
uv add <package>
```

- Add dev dependency:

```bash
uv add --group dev <package>
```

- Add rendering dependency:

```bash
uv add --group rendering <package>
```

- Remove dependency:

```bash
uv remove <package>
```

After any intentional dependency change, commit both:

- `pyproject.toml`
- `uv.lock`

## Notes for rendering on macOS

The rendering group installs Panda3D (`panda3d`).
Runtime behavior still depends on your OS graphics stack and windowing environment.
