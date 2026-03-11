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
