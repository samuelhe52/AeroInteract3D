# AeroInteract3D

[中文版本 / Chinese Version](README.zh-CN.md)

AeroInteract3D is a webcam-driven 3D interaction app. It lets you manipulate objects in a tabletop scene using hand gestures in front of your camera.

## What It Does

In the current build, you can:

- use one hand to hover, grab, move, and release objects
- use two hands to scale supported objects
- work inside a 3D tabletop scene with multiple props
- see the live camera feed inside the app window
- open the built-in settings and calibration views
- adjust local run defaults with a simple config file

The live pipeline also recovers from repeated camera read failures, stabilizes hover selection
between nearby props, and keeps two-hand scaling within scene-safe bounds.

## Quick Start

Requirements:

- Python `3.12`
- `uv`
- a working webcam
- macOS graphics support compatible with Panda3D

Install dependencies:

```bash
make setup
```

Run the app:

```bash
make run
```

To use a different camera or change runtime settings for one launch:

```bash
make run -- --camera-index 1
```

## Local Configuration

If you want persistent machine-local defaults, create a root-level `.run.yaml` from the template:

```bash
cp .run.example.yaml .run.yaml
```

This file is the right place to keep your preferred camera index, mirroring, frame size, FPS, and related runtime options.

## Main Commands

```bash
make run
make preview
make test
make lint
```

`make preview` starts the gesture-only live preview. `make run` starts the full 3D application.

## Project Layout

- [`main.py`](/Users/samuelhe/projects/AeroInteract3D/main.py): app entrypoint
- [`Makefile`](/Users/samuelhe/projects/AeroInteract3D/Makefile): common commands
- [`DEVELOPMENT.md`](/Users/samuelhe/projects/AeroInteract3D/DEVELOPMENT.md): developer setup
- [`assets/custom_models/`](/Users/samuelhe/projects/AeroInteract3D/assets/custom_models): scene models
- [`src/`](/Users/samuelhe/projects/AeroInteract3D/src): application code
- [`tests/`](/Users/samuelhe/projects/AeroInteract3D/tests): tests

## Notes

- The main app shows the camera preview inside the Panda3D window.
- Camera capture requests a low-latency single-frame backend buffer and automatically reopens after repeated read failures.
- Invalid runtime ranges are rejected before camera or rendering initialization.
- Calibration and UI settings are stored per machine under `~/.config/AeroInteract3D/calibration_profiles.json` unless `XDG_CONFIG_HOME` is set.
- For development details, see [DEVELOPMENT.md](DEVELOPMENT.md).
