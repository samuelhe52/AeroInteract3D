# Custom Model Import

This document describes the current custom model import workflow for the rendering module.

It covers the behavior that is already implemented today:

- zero-config model discovery
- optional sidecar JSON overrides
- duplicate name conflict handling
- reserved built-in model names

## Quick Start

To use a custom model:

1. Put the model file into `assets/custom_models`
2. Use the file name without extension as the `shape` value in your scene object
3. Optionally add a sidecar JSON file if you need template-level adjustments

Example:

- model file: `assets/custom_models/teapot.glb`
- shape id: `teapot`

Scene object example:

```python
{
    "object_id": "demo_teapot",
    "shape": "teapot",
    "init_pos": {"x": 0.0, "y": -0.08, "z": 0.18},
    "init_hpr": {"h": 0.0, "p": 0.0, "r": 0.0},
    "scale": {"x": 0.2, "y": 0.2, "z": 0.2},
    "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    "interactable": True,
}
```

## Supported Model Formats

The auto-scan currently supports top-level files in `assets/custom_models` with these extensions:

- `.glb`
- `.egg`
- `.bam`

Subdirectories are not scanned in the current version.

## Zero-Config Import

If you only place a model file in `assets/custom_models`, the renderer will still register it.

Default behavior for auto-scanned custom models:

- `shape_id` = file name without extension, converted to lower case
- `display_name` = same as `shape_id`
- `center_offset` = `(0.0, 0.0, 0.0)`
- `default_scale` = `(1.0, 1.0, 1.0)`
- `two_sided` = `false`
- `use_builtin_materials` = `false`

This path is intended for the fastest possible import flow.

## Optional Sidecar JSON

If you need to adjust the imported model without editing Python code, add a sidecar JSON file next to the model.

Naming rule:

- model file: `teapot.glb`
- sidecar file: `teapot.model.json`

The sidecar file is optional.

If the sidecar file is missing, the model still loads through the zero-config path.

## Supported Sidecar Fields

The current implementation supports these fields:

### `display_name`

- type: string
- optional: yes
- default: `shape_id`

Use this to provide a friendlier display label for logs or future UI surfaces.

### `default_scale`

- type: array of 3 positive numbers
- optional: yes
- default: `[1.0, 1.0, 1.0]`

This value is multiplied into the instance scale at creation time.

### `center_offset`

- type: array of 3 numbers
- optional: yes
- default: `[0.0, 0.0, 0.0]`

Use this to correct model anchor or pivot offset.

### `two_sided`

- type: boolean
- optional: yes
- default: `false`

Use this for thin geometry or one-sided surfaces that should remain visible from both sides.

### `use_builtin_materials`

- type: boolean
- optional: yes
- default: `false`

When `false`, imported assets keep their authored textures and materials.

## Sidecar Examples

Minimal example:

```json
{
  "default_scale": [0.18, 0.18, 0.18],
  "center_offset": [0.0, 0.0, -0.08]
}
```

Full example for the current implementation:

```json
{
  "display_name": "Orange Pyramid",
  "default_scale": [0.15, 0.15, 0.15],
  "center_offset": [0.0, 0.0, 0.0],
  "two_sided": false,
  "use_builtin_materials": false
}
```

## Invalid Sidecar Behavior

The current implementation uses tolerant parsing.

- Missing sidecar: ignored
- Invalid JSON: warning only, model still registers
- Invalid field value: that field is ignored, other valid fields still apply
- Unknown fields: ignored

This prevents a single config mistake from breaking the entire model import flow.

## Name Conflicts

The current version does not try to auto-resolve duplicate names.

Conflict rules:

- `shape_id` is derived from the file name without extension, converted to lower case
- if two custom model files map to the same `shape_id`, that `shape_id` is rejected
- if a custom model conflicts with a built-in shape name, it is rejected
- the conflicting `shape_id` is not registered

Examples:

- `teapot.glb` and `teapot.egg` conflict with each other
- `cube.glb` conflicts with the built-in `cube`

Current built-in reserved names include:

- `cube`
- `tile`
- `pillar`
- `plane`
- `sphere`
- `cylinder`

When a conflict happens, rename the custom model file and try again.

## Scene Object Config vs Model Sidecar

These two layers serve different purposes.

Use the model sidecar for template-level defaults:

- model display name
- default scale
- center offset
- two-sided rendering
- material strategy

Use the scene object config for instance-level behavior:

- `object_id`
- `shape`
- `init_pos`
- `init_hpr`
- instance `scale`
- `color`
- `interactable`

Do not treat the sidecar JSON as a replacement for scene object configuration.

## Common Issues

### The model file is in the folder, but it does not appear

Check:

- the file is inside `assets/custom_models`
- the file extension is supported
- the `shape` value matches the file name without extension
- there is no name conflict with another custom model or built-in shape

### The model is too large or too small

Adjust either:

- the scene object `scale`
- or the sidecar `default_scale`

Use sidecar `default_scale` for template-level defaults. Use scene object `scale` for per-instance tuning.

### The model pivot or visual center looks wrong

Add `center_offset` to the sidecar JSON.

### The model is visible only from one side

Set `two_sided` to `true` in the sidecar JSON.

### The model does not keep its original material look

Make sure `use_builtin_materials` is `false` in the sidecar JSON, or leave it unspecified and use the default imported-model behavior.