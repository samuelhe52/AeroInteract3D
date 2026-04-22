#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="$SCRIPT_DIR/pandoc-report.tex"

DEFAULT_INPUT="$SCRIPT_DIR/final-tech-report.md"
INPUT_PATH="${1:-$DEFAULT_INPUT}"
if [[ ! "$INPUT_PATH" = /* ]]; then
  INPUT_PATH="$PWD/$INPUT_PATH"
fi

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input markdown not found: $INPUT_PATH" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required but was not found in PATH" >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is required but was not found in PATH" >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex is required but was not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Pandoc template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

INPUT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
DEFAULT_OUTPUT_DIR="$INPUT_DIR/.build"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
if [[ ! "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_DIR="$PWD/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

if [[ -d "$INPUT_DIR/assets" ]]; then
  rm -rf "$OUTPUT_DIR/assets"
  cp -R "$INPUT_DIR/assets" "$OUTPUT_DIR/assets"
fi

INPUT_FILENAME="$(basename "$INPUT_PATH")"
INPUT_BASENAME="${INPUT_FILENAME%.md}"
OUTPUT_PATH="$OUTPUT_DIR/$INPUT_BASENAME.pdf"
OUTPUT_TEX="$OUTPUT_DIR/$INPUT_BASENAME.tex"

(
  cd "$INPUT_DIR"
  pandoc "$INPUT_FILENAME" \
    --standalone \
    --from markdown+yaml_metadata_block+raw_tex+tex_math_dollars \
    --to latex \
    --template "$TEMPLATE_PATH" \
    --natbib \
    --output "$OUTPUT_TEX"
)

(
  cd "$INPUT_DIR"
  TEXINPUTS="$SCRIPT_DIR:${TEXINPUTS:-}" \
  BIBINPUTS="$SCRIPT_DIR:${BIBINPUTS:-}" \
  BSTINPUTS="$SCRIPT_DIR:${BSTINPUTS:-}" \
  latexmk -xelatex -interaction=nonstopmode -outdir="$OUTPUT_DIR" "$OUTPUT_TEX"
)

echo "Built PDF: $OUTPUT_PATH"
