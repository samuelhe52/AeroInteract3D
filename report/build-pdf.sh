#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INPUT="$SCRIPT_DIR/final-tech-report.md"
DEFAULT_OUTPUT_DIR="$SCRIPT_DIR/.build"

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

if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex is required but was not found in PATH" >&2
  exit 1
fi

OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
if [[ ! "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_DIR="$PWD/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

INPUT_BASENAME="$(basename "$INPUT_PATH" .md)"
OUTPUT_PATH="$OUTPUT_DIR/$INPUT_BASENAME.pdf"
HEADER_PATH="$SCRIPT_DIR/pandoc-header.tex"

pandoc "$INPUT_PATH" \
  -o "$OUTPUT_PATH" \
  --pdf-engine xelatex \
  --pdf-engine-opt=-output-driver="xdvipdfmx -z 0" \
  -V documentclass:ctexart \
  -V papersize:a4 \
  -V lang:zh-CN \
  -V geometry:margin=2cm \
  -V fontsize:12pt \
  -V linestretch:1.5 \
  -V numbersections:true \
  -V colorlinks:true \
  -V linkcolor:blue \
  -V urlcolor:blue \
  -V mainfont="Source Han Serif SC" \
  -V CJKmainfont="Source Han Serif SC" \
  -V CJKmonofont="Source Han Serif SC" \
  -V monofont="JetBrains Mono" \
  -H "$HEADER_PATH"

echo "Built PDF: $OUTPUT_PATH"
