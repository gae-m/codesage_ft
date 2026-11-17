#!/usr/bin/env bash
set -euo pipefail

if ! command -v 7z &> /dev/null; then
    echo "Error: 7z is not installed or not in PATH." >&2
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <directory_path> [output_archive.7z]" >&2
    exit 1
fi

if [[ ! -d "$1" ]]; then
    echo "Error: '$1' is not a valid directory." >&2
    exit 1
fi

# Resolve absolute path (handles trailing slashes and '.' cleanly)
TARGET_DIR="$(cd "$1" && pwd)"
DIR_NAME="$(basename "$TARGET_DIR")"

# Use second argument as output filename, or default to <directory_name>.7z
OUTPUT_ARCHIVE="${2:-${DIR_NAME}.7z}"

7z a -t7z -m0=lzma2 -mx=9 -mfb=64 -md=512m -ms=on -mmt=on "$OUTPUT_ARCHIVE" "$TARGET_DIR"