#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Missing venv/bin/python. Run ./Setup.sh first." >&2
    exit 1
fi

exec "$PY" -m src.app "$@"