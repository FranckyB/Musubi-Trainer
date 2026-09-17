#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Unified setup for a single shared venv used by:
# - Musubi-Trainer (this repo)
#
# Usage examples:
#   ./Setup.sh
#   ./Setup.sh --cuda cu128
#   ./Setup.sh --sage-wheel /path/to/sageattention.whl
# If --cuda is omitted, the script asks interactively when stdin is a TTY.
# Otherwise it defaults to cu130.
# -----------------------------------------------------------------------------

CUDA_TAG="cu130"
CUDA_ARG_PROVIDED=0
SAGE_WHEEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --cuda" >&2
                exit 1
            fi
            CUDA_TAG="$2"
            CUDA_ARG_PROVIDED=1
            shift 2
            ;;
        --sage-wheel)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --sage-wheel" >&2
                exit 1
            fi
            SAGE_WHEEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ $CUDA_ARG_PROVIDED -eq 0 ]]; then
    if [[ -t 0 ]]; then
        echo
        echo "Select CUDA wheel profile for PyTorch:"
        echo "  [1] cu130 (recommended)"
        echo "  [2] cu128"
        echo "  [3] cu124"
        read -r -p "Choose 1/2/3 (Enter for 1): " CUDA_CHOICE
        CUDA_CHOICE="${CUDA_CHOICE:-1}"

        case "$CUDA_CHOICE" in
            1) CUDA_TAG="cu130" ;;
            2) CUDA_TAG="cu128" ;;
            3) CUDA_TAG="cu124" ;;
            *)
                echo "Invalid choice '$CUDA_CHOICE'. Defaulting to recommended cu130."
                CUDA_TAG="cu130"
                ;;
        esac
    fi
fi

case "$CUDA_TAG" in
    cu124|cu128|cu130)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/$CUDA_TAG"
        ;;
    *)
        echo "Unsupported --cuda value '$CUDA_TAG'. Use cu124, cu128, or cu130." >&2
        exit 1
        ;;
esac

torch_stack_matches() {
    "$PY" -c "import importlib.util, sys; modules=('torch', 'torchvision', 'torchaudio'); sys.exit(1 if any(importlib.util.find_spec(name) is None for name in modules) else 0)" >/dev/null 2>&1 || return 1

    "$PY" -c "import sys; tag='+' + '$CUDA_TAG'; import torch, torchvision, torchaudio; sys.exit(0 if ((tag in getattr(torch, '__version__', '')) and (tag in getattr(torchvision, '__version__', '')) and (tag in getattr(torchaudio, '__version__', ''))) else 1)" >/dev/null 2>&1
}

check_gui_runtime() {
    if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
        echo "Tkinter is not available in python3.12 on this system." >&2
        echo "Install the system Tk runtime and rerun Setup.sh." >&2
        echo "On CachyOS/Arch this is usually: sudo pacman -S tk" >&2
        return 1
    fi
}

find_python() {
    if command -v python3.12 >/dev/null 2>&1; then
        echo "python3.12"
        return 0
    fi

    return 1
}

if ! BASE_PYTHON="$(find_python)"; then
    echo "python3.12 is required. Install python3.12 and rerun Setup.sh." >&2
    exit 1
fi

"$BASE_PYTHON" -m venv venv

PY="venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Failed to create venv." >&2
    exit 1
fi

"$PY" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)"
"$PY" -c "import struct; raise SystemExit(0 if struct.calcsize('P')*8==64 else 1)"

check_gui_runtime

PY_VERSION="$($PY -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')"
echo "Using Python $PY_VERSION from $BASE_PYTHON"

REQ_FILE="$(mktemp)"
cleanup() {
    rm -f "$REQ_FILE"
}
trap cleanup EXIT

SKIPPED_REQUIREMENTS=()
while IFS= read -r requirement; do
    case "$requirement" in
        triton-windows==*)
            SKIPPED_REQUIREMENTS+=("$requirement (Windows-only)")
            continue
            ;;
        tensorflow==*|tensorflow\>=*)
            SKIPPED_REQUIREMENTS+=("$requirement (optional on this Linux setup; skipped to avoid interpreter-specific wheel issues)")
            continue
            ;;
        deepfilternet==*)
            SKIPPED_REQUIREMENTS+=("$requirement (optional; pulls Rust build dependency and is currently incompatible with the torchaudio version used here)")
            continue
            ;;
    esac
    printf '%s\n' "$requirement" >> "$REQ_FILE"
done < requirements.txt

echo
echo "[1/5] Upgrading pip/wheel and installing compatible setuptools (<82 for torch cu130)..."
"$PY" -m pip install --upgrade pip wheel "setuptools>=70,<82"

echo
echo "[2/5] Installing Torch stack for $CUDA_TAG..."
SKIP_TORCH_INSTALL=0
if torch_stack_matches; then
    echo "  Torch stack already matches $CUDA_TAG. Skipping reinstall."
    SKIP_TORCH_INSTALL=1
fi
if [[ $SKIP_TORCH_INSTALL -eq 0 ]]; then
    "$PY" -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
fi

echo
echo "[3/5] Installing unified Python dependencies..."
if [[ ${#SKIPPED_REQUIREMENTS[@]} -gt 0 ]]; then
    echo "  Skipping optional or unsupported requirements on Linux:"
    for skipped_requirement in "${SKIPPED_REQUIREMENTS[@]}"; do
        echo "    - $skipped_requirement"
    done
fi
"$PY" -m pip install -r "$REQ_FILE"

echo
echo "[4/5] Installing SageAttention (optional)..."
if [[ -n "$SAGE_WHEEL" ]]; then
    if [[ -f "$SAGE_WHEEL" ]]; then
        echo "  Using explicit wheel: $SAGE_WHEEL"
        "$PY" -m pip install "$SAGE_WHEEL"
    else
        echo "  Provided --sage-wheel path not found: $SAGE_WHEEL" >&2
        exit 1
    fi
else
    echo "  No SageAttention wheel provided. Trying pip package for Linux..."
    if "$PY" -m pip install sageattention; then
        echo "  Installed SageAttention from pip."
    else
        echo "  Could not install SageAttention from pip. Continuing without it."
        echo "  Tip: rerun with --sage-wheel /path/to/sageattention.whl if you have a compatible local build."
    fi
fi

echo
echo "[5/5] Summary"
echo "  Unified venv python: $SCRIPT_DIR/venv/bin/python"
echo "  CUDA profile: $CUDA_TAG"
echo
echo "Setup complete."
echo "Musubi-Trainer can use this venv at:"
echo "  $SCRIPT_DIR/venv/bin/python"