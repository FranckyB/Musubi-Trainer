"""CLI wrapper for download_models.download_component.

Usage:
    python download_cli.py --model <name> --component <comp>
                           [--ws-root <path>] [--location <loc>]
                           [--token <token>]

Prints progress lines to stdout (flush=True on each line).
Prints RESULT:<absolute_path> on success.
Exits 1 with error printed to stderr on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly: python src/download_cli.py or python -m src.download_cli
_src_dir = Path(__file__).parent
if str(_src_dir.parent) not in sys.path:
    sys.path.insert(0, str(_src_dir.parent))

from src.download_models import (  # noqa: E402
    DOWNLOAD_LOCATION_MODELS_FOLDER,
    download_component,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a single model component.")
    parser.add_argument("--model", required=True, help="Model name (e.g. klein-9b)")
    parser.add_argument("--component", required=True, help="Component key (e.g. dit, vae, text_encoder)")
    parser.add_argument("--ws-root", default=None, help="Workspace root path (for Models folder location)")
    parser.add_argument("--location", default=DOWNLOAD_LOCATION_MODELS_FOLDER, help="Download location type")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    ws_root = Path(args.ws_root) if args.ws_root else None

    def progress(msg: str) -> None:
        print(msg, flush=True)

    try:
        path = download_component(
            model_name=args.model,
            component=args.component,
            location=args.location,
            ws_root=ws_root,
            progress=progress,
            token=args.token or None,
        )
        print(f"RESULT:{path}", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
