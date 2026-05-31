import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .app_settings import (
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    MODEL_PATHS_KEY,
    MUSUBI_DIR_KEY,
)

DEFAULT_MODEL_VERSION = "klein-base-9b"


@dataclass(frozen=True)
class RuntimeConfig:
    musubi_dir: Path
    musubi_python: Path | None
    training_dir: Path
    model_version: str
    dit: Path | None
    vae: Path | None
    text_encoder: Path | None


def resolve_musubi_python(musubi_dir: Path) -> Path | None:
    _ = musubi_dir  # kept for call-site compatibility
    workspace_dir = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        candidates = [
            workspace_dir / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            workspace_dir / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def runtime_config_from_settings(settings: dict[str, str]) -> RuntimeConfig | None:
    musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    if not musubi_raw:
        return None

    musubi_dir = Path(musubi_raw).expanduser()
    musubi_python = resolve_musubi_python(musubi_dir)
    workspace_dir = Path(__file__).resolve().parent.parent
    # Jobs are now the canonical training root (job metadata + per-job training dirs).
    training_dir = workspace_dir / "Jobs"

    model_version = settings.get(KLEIN_MODEL_VERSION_KEY, "").strip() or DEFAULT_MODEL_VERSION

    dit_raw = settings.get(KLEIN_DIT_KEY, "").strip()
    vae_raw = settings.get(KLEIN_VAE_KEY, "").strip()
    text_encoder_raw = settings.get(KLEIN_TEXT_ENCODER_KEY, "").strip()

    dit_path = Path(dit_raw).expanduser() if dit_raw else None
    vae_path = Path(vae_raw).expanduser() if vae_raw else None
    text_encoder_path = Path(text_encoder_raw).expanduser() if text_encoder_raw else None

    return RuntimeConfig(
        musubi_dir=musubi_dir,
        musubi_python=musubi_python,
        training_dir=training_dir,
        model_version=model_version,
        dit=dit_path,
        vae=vae_path,
        text_encoder=text_encoder_path,
    )


def runtime_config_for_model(settings: dict[str, str], model_name: str) -> "RuntimeConfig | None":
    """Build a RuntimeConfig populated with paths for a specific model from MODEL_PATHS_KEY."""
    musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    if not musubi_raw:
        return None
    musubi_dir = Path(musubi_raw).expanduser()
    musubi_python = resolve_musubi_python(musubi_dir)
    workspace_dir = Path(__file__).resolve().parent.parent
    training_dir = workspace_dir / "Jobs"

    try:
        model_paths: dict[str, dict[str, str]] = json.loads(settings.get(MODEL_PATHS_KEY, "{}") or "{}")
    except Exception:
        model_paths = {}

    paths = model_paths.get(model_name, {})
    dit_raw = paths.get("dit", "").strip()
    vae_raw = paths.get("vae", "").strip()
    te_raw = paths.get("text_encoder", "").strip()

    return RuntimeConfig(
        musubi_dir=musubi_dir,
        musubi_python=musubi_python,
        training_dir=training_dir,
        model_version=model_name,
        dit=Path(dit_raw).expanduser() if dit_raw else None,
        vae=Path(vae_raw).expanduser() if vae_raw else None,
        text_encoder=Path(te_raw).expanduser() if te_raw else None,
    )
