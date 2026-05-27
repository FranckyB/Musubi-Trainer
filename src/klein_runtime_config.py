import sys
from dataclasses import dataclass
from pathlib import Path

from .app_settings import (
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    MUSUBI_DIR_KEY,
    MUSUBI_PYTHON_KEY,
)

DEFAULT_KLEIN_MODEL_VERSION = "klein-base-9b"


@dataclass(frozen=True)
class KleinRuntimeConfig:
    musubi_dir: Path
    musubi_python: Path | None
    training_dir: Path
    model_version: str
    dit: Path | None
    vae: Path | None
    text_encoder: Path | None


def resolve_musubi_python(musubi_dir: Path) -> Path | None:
    if sys.platform == "win32":
        candidates = [
            musubi_dir / ".venv" / "Scripts" / "python.exe",
            musubi_dir / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            musubi_dir / ".venv" / "bin" / "python",
            musubi_dir / "venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def klein_runtime_config_from_settings(settings: dict[str, str]) -> KleinRuntimeConfig | None:
    musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    if not musubi_raw:
        return None

    musubi_dir = Path(musubi_raw).expanduser()
    musubi_python_raw = settings.get(MUSUBI_PYTHON_KEY, "").strip()
    musubi_python_override = Path(musubi_python_raw).expanduser() if musubi_python_raw else None
    musubi_python = (
        musubi_python_override
        if musubi_python_override is not None and musubi_python_override.exists() and musubi_python_override.is_file()
        else resolve_musubi_python(musubi_dir)
    )
    workspace_dir = Path(__file__).resolve().parent.parent
    # Jobs are now the canonical training root (job metadata + per-job training dirs).
    training_dir = workspace_dir / "Jobs"

    klein_model_version = settings.get(KLEIN_MODEL_VERSION_KEY, "").strip() or DEFAULT_KLEIN_MODEL_VERSION

    dit_raw = settings.get(KLEIN_DIT_KEY, "").strip()
    vae_raw = settings.get(KLEIN_VAE_KEY, "").strip()
    text_encoder_raw = settings.get(KLEIN_TEXT_ENCODER_KEY, "").strip()

    dit_path = Path(dit_raw).expanduser() if dit_raw else None
    vae_path = Path(vae_raw).expanduser() if vae_raw else None
    text_encoder_path = Path(text_encoder_raw).expanduser() if text_encoder_raw else None

    return KleinRuntimeConfig(
        musubi_dir=musubi_dir,
        musubi_python=musubi_python,
        training_dir=training_dir,
        model_version=klein_model_version,
        dit=dit_path,
        vae=vae_path,
        text_encoder=text_encoder_path,
    )


# Backward-compatible aliases while code migrates.
RuntimeConfig = KleinRuntimeConfig
runtime_config_from_settings = klein_runtime_config_from_settings
