from dataclasses import dataclass
from pathlib import Path

from .app_settings import (
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    MUSUBI_DIR_KEY,
)

DEFAULT_KLEIN_MODEL_VERSION = "klein-base-9b"


@dataclass(frozen=True)
class RuntimeConfig:
    musubi_dir: Path
    training_dir: Path
    model_version: str
    dit: Path | None
    vae: Path | None
    text_encoder: Path | None


def runtime_config_from_settings(settings: dict[str, str]) -> RuntimeConfig | None:
    musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    if not musubi_raw:
        return None

    musubi_dir = Path(musubi_raw).expanduser()
    workspace_dir = Path(__file__).resolve().parent.parent
    training_dir = workspace_dir / "Training"

    klein_model_version = settings.get(KLEIN_MODEL_VERSION_KEY, "").strip() or DEFAULT_KLEIN_MODEL_VERSION

    dit_raw = settings.get(KLEIN_DIT_KEY, "").strip()
    vae_raw = settings.get(KLEIN_VAE_KEY, "").strip()
    text_encoder_raw = settings.get(KLEIN_TEXT_ENCODER_KEY, "").strip()

    dit_path = Path(dit_raw).expanduser() if dit_raw else None
    vae_path = Path(vae_raw).expanduser() if vae_raw else None
    text_encoder_path = Path(text_encoder_raw).expanduser() if text_encoder_raw else None

    return RuntimeConfig(
        musubi_dir=musubi_dir,
        training_dir=training_dir,
        model_version=klein_model_version,
        dit=dit_path,
        vae=vae_path,
        text_encoder=text_encoder_path,
    )
