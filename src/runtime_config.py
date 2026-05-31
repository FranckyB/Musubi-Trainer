import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .app_settings import (
    BACKENDS_ROOT_KEY,
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    MODEL_PATHS_KEY,
    MUSUBI_DIR_KEY,
    MUSUBI_LTX_DIR_KEY,
    MUSUBI_MAIN_DIR_KEY,
)

DEFAULT_MODEL_VERSION = "klein-base-9b"
_LTX_MODELS = {"ltx-2.3"}


@dataclass(frozen=True)
class RuntimeConfig:
    musubi_dir: Path
    musubi_python: Path | None
    training_dir: Path
    model_version: str
    dit: Path | None
    vae: Path | None
    text_encoder: Path | None


def _default_backends_root(workspace_dir: Path) -> Path:
    return workspace_dir / "Backends"


def _default_backend_path(workspace_dir: Path, folder_name: str) -> Path:
    return _default_backends_root(workspace_dir) / folder_name


def _resolve_musubi_backend_path(settings: dict[str, str], model_name: str) -> Path | None:
    workspace_dir = Path(__file__).resolve().parent.parent
    is_ltx_model = (model_name or "").strip().lower() in _LTX_MODELS

    backends_root_raw = settings.get(BACKENDS_ROOT_KEY, "").strip()
    backends_root = Path(backends_root_raw).expanduser() if backends_root_raw else _default_backends_root(workspace_dir)

    legacy_musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    legacy_musubi_path = Path(legacy_musubi_raw).expanduser() if legacy_musubi_raw else None

    main_raw = settings.get(MUSUBI_MAIN_DIR_KEY, "").strip()
    ltx_raw = settings.get(MUSUBI_LTX_DIR_KEY, "").strip()

    main_path = Path(main_raw).expanduser() if main_raw else (backends_root / "musubi-main")
    ltx_path = Path(ltx_raw).expanduser() if ltx_raw else (backends_root / "musubi-ltx")

    if is_ltx_model:
        return ltx_path
    if main_raw:
        return main_path
    if legacy_musubi_path is not None:
        return legacy_musubi_path
    return main_path


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
    model_version = settings.get(KLEIN_MODEL_VERSION_KEY, "").strip() or DEFAULT_MODEL_VERSION
    musubi_dir = _resolve_musubi_backend_path(settings, model_version)
    if musubi_dir is None:
        return None

    musubi_python = resolve_musubi_python(musubi_dir)
    workspace_dir = Path(__file__).resolve().parent.parent
    # Jobs are now the canonical training root (job metadata + per-job training dirs).
    training_dir = workspace_dir / "Jobs"

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
    musubi_dir = _resolve_musubi_backend_path(settings, model_name)
    if musubi_dir is None:
        return None

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
