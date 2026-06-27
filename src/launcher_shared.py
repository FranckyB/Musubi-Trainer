from __future__ import annotations

import re
from pathlib import Path

from . import app_settings

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VALID_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
}
LATENT_SUFFIX = "f2k9b"
DATASET_ORDER_KEY = "dataset_order"
DRAG_START_THRESHOLD_PX = 20
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
TRAIN_DIM_ALPHA_CHOICES = ("16", "32", "64", "128")
RESOLUTION_CHOICES = (256, 512, 768, 1024, 1280)
OPTIMIZER_TYPE_CHOICES = (
    "adamw8bit",
    "prodigy",
    "adamw",
    "adafactor",
    "pagedadamw8bit",
    "pagedadam8bit",
    "came8bit",
)
DEFAULT_PRODIGY_OPTIMIZER_ARGS = (
    "safeguard_warmup=True use_bias_correction=True weight_decay=0.01 betas=(0.9,0.99)"
)
JOB_SETTINGS_FILE_NAME = "settings.json"
JOBS_ORDER_FILE_NAME = "_order.json"
JOB_PRESET_FILE_SUFFIX = ".preset.json"
JOB_PROGRESS_FILE_NAME = "progress.json"
INVALID_FOLDER_CHARS = set('<>:"/\\|?*')


def get_positive_int_setting(settings: dict[str, str], key: str, fallback: int, minimum: int = 1) -> int:
    value = app_settings.parse_int_setting(settings, key)
    if value is None or value < minimum:
        return fallback
    return value


def get_non_negative_int_setting(settings: dict[str, str], key: str, fallback: int) -> int:
    value = app_settings.parse_int_setting(settings, key)
    if value is None or value < 0:
        return fallback
    return value


def get_train_log_backend_setting(settings: dict[str, str]) -> str:
    _value = settings.get(app_settings.TRAIN_LOG_BACKEND_KEY, "").strip().lower()
    return "tensorboard"


def is_truthy(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def is_valid_folder_name(name: str) -> bool:
    candidate = name.strip()
    if not candidate or candidate in {".", ".."}:
        return False
    if candidate[-1] in {" ", "."}:
        return False
    if any(ord(ch) < 32 for ch in candidate):
        return False
    if any(ch in INVALID_FOLDER_CHARS for ch in candidate):
        return False
    base = candidate.split(".")[0].strip().upper()
    return True


def load_dataset_order(settings: dict[str, str]) -> list[str]:
    raw = settings.get(DATASET_ORDER_KEY, "").strip()
    if not raw:
        return []
    return [name for name in raw.split("|") if name]


def save_dataset_order(settings: dict[str, str], dataset_order: list[str]) -> None:
    settings[DATASET_ORDER_KEY] = "|".join(dataset_order)


def latest_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    if not output_dir.exists():
        return None, 0

    pattern = re.compile(rf"^{re.escape(dataset_name)}_Klein-step(\d+)\.safetensors$", re.IGNORECASE)
    latest_step = 0
    latest_path: Path | None = None

    for checkpoint_path in output_dir.glob("*.safetensors"):
        match = pattern.match(checkpoint_path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = checkpoint_path

    return latest_path, latest_step


def scan_training_folders(training_dir: Path) -> list[str]:
    if not training_dir.exists():
        return []
    return sorted([path.name for path in training_dir.iterdir() if path.is_dir()])


def dataset_image_files(training_dir: Path, dataset_name: str) -> list[Path]:
    dataset_dir = training_dir / dataset_name
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []

    return sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])


def dataset_audio_files(training_dir: Path, dataset_name: str) -> list[Path]:
    dataset_dir = training_dir / dataset_name
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []

    return sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_AUDIO_EXTENSIONS])


def is_step1_ready(training_dir: Path, dataset_name: str) -> bool:
    dataset_dir = training_dir / dataset_name
    dataset_toml_exists = (dataset_dir / "dataset.toml").exists()
    image_files = dataset_image_files(training_dir, dataset_name)

    if not dataset_toml_exists or not image_files:
        return False

    return all(image_path.with_suffix(".txt").exists() for image_path in image_files)


def is_step2_ready(training_dir: Path, dataset_name: str) -> bool:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        pattern = f"{image_path.stem}_*_{LATENT_SUFFIX}.safetensors"
        if not any(cache_dir.glob(pattern)):
            return False

    return True


def is_step3_ready(training_dir: Path, dataset_name: str) -> bool:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        expected = cache_dir / f"{image_path.stem}_{LATENT_SUFFIX}_te.safetensors"
        if not expected.exists():
            return False

    return True


def dataset_status(training_dir: Path, dataset_name: str) -> dict[str, bool]:
    step1 = is_step1_ready(training_dir, dataset_name)
    step2 = is_step2_ready(training_dir, dataset_name)
    step3 = is_step3_ready(training_dir, dataset_name)
    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "ready_to_train": step1 and step2 and step3,
    }
