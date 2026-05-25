import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
WINDOW_X_KEY = "window_x"
WINDOW_Y_KEY = "window_y"
WINDOW_WIDTH_KEY = "window_width"
WINDOW_HEIGHT_KEY = "window_height"
MUSUBI_DIR_KEY = "musubi_dir"
MUSUBI_PYTHON_KEY = "musubi_python"

KLEIN_MODEL_VERSION_KEY = "klein_model_version"
KLEIN_DIT_KEY = "klein_dit"
KLEIN_VAE_KEY = "klein_vae"
KLEIN_TEXT_ENCODER_KEY = "klein_text_encoder"

LTX_MODEL_VERSION_KEY = "ltx_model_version"
LTX_DIT_KEY = "ltx_dit"
LTX_VAE_KEY = "ltx_vae"
LTX_TEXT_ENCODER_KEY = "ltx_text_encoder"
DEFAULT_CAPTION_KEYWORD_KEY = "default_caption_keyword"
ENABLE_COMPILE_OPTIMIZATIONS_KEY = "enable_compile_optimizations"
ENABLE_COMPILE_CACHE_SIZE_LIMIT_KEY = "enable_compile_cache_size_limit"
ENABLE_CUDA_ALLOW_TF32_KEY = "enable_cuda_allow_tf32"
ENABLE_CUDA_CUDNN_BENCHMARK_KEY = "enable_cuda_cudnn_benchmark"
ENABLE_FP8_DIT_KEY = "enable_fp8_dit"
ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY = "enable_gradient_checkpointing_cpu_offload"
TRAIN_RESOLUTION_KEY = "train_resolution"
TRAIN_NETWORK_DIM_KEY = "train_network_dim"
TRAIN_NETWORK_ALPHA_KEY = "train_network_alpha"
TRAIN_LEARNING_RATE_KEY = "train_learning_rate"
TRAIN_STEPS_KEY = "train_steps"
TRAIN_ENABLE_LOGGING_KEY = "train_enable_logging"
TRAIN_LOG_BACKEND_KEY = "train_log_backend"
TRAIN_LOG_TRACKER_NAME_KEY = "train_log_tracker_name"
TRAIN_SAVE_CHECKPOINT_METADATA_KEY = "train_save_checkpoint_metadata"
TRAIN_AUTO_CLEANUP_STATES_KEY = "train_auto_cleanup_states"


def load_settings() -> dict[str, str]:
    if not SETTINGS_FILE.exists():
        return {}

    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    return {str(k): str(v) for k, v in raw.items()}


def save_settings(settings: dict[str, str]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        # Keep UI running even if settings file is temporarily locked/read-only.
        return


def parse_int_setting(settings: dict[str, str], key: str) -> int | None:
    raw = settings.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def load_window_position(settings: dict[str, str]) -> tuple[int, int] | None:
    x = parse_int_setting(settings, WINDOW_X_KEY)
    y = parse_int_setting(settings, WINDOW_Y_KEY)
    if x is None or y is None:
        return None
    return x, y


def load_window_size(settings: dict[str, str]) -> tuple[int, int] | None:
    width = parse_int_setting(settings, WINDOW_WIDTH_KEY)
    height = parse_int_setting(settings, WINDOW_HEIGHT_KEY)
    if width is None or height is None:
        return None
    return width, height
