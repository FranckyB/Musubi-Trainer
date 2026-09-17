import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
WINDOW_X_KEY = "window_x"
WINDOW_Y_KEY = "window_y"
WINDOW_WIDTH_KEY = "window_width"
WINDOW_HEIGHT_KEY = "window_height"
SASH_POSITION_KEY = "sash_position"
MUSUBI_DIR_KEY = "musubi_dir"
MUSUBI_PYTHON_KEY = "musubi_python"
BACKENDS_ROOT_KEY = "backends_root"
TRAINERS_ROOT_KEY = "trainers_root"
MUSUBI_MAIN_DIR_KEY = "musubi_main_dir"
MUSUBI_LTX_DIR_KEY = "musubi_ltx_dir"
SD_SCRIPTS_DIR_KEY = "sd_scripts_dir"

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
TRAIN_ENABLE_LOGGING_KEY = "train_enable_logging"
TRAIN_LOG_BACKEND_KEY = "train_log_backend"
TRAIN_LOG_TRACKER_NAME_KEY = "train_log_tracker_name"
TRAIN_STREAM_TO_LOGGER_KEY = "train_stream_to_logger"
TRAIN_AUTO_START_TENSORBOARD_KEY = "train_auto_start_tensorboard"
TRAIN_AUTO_CLEANUP_STATES_KEY = "train_auto_cleanup_states"
TRAIN_SAVE_EVERY_N_STEPS_KEY = "train_save_every_n_steps"
MODEL_DOWNLOAD_LOCATION_KEY = "model_download_location"
HF_TOKEN_KEY = "hf_token"
MODEL_PATHS_KEY = "model_paths"
EXTRA_SEARCH_PATHS_KEY = "extra_search_paths"
PREFERRED_PRESETS_BY_FAMILY_KEY = "preferred_presets_by_family"

_WORKSPACE_PATH_KEYS = {
    BACKENDS_ROOT_KEY,
    TRAINERS_ROOT_KEY,
    MUSUBI_DIR_KEY,
    MUSUBI_PYTHON_KEY,
    MUSUBI_MAIN_DIR_KEY,
    MUSUBI_LTX_DIR_KEY,
    SD_SCRIPTS_DIR_KEY,
    KLEIN_DIT_KEY,
    KLEIN_VAE_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    LTX_DIT_KEY,
    LTX_VAE_KEY,
    LTX_TEXT_ENCODER_KEY,
}


def _looks_like_windows_absolute_path(raw: str) -> bool:
    raw = str(raw or "").strip()
    if len(raw) < 3:
        return False
    return raw[1] == ":" and raw[0].isalpha() and raw[2] in {"\\", "/"}


def _workspace_root() -> Path:
    return SETTINGS_FILE.resolve().parent.parent


def _split_path_parts(raw: str) -> list[str]:
    normalized = str(raw or "").replace("\\", "/").strip()
    if not normalized:
        return []
    return [part for part in normalized.split("/") if part and part != "."]


def _rebase_workspace_path(raw: str) -> Path | None:
    workspace_root = _workspace_root()
    workspace_name = workspace_root.name.casefold()
    parts = _split_path_parts(raw)
    lowered_parts = [part.casefold() for part in parts]
    try:
        workspace_index = lowered_parts.index(workspace_name)
    except ValueError:
        return None

    suffix = parts[workspace_index + 1 :]
    rebased = workspace_root
    for part in suffix:
        rebased /= part
    return rebased


def _load_path_value(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""

    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    if _looks_like_windows_absolute_path(value):
        rebased = _rebase_workspace_path(value)
        return str(rebased) if rebased is not None else value

    rebased = _rebase_workspace_path(value)
    if rebased is not None:
        return str(rebased)

    return str((_workspace_root() / expanded).resolve(strict=False))


def _store_path_value(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""

    if _looks_like_windows_absolute_path(value):
        rebased = _rebase_workspace_path(value)
        if rebased is not None:
            try:
                return rebased.relative_to(_workspace_root()).as_posix()
            except ValueError:
                return str(rebased)
        return value

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.relative_to(_workspace_root()).as_posix()
    except ValueError:
        return str(candidate)


def _normalize_loaded_settings(settings: dict[str, str]) -> tuple[dict[str, str], bool]:
    normalized = dict(settings)
    changed = False

    for key in _WORKSPACE_PATH_KEYS:
        value = normalized.get(key, "")
        loaded_value = _load_path_value(value)
        if loaded_value != value:
            normalized[key] = loaded_value
            changed = True

    raw_extra = normalized.get(EXTRA_SEARCH_PATHS_KEY, "").strip()
    if raw_extra:
        try:
            extra_paths = json.loads(raw_extra)
        except Exception:
            extra_paths = None
        if isinstance(extra_paths, list):
            loaded_extra = [_load_path_value(str(item)) for item in extra_paths]
            if loaded_extra != [str(item) for item in extra_paths]:
                normalized[EXTRA_SEARCH_PATHS_KEY] = json.dumps(loaded_extra)
                changed = True

    raw_model_paths = normalized.get(MODEL_PATHS_KEY, "").strip()
    if raw_model_paths:
        try:
            model_paths = json.loads(raw_model_paths)
        except Exception:
            model_paths = None
        if isinstance(model_paths, dict):
            loaded_model_paths: dict[str, dict[str, str]] = {}
            for model_name, component_map in model_paths.items():
                if not isinstance(component_map, dict):
                    continue
                loaded_model_paths[str(model_name)] = {
                    str(component): _load_path_value(str(path_value))
                    for component, path_value in component_map.items()
                    if str(path_value).strip()
                }
            if loaded_model_paths != model_paths:
                normalized[MODEL_PATHS_KEY] = json.dumps(loaded_model_paths)
                changed = True

    return normalized, changed


def _prepare_settings_for_disk(settings: dict[str, str]) -> dict[str, str]:
    prepared = dict(settings)

    for key in _WORKSPACE_PATH_KEYS:
        prepared[key] = _store_path_value(prepared.get(key, ""))

    raw_extra = prepared.get(EXTRA_SEARCH_PATHS_KEY, "").strip()
    if raw_extra:
        try:
            extra_paths = json.loads(raw_extra)
        except Exception:
            extra_paths = None
        if isinstance(extra_paths, list):
            prepared[EXTRA_SEARCH_PATHS_KEY] = json.dumps([_store_path_value(str(item)) for item in extra_paths])

    raw_model_paths = prepared.get(MODEL_PATHS_KEY, "").strip()
    if raw_model_paths:
        try:
            model_paths = json.loads(raw_model_paths)
        except Exception:
            model_paths = None
        if isinstance(model_paths, dict):
            stored_model_paths: dict[str, dict[str, str]] = {}
            for model_name, component_map in model_paths.items():
                if not isinstance(component_map, dict):
                    continue
                stored_model_paths[str(model_name)] = {
                    str(component): _store_path_value(str(path_value))
                    for component, path_value in component_map.items()
                    if str(path_value).strip()
                }
            prepared[MODEL_PATHS_KEY] = json.dumps(stored_model_paths)

    return prepared


def load_settings() -> dict[str, str]:
    if not SETTINGS_FILE.exists():
        return {}

    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    loaded = {str(k): str(v) for k, v in raw.items()}
    normalized, changed = _normalize_loaded_settings(loaded)
    if changed:
        save_settings(normalized)
    return normalized


def save_settings(settings: dict[str, str]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(_prepare_settings_for_disk(settings), indent=2), encoding="utf-8")
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
