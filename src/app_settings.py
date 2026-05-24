import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"
WINDOW_X_KEY = "window_x"
WINDOW_Y_KEY = "window_y"
WINDOW_WIDTH_KEY = "window_width"
WINDOW_HEIGHT_KEY = "window_height"
MUSUBI_DIR_KEY = "musubi_dir"

KLEIN_MODEL_VERSION_KEY = "klein_model_version"
KLEIN_DIT_KEY = "klein_dit"
KLEIN_VAE_KEY = "klein_vae"
KLEIN_TEXT_ENCODER_KEY = "klein_text_encoder"

LTX_MODEL_VERSION_KEY = "ltx_model_version"
LTX_DIT_KEY = "ltx_dit"
LTX_VAE_KEY = "ltx_vae"
LTX_TEXT_ENCODER_KEY = "ltx_text_encoder"


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
