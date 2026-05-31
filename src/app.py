import sys
import argparse
import os
import re
import json
import configparser
import ctypes
import socket
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import Callable

from .app_settings import (
    DEFAULT_CAPTION_KEYWORD_KEY,
    ENABLE_CUDA_ALLOW_TF32_KEY,
    ENABLE_CUDA_CUDNN_BENCHMARK_KEY,
    ENABLE_COMPILE_OPTIMIZATIONS_KEY,
    ENABLE_FP8_DIT_KEY,
    ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY,
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    LTX_DIT_KEY,
    LTX_MODEL_VERSION_KEY,
    LTX_TEXT_ENCODER_KEY,
    LTX_VAE_KEY,
    MUSUBI_DIR_KEY,
    MUSUBI_PYTHON_KEY,
    SETTINGS_FILE,
    TRAIN_LOG_BACKEND_KEY,
    TRAIN_LOG_TRACKER_NAME_KEY,
    TRAIN_STREAM_TO_LOGGER_KEY,
    TRAIN_AUTO_START_TENSORBOARD_KEY,
    TRAIN_AUTO_CLEANUP_STATES_KEY,
    TRAIN_SAVE_EVERY_N_STEPS_KEY,
    TRAIN_ENABLE_LOGGING_KEY,
    WINDOW_HEIGHT_KEY,
    WINDOW_WIDTH_KEY,
    WINDOW_X_KEY,
    WINDOW_Y_KEY,
    SASH_POSITION_KEY,
    MODEL_DOWNLOAD_LOCATION_KEY,
    HF_TOKEN_KEY,
    MODEL_PATHS_KEY,
    EXTRA_SEARCH_PATHS_KEY,
    PREFERRED_PRESETS_BY_FAMILY_KEY,
    load_settings,
    parse_int_setting,
    load_window_size,
    load_window_position,
    save_settings,
)
from .download_models import (
    MODELS as DOWNLOAD_MODELS,
    MODEL_FAMILIES as DOWNLOAD_MODEL_FAMILIES,
    MODEL_DISPLAY_NAMES as DOWNLOAD_MODEL_DISPLAY_NAMES,
    COMPONENT_FRIENDLY_NAMES as DOWNLOAD_COMPONENT_FRIENDLY_NAMES,
    DOWNLOAD_LOCATIONS,
    DOWNLOAD_LOCATION_MODELS_FOLDER,
    find_component,
    find_in_extra_paths,
    auto_resolve_klein,
    workspace_root as download_workspace_root,
)
from .runtime_config import RuntimeConfig, runtime_config_from_settings, runtime_config_for_model, resolve_musubi_python
from .train_utils import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_NETWORK_DIM,
    DEFAULT_RESOLUTION,
    DEFAULT_SAVE_EVERY_N_STEPS,
    DEFAULT_TRAIN_STEPS,
    JOB_EXIT_CANCELLED,
    JOB_EXIT_FAILED,
    JOB_EXIT_SUCCESS,
)
from .train_flux2 import run_job as _run_job_flux2, train_models
from .train_ltx import run_job as _run_job_ltx
from .train_wan import run_job as _run_job_wan
from .train_zimage import run_job as _run_job_zimage
from .train_qwen import run_job as _run_job_qwen

# Model name → run_job function
_KLEIN_MODELS = {"flux2-dev", "klein-base-9b", "klein-9b", "klein-base-4b", "klein-4b"}
_LTX_MODELS = {"ltx-2.3"}
_WAN_MODELS = {"wan2.1-t2v-14b", "wan2.1-i2v-720p-14b", "wan2.1-i2v-480p-14b", "wan2.2-t2v-14b", "wan2.2-i2v-720p-14b", "wan2.2-i2v-480p-14b"}
_ZIMAGE_MODELS = {"zimage-de-turbo"}
_QWEN_MODELS = {"qwen-image", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511", "qwen-image-layered"}


def _run_job_for_model(model_name: str):
    """Return the appropriate run_job function for a given model name."""
    if model_name in _KLEIN_MODELS:
        return _run_job_flux2
    if model_name in _LTX_MODELS:
        return _run_job_ltx
    if model_name in _WAN_MODELS:
        return _run_job_wan
    if model_name in _ZIMAGE_MODELS:
        return _run_job_zimage
    if model_name in _QWEN_MODELS:
        return _run_job_qwen
    return None


# Model files
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"
DATASET_ORDER_KEY = "dataset_order"
DRAG_START_THRESHOLD_PX = 20
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
TRAIN_DIM_ALPHA_CHOICES = ("16", "32", "64")
RESOLUTION_CHOICES = (512, 768, 1024, 1280)
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
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def get_positive_int_setting(settings: dict[str, str], key: str, fallback: int, minimum: int = 1) -> int:
    value = parse_int_setting(settings, key)
    if value is None or value < minimum:
        return fallback
    return value


def get_non_negative_int_setting(settings: dict[str, str], key: str, fallback: int) -> int:
    value = parse_int_setting(settings, key)
    if value is None or value < 0:
        return fallback
    return value


def get_train_log_backend_setting(settings: dict[str, str]) -> str:
    _value = settings.get(TRAIN_LOG_BACKEND_KEY, "").strip().lower()
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
    if base in WINDOWS_RESERVED_NAMES:
        return False
    return True


def load_ui_config(config_path: Path) -> dict[str, int]:
    defaults = {
        "window_width": 780,
        "window_height": 1000,
        "min_window_width": 780,
        "min_window_height": 1000,
        "card_gap": 8,
        "card_width": 172,
        "thumbnail_size": 152,
        "card_height": 212,
        "relayout_debounce_ms": 120,
    }

    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except Exception:
        return defaults

    if "ui" not in parser:
        return defaults

    ui = parser["ui"]
    resolved = defaults.copy()
    for key, fallback in defaults.items():
        try:
            resolved[key] = max(1, ui.getint(key, fallback=fallback))
        except (TypeError, ValueError):
            resolved[key] = fallback
    return resolved


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


def launch_ui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
        from tkinter import ttk
        from PIL import Image, ImageDraw, ImageTk
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
            tkdnd_available = True
        except Exception:
            DND_FILES = None  # type: ignore[assignment]
            TkinterDnD = None  # type: ignore[assignment]
            tkdnd_available = False
    except ImportError:
        print("Tkinter and Pillow are required for the visual launcher.")
        print("Use CLI mode with names and step flags, or install Pillow.")
        return 1

    bg_root = "#181818"
    bg_panel = "#242424"
    bg_card = "#2d2d2d"
    fg_text = "#e6e6e6"
    fg_muted = "#a9a9a9"
    border_dark = "#3a3a3a"
    color_green = "#35c46a"
    color_start_enabled = "#35c46a"
    color_start_enabled_active = "#4dd97f"
    color_start_disabled = "#3b3b3b"
    color_start_in_progress = "#ff8c00"
    workspace_dir = Path(__file__).resolve().parent.parent
    ui_config = load_ui_config(Path(__file__).resolve().parent / "app.config")
    default_models_dir = workspace_dir / "Models"
    app_user_model_id = "MusubiTrainer.Launcher"

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_user_model_id)
        except Exception:
            pass

    def set_dark_title_bar(window: tk.Misc) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = window.winfo_id()
            value = ctypes.c_int(1)
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Win10 1809+)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            return

    def center_window(window: tk.Misc) -> None:
        window.update_idletasks()
        width = max(window.winfo_width(), window.winfo_reqwidth())
        height = max(window.winfo_height(), window.winfo_reqheight())
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        pos_x = max(0, (screen_w - width) // 2)
        pos_y = max(0, (screen_h - height) // 2)
        window.geometry(f"+{pos_x}+{pos_y}")

    if tkdnd_available and TkinterDnD is not None:
        try:
            root = TkinterDnD.Tk()
        except Exception as exc:
            # Keep launcher usable even when tkdnd native DLL fails (common 32/64-bit mismatch).
            print(f"[UI] Drag-and-drop disabled: tkinterdnd2/tkdnd failed to load: {exc}")
            tkdnd_available = False

            # TkinterDnD may leave behind a partially-initialized default root window.
            # Clean it up before creating our fallback root to avoid a floating ghost "tk" window.
            try:
                ghost_root = getattr(tk, "_default_root", None)
                if ghost_root is not None:
                    ghost_root.destroy()
                tk._default_root = None  # type: ignore[attr-defined]
            except Exception:
                pass

            root = tk.Tk()
    else:
        root = tk.Tk()
    root.withdraw()
    root.title("Musubi Training Launcher")
    root.geometry(f"{ui_config['window_width']}x{ui_config['window_height']}")
    root.resizable(False, True)
    root.minsize(ui_config["min_window_width"], ui_config["min_window_height"])
    root.configure(bg=bg_root)
    ico_path = Path(__file__).resolve().parent / "icons" / "logo.ico"
    if sys.platform == "win32" and ico_path.exists():
        try:
            root.iconbitmap(default=str(ico_path))
        except Exception:
            pass
    set_dark_title_bar(root)

    settings_state = load_settings()
    dataset_order: list[str] = load_dataset_order(settings_state)
    window_position_applied = False
    settings_reset_requested = False

    def apply_initial_main_window_position() -> None:
        nonlocal window_position_applied
        root.update_idletasks()

        min_width = root.winfo_reqwidth()
        min_height = root.winfo_reqheight()

        saved_size = load_window_size(settings_state)
        if saved_size is None:
            width = max(root.winfo_width(), min_width)
            height = max(root.winfo_height(), min_height, ui_config["min_window_height"])
        else:
            saved_width, saved_height = saved_size
            width = max(min_width, saved_width)
            height = max(min_height, ui_config["min_window_height"], saved_height)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        width = min(width, screen_w)
        height = min(height, screen_h)

        saved_pos = load_window_position(settings_state)
        if saved_pos is None:
            target_x = max(0, (screen_w - width) // 2)
            target_y = max(0, (screen_h - height) // 2)
        else:
            target_x, target_y = saved_pos

        target_x = min(max(0, target_x), max(0, screen_w - width))
        target_y = min(max(0, target_y), max(0, screen_h - height))
        root.geometry(f"{width}x{height}+{target_x}+{target_y}")
        root.deiconify()
        window_position_applied = True

        def restore_sash() -> None:
            saved_sash = parse_int_setting(settings_state, SASH_POSITION_KEY)
            if saved_sash is not None and saved_sash > 0:
                try:
                    paned.sashpos(0, saved_sash)
                except Exception:
                    pass
            else:
                try:
                    paned.sashpos(0, int(paned.winfo_height() * 3 / 5))
                except Exception:
                    pass
        root.after(50, restore_sash)

    def save_main_window_position_now() -> None:
        nonlocal settings_state
        if settings_reset_requested:
            return
        settings_state[WINDOW_X_KEY] = str(root.winfo_x())
        settings_state[WINDOW_Y_KEY] = str(root.winfo_y())
        settings_state[WINDOW_WIDTH_KEY] = str(root.winfo_width())
        settings_state[WINDOW_HEIGHT_KEY] = str(root.winfo_height())
        try:
            settings_state[SASH_POSITION_KEY] = str(paned.sashpos(0))
        except Exception:
            pass
        save_settings(settings_state)

    _save_after_id: str | None = None

    def schedule_main_window_position_save(_event: tk.Event) -> None:
        nonlocal _save_after_id
        if not window_position_applied:
            return
        if root.state() != "normal":
            return
        if _save_after_id is not None:
            root.after_cancel(_save_after_id)
        _save_after_id = root.after(300, save_main_window_position_now)

    def on_root_close() -> None:
        if (not settings_reset_requested) and window_position_applied and root.winfo_exists():
            save_main_window_position_now()
        stop_tensorboard_started_by_app()
        root.destroy()

    root.bind("<Configure>", schedule_main_window_position_save)
    root.protocol("WM_DELETE_WINDOW", on_root_close)
    root.after(0, apply_initial_main_window_position)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=bg_panel, foreground=fg_text, font=("Segoe UI", 9))
    style.configure("TFrame", background=bg_panel)
    style.configure("TLabel", background=bg_panel, foreground=fg_text)
    style.configure(
        "TLabelframe",
        background=bg_panel,
        foreground=fg_text,
        bordercolor=border_dark,
        lightcolor=border_dark,
        darkcolor=border_dark,
    )
    style.configure("TLabelframe.Label", background=bg_panel, foreground=fg_text)
    style.configure(
        "TButton",
        background="#353535",
        foreground=fg_text,
        padding=(10, 2),
        bordercolor=border_dark,
        lightcolor=border_dark,
        darkcolor=border_dark,
        relief="flat",
    )
    style.map("TButton", background=[("active", "#404040")])
    style.configure(
        "TEntry",
        fieldbackground="#1f1f1f",
        foreground=fg_text,
        bordercolor=border_dark,
        lightcolor=border_dark,
        darkcolor=border_dark,
    )
    style.map(
        "TEntry",
        fieldbackground=[("readonly", "#1f1f1f"), ("disabled", "#2a2a2a")],
        foreground=[("disabled", fg_muted)],
    )
    style.configure(
        "Flat.TEntry",
        fieldbackground="#1f1f1f",
        foreground=fg_text,
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Flat.TEntry",
        fieldbackground=[("readonly", "#1f1f1f"), ("disabled", "#2a2a2a")],
        foreground=[("disabled", fg_muted)],
    )
    # Ensure insertion caret is visible on dark entry backgrounds.
    style.configure("TEntry", insertcolor=fg_text)
    style.configure("Flat.TEntry", insertcolor=fg_text)
    style.configure("PathDisplay.TLabel", background="#1f1f1f", foreground=fg_text)
    style.configure(
        "FamilyHeader.TButton",
        background="#2d2d2d",
        foreground=fg_text,
        font=("Segoe UI", 9, "bold"),
        anchor="w",
        padding=(8, 4),
    )
    style.map(
        "FamilyHeader.TButton",
        background=[("active", "#383838"), ("pressed", "#383838")],
    )
    style.configure(
        "TCombobox",
        fieldbackground="#1f1f1f",
        background="#1f1f1f",
        foreground=fg_text,
        arrowcolor=fg_text,
        bordercolor=border_dark,
        lightcolor=border_dark,
        darkcolor=border_dark,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#1f1f1f"), ("disabled", "#2a2a2a")],
        foreground=[("readonly", fg_text), ("disabled", fg_muted)],
        background=[("readonly", "#1f1f1f")],
        selectbackground=[("readonly", "#2f4f66")],
        selectforeground=[("readonly", "#ffffff")],
    )
    style.configure("TCombobox", insertcolor=fg_text)
    # Option database fallbacks help classic tk.Entry and themed entries on some platforms.
    root.option_add("*insertBackground", fg_text)
    root.option_add("*insertWidth", 2)
    # Style ttk.Combobox dropdown list to avoid OS-default bright colors.
    root.option_add("*TCombobox*Listbox.background", "#1f1f1f")
    root.option_add("*TCombobox*Listbox.foreground", fg_text)
    root.option_add("*TCombobox*Listbox.selectBackground", "#2f4f66")
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.highlightThickness", 0)
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    style.configure("TCheckbutton", background=bg_panel, foreground=fg_text)
    style.map("TCheckbutton", background=[("active", bg_panel)], foreground=[("disabled", fg_muted)])
    style.configure(
        "TNotebook",
        background=bg_panel,
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background="#1f1f1f",
        foreground=fg_muted,
        padding=(10, 5),
        borderwidth=0,
        focuscolor=bg_panel,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", bg_panel), ("active", "#2e2e2e")],
        foreground=[("selected", fg_text), ("active", fg_text)],
    )
    style.configure(
        "Card.TFrame",
        background=bg_card,
        relief="solid",
        borderwidth=1,
        bordercolor=border_dark,
        lightcolor=border_dark,
        darkcolor=border_dark,
    )
    style.configure(
        "ActiveCard.TFrame",
        background=bg_card,
        relief="solid",
        borderwidth=2,
        bordercolor="#5a5a5a",
        lightcolor="#5a5a5a",
        darkcolor="#5a5a5a",
    )
    style.configure(
        "SelectedCard.TFrame",
        background=bg_card,
        relief="solid",
        borderwidth=2,
        bordercolor="#5db6ff",
        lightcolor="#5db6ff",
        darkcolor="#5db6ff",
    )
    style.configure(
        "DoneCard.TFrame",
        background="#262626",
        relief="solid",
        borderwidth=1,
        bordercolor="#3e5a47",
        lightcolor="#3e5a47",
        darkcolor="#3e5a47",
    )
    style.configure(
        "DragSourceCard.TFrame",
        background=bg_card,
        relief="solid",
        borderwidth=2,
        bordercolor="#e3b45c",
        lightcolor="#e3b45c",
        darkcolor="#e3b45c",
    )
    style.configure(
        "DropTargetCard.TFrame",
        background=bg_card,
        relief="solid",
        borderwidth=2,
        bordercolor="#7ec4ff",
        lightcolor="#7ec4ff",
        darkcolor="#7ec4ff",
    )
    style.configure("CardTitle.TLabel", background=bg_card, foreground=fg_text)
    style.configure("CardMeta.TLabel", background=bg_card, foreground=fg_muted)
    style.configure("DoneCardTitle.TLabel", background="#262626", foreground=fg_text)
    style.configure("DoneCardMeta.TLabel", background="#262626", foreground=fg_muted)
    style.configure("Card.TCheckbutton", background=bg_card, foreground=fg_text)
    style.map("Card.TCheckbutton", background=[("active", bg_card)], foreground=[("disabled", fg_muted)])
    style.configure(
        "Dataset.Vertical.TScrollbar",
        background="#3a3a3a",
        troughcolor="#2a2a2a",
        bordercolor="#323232",
        lightcolor="#323232",
        darkcolor="#323232",
        arrowcolor="#8c8c8c",
        relief="flat",
        arrowsize=12,
        width=12,
    )
    style.map(
        "Dataset.Vertical.TScrollbar",
        background=[("active", "#454545")],
        arrowcolor=[("active", "#b0b0b0")],
    )
    style.configure(
        "Dark.Vertical.TScrollbar",
        background="#3a3a3a",
        troughcolor="#2a2a2a",
        bordercolor="#323232",
        lightcolor="#323232",
        darkcolor="#323232",
        arrowcolor="#8c8c8c",
        relief="flat",
        arrowsize=12,
        width=12,
    )
    style.map(
        "Dark.Vertical.TScrollbar",
        background=[("active", "#454545")],
        arrowcolor=[("active", "#b0b0b0")],
    )
    style.configure(
        "Dark.Horizontal.TScrollbar",
        background="#3a3a3a",
        troughcolor="#2a2a2a",
        bordercolor="#323232",
        lightcolor="#323232",
        darkcolor="#323232",
        arrowcolor="#8c8c8c",
        relief="flat",
        arrowsize=12,
        width=12,
    )
    style.map(
        "Dark.Horizontal.TScrollbar",
        background=[("active", "#454545")],
        arrowcolor=[("active", "#b0b0b0")],
    )
    style.configure(
        "Queue.Treeview",
        background="#141924",
        fieldbackground="#141924",
        foreground=fg_text,
        font=("Segoe UI", 10),
        borderwidth=0,
        relief="flat",
        rowheight=52,
    )
    style.map(
        "Queue.Treeview",
        background=[("selected", "#1e4a7a")],
    )
    style.configure(
        "Queue.Treeview.Heading",
        background="#1a2233",
        foreground="#8ba7cc",
        font=("Segoe UI", 8, "bold"),
        borderwidth=0,
        relief="flat",
        padding=(4, 6),
    )
    style.map(
        "Queue.Treeview.Heading",
        background=[("active", "#1e2a40")],
    )
    style.configure(
        "QueueAction.TButton",
        background="#1f1f1f",
        padding=(0, 0),
        borderwidth=0,
        relief="flat",
        focuscolor="#1f1f1f",
        font=("Segoe UI Emoji", 10),
    )
    style.map(
        "QueueAction.TButton",
        background=[("active", "#1f1f1f"), ("disabled", "#1f1f1f")],
    )
    style.configure(
        "StartDisabled.TButton",
        background=color_start_disabled,
        foreground="#c6c6c6",
        padding=(10, 4),
        borderwidth=1,
        bordercolor="#4a4a4a",
        lightcolor="#565656",
        darkcolor="#2f2f2f",
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartDisabled.TButton",
        background=[("active", color_start_disabled), ("disabled", color_start_disabled)],
        foreground=[("disabled", "#c6c6c6")],
    )
    style.configure(
        "StartEnabled.TButton",
        background=color_start_enabled,
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=1,
        bordercolor="#2ea95a",
        lightcolor="#63e394",
        darkcolor="#238149",
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartEnabled.TButton",
        background=[("active", color_start_enabled_active), ("disabled", color_start_disabled)],
        foreground=[("active", "#ffffff"), ("disabled", "#c6c6c6")],
    )
    style.configure(
        "StartInProgress.TButton",
        background=color_start_in_progress,
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=1,
        bordercolor="#cc7000",
        lightcolor="#ffb347",
        darkcolor="#a85b00",
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartInProgress.TButton",
        background=[("active", color_start_in_progress), ("disabled", color_start_in_progress)],
        foreground=[("active", "#ffffff"), ("disabled", "#ffffff")],
    )
    style.configure(
        "Sash",
        sashthickness=6,
        sashrelief="flat",
        background="#2e2e2e",
    )
    style.configure("TPanedwindow", background="#2e2e2e")

    vars_by_name: dict[str, tk.BooleanVar] = {}
    card_widgets: list[tk.Widget] = []
    thumbnail_cache: dict[tuple[str, str, int, int], ImageTk.PhotoImage] = {}
    first_image_cache: dict[str, Path | None] = {}
    checkpoint_cache: dict[str, tuple[Path | None, int]] = {}
    run_state_by_name: dict[str, str] = {}
    card_frame_by_name: dict[str, ttk.Frame] = {}
    card_thumb_by_name: dict[str, ImageTk.PhotoImage] = {}
    resize_after_id: str | None = None
    last_canvas_width = 0
    run_in_progress = False
    run_cancel_event: threading.Event | None = None
    drag_dataset_name: str | None = None
    drag_hover_dataset_name: str | None = None
    drag_preview: tk.Toplevel | None = None
    drag_preview_photo: ImageTk.PhotoImage | None = None
    drag_moved = False
    drag_start_x: int | None = None
    drag_start_y: int | None = None
    tensorboard_launch_in_progress = False
    tensorboard_started_by_app = False
    tensorboard_process: subprocess.Popen | None = None
    metrics_viewer_button: ttk.Button | None = None
    job_queue: list[dict[str, str]] = []
    queue_drag_index: int | None = None
    queue_drag_moved = False
    queue_drag_allowed = False
    queue_row_drag_handles: dict[str, tk.Label] = {}
    queue_row_action_buttons: dict[str, tk.Label] = {}
    queue_row_thumb_labels: dict[str, tk.Label] = {}
    queue_row_checkbox_labels: dict[str, tk.Label] = {}
    queue_row_dividers: dict[str, tk.Frame] = {}
    queue_col_dividers: list[tk.Frame] = []
    queue_thumb_by_item: dict[str, ImageTk.PhotoImage] = {}
    runtime_config = runtime_config_from_settings(settings_state)
    tensorboard_host = "127.0.0.1"
    tensorboard_port = 6006
    tensorboard_url = f"http://{tensorboard_host}:{tensorboard_port}"

    def is_tensorboard_running() -> bool:
        try:
            with socket.create_connection((tensorboard_host, tensorboard_port), timeout=0.3):
                return True
        except OSError:
            return False

    def python_has_module(python_path: Path, module_name: str) -> bool:
        if not python_path.is_file():
            return False
        try:
            result = subprocess.run(
                [str(python_path), "-c", f"import {module_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
        except Exception:
            return False
        return result.returncode == 0

    def resolve_tensorboard_python() -> Path | None:
        if runtime_config is None or runtime_config.musubi_python is None:
            return None

        musubi_python = runtime_config.musubi_python
        if python_has_module(musubi_python, "tensorboard"):
            return musubi_python

        return None

    def launch_tensorboard_background() -> bool:
        nonlocal tensorboard_started_by_app, tensorboard_process
        if is_tensorboard_running():
            return True
        if runtime_config is None:
            return False

        logs_root = runtime_config.training_dir
        try:
            logs_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[Metrics Viewer] Could not create logs directory: {exc}")
            return False

        python_path = resolve_tensorboard_python()
        if python_path is None:
            print(
                "[Metrics Viewer] TensorBoard is not installed in the app venv Python."
            )
            return False

        command = [
            str(python_path),
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logs_root),
            "--port",
            str(tensorboard_port),
            "--reload_interval",
            "5",
        ]

        popen_kwargs: dict[str, object] = {
            "cwd": str(logs_root),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )

        try:
            process = subprocess.Popen(command, **popen_kwargs)
            tensorboard_process = process
            tensorboard_started_by_app = True
        except OSError as exc:
            print(f"[Metrics Viewer] Could not start TensorBoard: {exc}")
            return False

        for _ in range(30):
            if is_tensorboard_running():
                return True
            if process.poll() is not None:
                print("[Metrics Viewer] TensorBoard process exited before becoming ready.")
                return False
            time.sleep(0.1)

        # Slow startup is common; if the process is still alive, treat this as launched.
        if process.poll() is None:
            print("[Metrics Viewer] TensorBoard is still starting in the background.")
            return True

        return False

    def stop_tensorboard_started_by_app() -> None:
        nonlocal tensorboard_started_by_app, tensorboard_process
        if not tensorboard_started_by_app:
            return
        if tensorboard_process is None:
            return

        if tensorboard_process.poll() is not None:
            tensorboard_started_by_app = False
            tensorboard_process = None
            return

        try:
            tensorboard_process.terminate()
            tensorboard_process.wait(timeout=3)
        except Exception:
            try:
                tensorboard_process.kill()
                tensorboard_process.wait(timeout=3)
            except Exception:
                pass
        finally:
            tensorboard_started_by_app = False
            tensorboard_process = None

    def maybe_autostart_tensorboard() -> None:
        if runtime_config is None:
            return
        if settings_state.get(TRAIN_AUTO_START_TENSORBOARD_KEY, "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return
        threading.Thread(target=launch_tensorboard_background, daemon=True).start()

    def is_valid_musubi_tuner_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return (
            (path / "pyproject.toml").exists()
            or (path / "src" / "musubi_tuner").is_dir()
            or (path / "flux_2_train_network.py").exists()
        )

    def apply_musubi_dir_setting(musubi_dir: Path) -> bool:
        nonlocal runtime_config, settings_state
        if not is_valid_musubi_tuner_dir(musubi_dir):
            return False
        settings_state[MUSUBI_DIR_KEY] = str(musubi_dir)
        settings_state[MUSUBI_PYTHON_KEY] = ""
        save_settings(settings_state)
        runtime_config = runtime_config_from_settings(settings_state)
        return runtime_config is not None

    def prompt_to_clone_or_select_musubi() -> bool:
        choice = messagebox.askyesnocancel(
            "Musubi-Tuner not found",
            "Musubi-Tuner folder is not configured or is invalid.\n\n"
            "Yes: Clone Musubi-Tuner now\n"
            "No: Choose it manually in Settings\n"
            "Cancel: Exit",
            parent=root,
        )
        if choice is None:
            return False
        if choice is False:
            return True

        clone_parent = filedialog.askdirectory(
            title="Choose parent folder for Musubi-Tuner clone",
            initialdir=str(workspace_dir.parent),
            parent=root,
        )
        if not clone_parent:
            return True

        clone_target = Path(clone_parent).expanduser() / "Musubi-Tuner"
        if clone_target.exists():
            if is_valid_musubi_tuner_dir(clone_target):
                use_existing = messagebox.askyesno(
                    "Use existing Musubi-Tuner",
                    f"Found existing Musubi-Tuner at:\n{clone_target}\n\nUse this folder?",
                    parent=root,
                )
                if use_existing:
                    if apply_musubi_dir_setting(clone_target):
                        return True
                    messagebox.showerror(
                        "Invalid Musubi-Tuner",
                        f"Could not use folder:\n{clone_target}",
                        parent=root,
                    )
                return True

            if any(clone_target.iterdir()):
                messagebox.showerror(
                    "Clone target not empty",
                    f"Target folder already exists and is not a Musubi-Tuner checkout:\n{clone_target}\n\n"
                    "Choose a different parent folder.",
                    parent=root,
                )
                return True

        try:
            clone_target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "https://github.com/kohya-ss/musubi-tuner.git", str(clone_target)],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            messagebox.showerror(
                "Clone failed",
                f"Could not run git clone:\n{exc}\n\nInstall Git or clone manually, then select the folder in Settings.",
                parent=root,
            )
            return True

        if result.returncode != 0:
            err_tail = (result.stderr or result.stdout or "").strip()
            if len(err_tail) > 1500:
                err_tail = err_tail[-1500:]
            messagebox.showerror(
                "Clone failed",
                "git clone did not complete successfully.\n\n"
                f"Target: {clone_target}\n\n"
                f"Details:\n{err_tail}",
                parent=root,
            )
            return True

        if not apply_musubi_dir_setting(clone_target):
            messagebox.showerror(
                "Clone incomplete",
                f"Clone finished but folder does not look valid:\n{clone_target}\n\n"
                "Choose the Musubi-Tuner folder manually in Settings.",
                parent=root,
            )
        return True

    def persist_dataset_order() -> None:
        nonlocal settings_state
        save_dataset_order(settings_state, dataset_order)
        save_settings(settings_state)

    def datasets_root_dir() -> Path:
        return runtime_config.training_dir.parent / "Datasets"

    def ensure_datasets_root_dir() -> Path:
        path = datasets_root_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dataset_dir_path(dataset_name: str) -> Path:
        return datasets_root_dir() / dataset_name

    def training_job_dir_path(training_name: str) -> Path:
        return runtime_config.training_dir / training_name

    def ensure_training_job_structure(
        training_name: str,
        default_caption_keyword: str,
        datasets: list[dict] | None = None,
        dataset_name: str = "",
        resolution: int = DEFAULT_RESOLUTION,
        batch_size: int = 1,
        model_name: str = "",
    ) -> tuple[Path, Path, int]:
        # Normalise: support legacy single-dataset callers via dataset_name=
        if not datasets:
            if dataset_name:
                datasets = [{"name": dataset_name, "num_repeats": 1}]
            else:
                raise RuntimeError("No datasets specified for training job structure.")

        job_dir = training_job_dir_path(training_name)
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        created_captions = 0
        caption_text = default_caption_keyword.strip()

        model_name_key = (model_name or "").strip().lower()
        if model_name_key == "ltx-2.3":
            if int(resolution) == 1280:
                toml_width, toml_height = 1280, 720
            else:
                toml_width, toml_height = 1920, 1080
        else:
            toml_width, toml_height = int(resolution), int(resolution)

        toml_lines: list[str] = [
            "[general]",
            f"resolution = [{toml_width}, {toml_height}]",
            'caption_extension = ".txt"',
            f"batch_size = {batch_size}",
            "enable_bucket = true",
            "bucket_no_upscale = false",
            "",
        ]

        for ds_idx, ds in enumerate(datasets):
            ds_name = ds["name"]
            num_repeats = int(ds.get("num_repeats", 1))

            image_files = dataset_image_files(datasets_root_dir(), ds_name)
            if not image_files:
                raise RuntimeError(f"Dataset images not found for: {ds_name}")
            images_dir = image_files[0].parent

            for image_path in image_files:
                caption_path = image_path.with_suffix(".txt")
                if caption_path.exists():
                    continue
                caption_path.write_text(caption_text, encoding="utf-8")
                created_captions += 1

            # First dataset uses "cache" for backward compatibility; additional use "cache_{name}"
            cache_dir = job_dir / "cache" if ds_idx == 0 else job_dir / f"cache_{ds_name}"
            cache_dir.mkdir(parents=True, exist_ok=True)

            toml_lines += [
                "[[datasets]]",
                f'image_directory = "{images_dir.resolve().as_posix()}"',
                f'cache_directory = "{cache_dir.resolve().as_posix()}"',
                f"num_repeats = {num_repeats}",
                "",
            ]

        dataset_toml_path = job_dir / "dataset.toml"
        dataset_toml_path.write_text("\n".join(toml_lines), encoding="utf-8")

        return job_dir, output_dir, created_captions

    def open_settings_dialog(required: bool) -> RuntimeConfig | None:
        current_dir = ""
        if runtime_config is not None:
            current_dir = str(runtime_config.musubi_dir)
        current_musubi_python_path = resolve_musubi_python(Path(current_dir).expanduser()) if current_dir else None
        current_default_caption_keyword = settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, "")
        current_compile_optimizations = settings_state.get(ENABLE_COMPILE_OPTIMIZATIONS_KEY, "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_cuda_allow_tf32 = settings_state.get(ENABLE_CUDA_ALLOW_TF32_KEY, "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_cuda_cudnn_benchmark = settings_state.get(ENABLE_CUDA_CUDNN_BENCHMARK_KEY, "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_fp8_dit = settings_state.get(ENABLE_FP8_DIT_KEY, "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_enable_training_logging = settings_state.get(
            TRAIN_ENABLE_LOGGING_KEY,
            "1",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_train_log_tracker_name = settings_state.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip()
        current_train_stream_to_logger = settings_state.get(
            TRAIN_STREAM_TO_LOGGER_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_auto_start_tensorboard = settings_state.get(
            TRAIN_AUTO_START_TENSORBOARD_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_auto_cleanup_states = settings_state.get(
            TRAIN_AUTO_CLEANUP_STATES_KEY,
            "1",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_train_save_every_n_steps = str(
            get_positive_int_setting(
                settings_state,
                TRAIN_SAVE_EVERY_N_STEPS_KEY,
                DEFAULT_SAVE_EVERY_N_STEPS,
                minimum=1,
            )
        )
        current_gc_cpu_offload = settings_state.get(
            ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        result: RuntimeConfig | None = None
        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("Settings")
        dialog.transient(root)
        dialog.resizable(True, True)
        dialog.configure(bg=bg_panel)
        set_dark_title_bar(dialog)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=0)

        notebook = ttk.Notebook(dialog)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        general_tab = ttk.Frame(notebook, padding=10)
        general_tab.columnconfigure(0, weight=1)
        notebook.add(general_tab, text="  General  ")

        models_tab = ttk.Frame(notebook, padding=10)
        models_tab.columnconfigure(0, weight=1)
        notebook.add(models_tab, text="  Models  ")

        footer = ttk.Frame(dialog, padding=(10, 8, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        musubi_section = ttk.LabelFrame(general_tab, text="Musubi-Tuner", padding=8)
        musubi_section.grid(row=0, column=0, sticky="ew")
        musubi_section.columnconfigure(1, weight=1)

        captions_section = ttk.LabelFrame(general_tab, text="Captions", padding=8)
        captions_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        captions_section.columnconfigure(1, weight=1)

        advanced_section = ttk.LabelFrame(general_tab, text="Training", padding=8)
        advanced_section.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        advanced_section.columnconfigure(0, weight=0)
        advanced_section.columnconfigure(1, weight=1)
        advanced_section.columnconfigure(2, weight=0)
        advanced_section.columnconfigure(3, weight=1)

        # ── Models tab ────────────────────────────────────────────────────
        model_loc_frame = ttk.Frame(models_tab)
        model_loc_frame.grid(row=0, column=0, sticky="ew")
        model_location_var = tk.StringVar(
            value=settings_state.get(MODEL_DOWNLOAD_LOCATION_KEY, DOWNLOAD_LOCATION_MODELS_FOLDER)
        )
        ttk.Label(model_loc_frame, text="Download location:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(
            model_loc_frame,
            textvariable=model_location_var,
            values=list(DOWNLOAD_LOCATIONS),
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="w")
        hf_token_var = tk.StringVar(value=settings_state.get(HF_TOKEN_KEY, ""))
        ttk.Label(model_loc_frame, text="HuggingFace token:").grid(row=0, column=2, sticky="w", padx=(24, 8))
        ttk.Entry(model_loc_frame, textvariable=hf_token_var, show="*", width=36, style="Flat.TEntry").grid(row=0, column=3, sticky="ew")
        model_loc_frame.columnconfigure(3, weight=1)

        # ── ComfyUI path + scan ───────────────────────────────────────────
        _raw_extra = settings_state.get(EXTRA_SEARCH_PATHS_KEY, "")
        import json as _json_extra
        try:
            _extra_paths_list: list[str] = _json_extra.loads(_raw_extra) if _raw_extra else []
        except Exception:
            _extra_paths_list = []

        extra_paths_frame = ttk.Frame(models_tab)
        extra_paths_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        extra_paths_frame.columnconfigure(1, weight=1)
        ttk.Label(extra_paths_frame, text="ComfyUI models path:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        extra_path_var = tk.StringVar(value=_extra_paths_list[0] if _extra_paths_list else "")
        ttk.Entry(extra_paths_frame, textvariable=extra_path_var, style="Flat.TEntry").grid(row=0, column=1, sticky="ew")

        def _browse_extra_path() -> None:
            picked = filedialog.askdirectory(
                parent=dialog,
                title="Select ComfyUI models folder",
                initialdir=extra_path_var.get().strip() or str(Path.home()),
            )
            if picked:
                extra_path_var.set(picked)

        def _scan_all_sources() -> None:
            """Scan Models folder, HF cache, and optionally the ComfyUI path for any missing files."""
            ws_root = download_workspace_root()
            comfy_dir = extra_path_var.get().strip()
            extra = [comfy_dir] if comfy_dir and Path(comfy_dir).is_dir() else []
            found_count = 0
            for mn, comps in DOWNLOAD_MODELS.items():
                for comp in comps:
                    if pending_model_paths.get(mn, {}).get(comp):
                        continue  # already set
                    hit = find_component(mn, comp, ws_root, extra or None)
                    if hit is not None:
                        pending_model_paths.setdefault(mn, {})[comp] = str(hit)
                        found_count += 1
            # Refresh all entry StringVars and status labels
            for mn in DOWNLOAD_MODELS:
                for comp, cpv in _comp_path_vars_all.get(mn, {}).items():
                    new_val = pending_model_paths.get(mn, {}).get(comp, "")
                    if cpv.get() != new_val:
                        cpv.set(new_val)
                _refresh_status(mn)
            if found_count:
                messagebox.showinfo("Scan complete", f"Found {found_count} new file(s). Click Save to apply.", parent=dialog)
            else:
                messagebox.showinfo("Scan complete", "No new files found.", parent=dialog)

        ttk.Button(extra_paths_frame, text="Browse…", command=_browse_extra_path).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(extra_paths_frame, text="Scan for models", command=_scan_all_sources).grid(row=0, column=3, padx=(6, 0))

        # Registry of all component StringVars so _scan_extra_path can update them
        _comp_path_vars_all: dict[str, dict[str, tk.StringVar]] = {}

        models_canvas = tk.Canvas(models_tab, bg=bg_panel, highlightthickness=0)
        models_scrollbar = ttk.Scrollbar(models_tab, orient="vertical", command=models_canvas.yview)
        models_canvas.configure(yscrollcommand=models_scrollbar.set)
        models_canvas.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        models_scrollbar.grid(row=2, column=1, sticky="ns", pady=(10, 0))
        models_tab.rowconfigure(2, weight=1)

        models_inner = ttk.Frame(models_canvas)
        models_inner.columnconfigure(0, weight=1)
        _mw_id = models_canvas.create_window((0, 0), window=models_inner, anchor="nw")

        def _on_models_inner_configure(event: tk.Event) -> None:
            models_canvas.configure(scrollregion=models_canvas.bbox("all"))

        def _on_models_canvas_configure(event: tk.Event) -> None:
            models_canvas.itemconfig(_mw_id, width=event.width)

        models_inner.bind("<Configure>", _on_models_inner_configure)
        models_canvas.bind("<Configure>", _on_models_canvas_configure)

        def _bind_mousewheel(widget: tk.Widget) -> None:
            widget.bind("<MouseWheel>", lambda e: models_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        selected_musubi_path = current_dir
        selected_musubi_python = str(current_musubi_python_path) if current_musubi_python_path is not None else ""

        # pending_model_paths: model_name → {component: path_str}
        import json as _json_settings
        _raw_model_paths = settings_state.get(MODEL_PATHS_KEY, "")
        try:
            pending_model_paths: dict[str, dict[str, str]] = _json_settings.loads(_raw_model_paths) if _raw_model_paths else {}
        except Exception:
            pending_model_paths = {}
        preset_none_label = "---------"
        _raw_preferred_presets = settings_state.get(PREFERRED_PRESETS_BY_FAMILY_KEY, "")
        try:
            _preferred_presets_loaded = _json_settings.loads(_raw_preferred_presets) if _raw_preferred_presets else {}
        except Exception:
            _preferred_presets_loaded = {}
        preferred_preset_by_family: dict[str, str] = {}
        if isinstance(_preferred_presets_loaded, dict):
            preferred_preset_by_family = {
                str(k): str(v)
                for k, v in _preferred_presets_loaded.items()
                if isinstance(k, str) and isinstance(v, str)
            }

        def _preset_names_for_family_settings(family_name: str) -> list[str]:
            names: set[str] = set()
            presets_dir = download_workspace_root() / "Presets"
            if not presets_dir.exists() or not presets_dir.is_dir():
                return []
            for path in sorted(presets_dir.glob(f"*{JOB_PRESET_FILE_SUFFIX}"), key=lambda p: p.name.casefold()):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                payload_family = str(payload.get("family", "")).strip()
                preset_name = str(payload.get("name", "")).strip()
                if payload_family == family_name and preset_name:
                    names.add(preset_name)
            return sorted(names, key=str.casefold)

        preferred_preset_vars: dict[str, tk.StringVar] = {}
        musubi_display_var = tk.StringVar(value=current_dir if current_dir else "(none)")
        default_caption_keyword_var = tk.StringVar(value=current_default_caption_keyword)
        compile_optimizations_var = tk.BooleanVar(value=current_compile_optimizations)
        cuda_allow_tf32_var = tk.BooleanVar(value=current_cuda_allow_tf32)
        cuda_cudnn_benchmark_var = tk.BooleanVar(value=current_cuda_cudnn_benchmark)
        fp8_dit_var = tk.BooleanVar(value=current_fp8_dit)
        gc_cpu_offload_var = tk.BooleanVar(value=current_gc_cpu_offload)
        enable_training_logging_var = tk.BooleanVar(value=current_enable_training_logging)
        train_log_tracker_name_var = tk.StringVar(value=current_train_log_tracker_name)
        stream_to_logger_var = tk.BooleanVar(value=current_train_stream_to_logger)
        auto_start_tensorboard_var = tk.BooleanVar(value=current_auto_start_tensorboard)
        auto_cleanup_states_var = tk.BooleanVar(value=current_auto_cleanup_states)
        train_save_every_default_var = tk.StringVar(value=current_train_save_every_n_steps)

        ttk.Label(musubi_section, text="Musubi-Tuner folder:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        musubi_display = ttk.Label(
            musubi_section, textvariable=musubi_display_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        musubi_display.grid(row=0, column=1, sticky="ew")
        ttk.Label(
            musubi_section,
            text="Python interpreter is managed automatically by this app.",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(captions_section, text="Default caption keyword:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        default_caption_keyword_entry = ttk.Entry(
            captions_section,
            textvariable=default_caption_keyword_var,
            style="Flat.TEntry",
        )
        default_caption_keyword_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(captions_section, text="Leave blank to create empty .txt captions.").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

        training_defaults_section = ttk.LabelFrame(advanced_section, text="Training defaults", padding=8)
        training_defaults_section.grid(row=0, column=0, columnspan=4, sticky="ew")
        training_defaults_section.columnconfigure(1, weight=1)
        ttk.Label(training_defaults_section, text="Save every N steps:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(training_defaults_section, textvariable=train_save_every_default_var, style="Flat.TEntry").grid(
            row=0,
            column=1,
            sticky="w",
        )

        flags_section = ttk.LabelFrame(advanced_section, text="Advanced flags", padding=8)
        flags_section.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        flags_section.columnconfigure(0, weight=1)
        flags_section.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            flags_section,
            text="Enable Torch Compile",
            variable=compile_optimizations_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            flags_section,
            text="Enable Allow TF32",
            variable=cuda_allow_tf32_var,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            flags_section,
            text="Enable cuDNN Benchmark",
            variable=cuda_cudnn_benchmark_var,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            flags_section,
            text="Auto-clean resume state folders",
            variable=auto_cleanup_states_var,
        ).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            flags_section,
            text="Enable FP8 (Lower VRAM)",
            variable=fp8_dit_var,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            flags_section,
            text="Enable Gradient Checkpointing (Lower VRAM)",
            variable=gc_cpu_offload_var,
        ).grid(row=2, column=1, sticky="w", pady=(6, 0))

        logging_section = ttk.LabelFrame(advanced_section, text="Logging & metadata", padding=8)
        logging_section.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        logging_section.columnconfigure(0, weight=0)
        logging_section.columnconfigure(1, weight=1)
        logging_section.columnconfigure(2, weight=0)
        logging_section.columnconfigure(3, weight=1)

        ttk.Checkbutton(
            logging_section,
            text="Enable TensorBoard",
            variable=enable_training_logging_var,
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(logging_section, text="Tracker name:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(logging_section, textvariable=train_log_tracker_name_var, style="Flat.TEntry").grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )
        ttk.Checkbutton(
            logging_section,
            text="Show full training output in app console",
            variable=stream_to_logger_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            logging_section,
            text="Keep TensorBoard running in background",
            variable=auto_start_tensorboard_var,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(
            logging_section,
            text="Logs are stored per job under each Training/<job>/logs folder and can be viewed via TensorBoard.",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # ── Model family sections ──────────────────────────────────────
        _status_vars: dict[str, tk.StringVar] = {}

        _COMPONENT_LABELS: dict[str, str] = {
            "dit": "Model",
            "vae": "VAE",
            "text_encoder": "Text Encoder",
            "t5": "T5",
            "clip": "CLIP",
        }

        def _model_status_str(model_name: str) -> str:
            components = list(DOWNLOAD_MODELS.get(model_name, {}).keys())
            if not components:
                return "Unknown"
            from pathlib import Path as _P
            stored = pending_model_paths.get(model_name, {})
            found = sum(1 for c in components if stored.get(c) and _P(stored[c]).is_file())
            if found == 0:
                return "Not configured"
            if found < len(components):
                return f"Partial ({found}/{len(components)})"
            return "✓ Ready"

        def _refresh_status(model_name: str) -> None:
            sv = _status_vars.get(model_name)
            if sv:
                sv.set(_model_status_str(model_name))

        def _apply_status_color(lbl: ttk.Label, sv: tk.StringVar, *_a: object) -> None:
            val = sv.get()
            if val.startswith("✓"):
                lbl.configure(foreground="#6fcf6f")
            elif val.startswith("Partial"):
                lbl.configure(foreground="#f0b429")
            else:
                lbl.configure(foreground=fg_muted)

        def _make_family_section(parent: ttk.Frame, family_name: str, model_names: list[str], row: int, expanded: bool) -> None:
            section_frame = ttk.Frame(parent)
            section_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            section_frame.columnconfigure(0, weight=1)

            fam_expanded_var = tk.BooleanVar(value=expanded)

            fam_header_btn = ttk.Button(
                section_frame,
                text=f"{'▼' if expanded else '▶'}  {family_name}",
                style="FamilyHeader.TButton",
                command=lambda: _toggle_family(fam_header_btn, fam_body, fam_expanded_var, family_name),
            )
            fam_header_btn.grid(row=0, column=0, sticky="ew")

            fam_body = ttk.Frame(section_frame, padding=(4, 2, 4, 2))
            fam_body.columnconfigure(0, weight=1)
            if expanded:
                fam_body.grid(row=1, column=0, sticky="ew")

            family_preset_names = _preset_names_for_family_settings(family_name)
            preferred_initial = preferred_preset_by_family.get(family_name, "").strip()
            if preferred_initial and preferred_initial not in family_preset_names:
                preferred_initial = ""
            preferred_var = tk.StringVar(value=(preferred_initial or preset_none_label))
            preferred_preset_vars[family_name] = preferred_var

            preset_row = ttk.Frame(fam_body)
            preset_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
            preset_row.columnconfigure(1, weight=1)
            ttk.Label(preset_row, text="Preferred Preset:").grid(row=0, column=0, sticky="w", padx=(22, 8))
            ttk.Combobox(
                preset_row,
                textvariable=preferred_var,
                values=[preset_none_label] + family_preset_names,
                state="readonly",
                width=28,
            ).grid(row=0, column=1, sticky="w")

            for r, mn in enumerate(model_names, start=1):
                display_name = DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn)
                sv = tk.StringVar(value=_model_status_str(mn))
                _status_vars[mn] = sv

                model_block = ttk.Frame(fam_body)
                model_block.grid(row=r, column=0, sticky="ew", pady=(1, 0))
                model_block.columnconfigure(0, weight=1)

                # ─ Model header row ─────────────────────────────────
                hdr = ttk.Frame(model_block)
                hdr.grid(row=0, column=0, sticky="ew")
                hdr.columnconfigure(1, weight=1)

                detail_expanded_var = tk.BooleanVar(value=False)
                detail_frame = ttk.Frame(model_block, padding=(20, 2, 0, 2))
                detail_frame.columnconfigure(1, weight=1)
                # detail_frame is NOT gridded yet (hidden by default)

                expand_btn = ttk.Button(hdr, text="▶", width=2)
                expand_btn.grid(row=0, column=0, padx=(0, 4))

                ttk.Label(hdr, text=display_name, width=28, anchor="w").grid(row=0, column=1, sticky="w")

                status_lbl = ttk.Label(hdr, textvariable=sv, anchor="w", width=18)
                status_lbl.grid(row=0, column=2, padx=(8, 0), sticky="w")
                sv.trace_add("write", lambda *a, lbl=status_lbl, s=sv: _apply_status_color(lbl, s))
                _apply_status_color(status_lbl, sv)

                ttk.Button(
                    hdr, text="Auto-download",
                    command=lambda mn=mn: _auto_download_model(mn),
                ).grid(row=0, column=3, padx=(8, 0))

                # ─ Detail rows (component paths) ─────────────────────
                components = list(DOWNLOAD_MODELS.get(mn, {}).keys())
                _comp_path_vars: dict[str, tk.StringVar] = {}

                for cr, comp in enumerate(components):
                    comp_label = _COMPONENT_LABELS.get(comp, comp.capitalize())
                    # Derive the friendly name for this specific component slot
                    comp_info = DOWNLOAD_MODELS.get(mn, {}).get(comp, {})
                    folder_name = comp_info.get("folder_name", "")
                    friendly = DOWNLOAD_COMPONENT_FRIENDLY_NAMES.get(folder_name, "")
                    label_text = f"{comp_label} ({friendly}):" if friendly else f"{comp_label}:"

                    stored_path = pending_model_paths.get(mn, {}).get(comp, "")
                    cpv = tk.StringVar(value=stored_path)
                    _comp_path_vars[comp] = cpv

                    ttk.Label(detail_frame, text=label_text, anchor="e").grid(
                        row=cr, column=0, sticky="e", padx=(0, 6), pady=1
                    )
                    path_entry = ttk.Entry(
                        detail_frame, textvariable=cpv, style="Flat.TEntry",
                    )
                    path_entry.grid(row=cr, column=1, sticky="ew", pady=1)

                    def _save_path(mn: str = mn, comp: str = comp, cpv: tk.StringVar = cpv) -> None:
                        val = cpv.get().strip()
                        if val:
                            pending_model_paths.setdefault(mn, {})[comp] = val
                        elif mn in pending_model_paths and comp in pending_model_paths[mn]:
                            del pending_model_paths[mn][comp]
                        _refresh_status(mn)

                    path_entry.bind("<FocusOut>", lambda e, mn=mn, comp=comp, cpv=cpv: _save_path(mn, comp, cpv))
                    path_entry.bind("<Return>", lambda e, mn=mn, comp=comp, cpv=cpv: _save_path(mn, comp, cpv))

                    def _browse_comp(mn: str = mn, comp: str = comp, cpv: tk.StringVar = cpv, friendly: str = friendly) -> None:
                        cur = cpv.get().strip()
                        initial = str(Path(cur).parent) if cur and Path(cur).parent.exists() else str(default_models_dir if default_models_dir.exists() else Path.home())
                        title_label = friendly or _COMPONENT_LABELS.get(comp, comp)
                        picked = filedialog.askopenfilename(
                            parent=dialog,
                            title=f"Select {title_label} for {DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn)}",
                            initialdir=initial,
                            filetypes=[("Safetensors / PTH", "*.safetensors *.pth"), ("All files", "*.*")],
                        )
                        if picked:
                            pending_model_paths.setdefault(mn, {})[comp] = picked
                            cpv.set(picked)
                            _refresh_status(mn)

                    ttk.Button(detail_frame, text="Browse", command=_browse_comp).grid(
                        row=cr, column=2, padx=(6, 0), pady=1
                    )

                # Register vars so _scan_extra_path can update them
                _comp_path_vars_all[mn] = _comp_path_vars

                def _toggle_detail(
                    btn: ttk.Button = expand_btn,
                    det: ttk.Frame = detail_frame,
                    var: tk.BooleanVar = detail_expanded_var,
                ) -> None:
                    if var.get():
                        det.grid_remove()
                        var.set(False)
                        btn.configure(text="▶")
                    else:
                        det.grid(row=1, column=0, sticky="ew")
                        var.set(True)
                        btn.configure(text="▼")

                expand_btn.configure(command=_toggle_detail)
                _bind_mousewheel(hdr)
                _bind_mousewheel(detail_frame)

            _bind_mousewheel(fam_body)

        def _toggle_family(
            btn: ttk.Button,
            body: ttk.Frame,
            var: tk.BooleanVar,
            family_name: str,
        ) -> None:
            if var.get():
                body.grid_remove()
                var.set(False)
                btn.configure(text=f"▶  {family_name}")
            else:
                body.grid(row=1, column=0, sticky="ew")
                var.set(True)
                btn.configure(text=f"▼  {family_name}")

        _family_row = 0
        for _fam_name, _fam_models in DOWNLOAD_MODEL_FAMILIES.items():
            _make_family_section(models_inner, _fam_name, _fam_models, _family_row, expanded=(_fam_name == "FLUX.2"))
            _family_row += 1

        _bind_mousewheel(models_inner)

        # ── Auto-download handler ──────────────────────────────────────
        def _auto_download_model(model_name: str) -> None:
            location = model_location_var.get()
            ws_root = download_workspace_root()
            hf_token = hf_token_var.get().strip() or None
            components = list(DOWNLOAD_MODELS.get(model_name, {}).keys())

            from pathlib import Path as _P
            missing = [c for c in components if find_component(model_name, c, ws_root) is None]
            # Also check pending_model_paths
            stored = pending_model_paths.get(model_name, {})
            missing = [c for c in missing if not (stored.get(c) and _P(stored.get(c, "")).is_file())]

            if not missing:
                messagebox.showinfo(
                    "Models found",
                    f"All '{model_name}' files are already available.",
                    parent=dialog,
                )
                _refresh_status(model_name)
                return

            confirmed = messagebox.askyesno(
                "Download models",
                f"The following files for '{model_name}' were not found:\n\n"
                + "\n".join(f"  \u2022 {m}" for m in missing)
                + f"\n\nDownload to: {location}?\n\nThis may take a while.",
                parent=dialog,
            )
            if not confirmed:
                return

            # Resolve which Python to use — prefer the configured Musubi-Tuner venv
            # so that huggingface_hub is available and the process has a real stdout.
            python_exe = selected_musubi_python
            if not python_exe:
                messagebox.showerror(
                    "Python not found",
                    "App venv Python was not found. Run Setup.bat and try again.",
                    parent=dialog,
                )
                return
            cli_script = str(Path(__file__).parent / "download_cli.py")

            error_holder: list[str] = []
            result_holder: dict[str, object] = {}

            log(f"━━━ Downloading {model_name} ({', '.join(missing)}) ━━━")

            def _do_download() -> None:
                for comp in missing:
                    cmd = [
                        python_exe, cli_script,
                        "--model", model_name,
                        "--component", comp,
                        "--ws-root", str(ws_root) if ws_root else "",
                        "--location", location,
                    ]
                    if hf_token:
                        cmd += ["--token", hf_token]

                    log(f"  ↓ {comp}…")
                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        comp_result: str | None = None
                        assert proc.stdout is not None
                        # Read with \r awareness so tqdm in-place updates stream through.
                        _buf = ""
                        while True:
                            chunk = proc.stdout.read(256)
                            if not chunk:
                                break
                            _buf += chunk
                            while True:
                                nl = _buf.find("\n")
                                cr = _buf.find("\r")
                                if nl == -1 and cr == -1:
                                    break
                                if cr != -1 and (nl == -1 or cr < nl):
                                    line = _buf[:cr]
                                    _buf = _buf[cr + 1:]
                                    if line.startswith("RESULT:"):
                                        comp_result = line[7:]
                                    elif line.strip():
                                        log(f"\r    {line.strip()}")
                                else:
                                    line = _buf[:nl].rstrip("\r")
                                    _buf = _buf[nl + 1:]
                                    if line.startswith("RESULT:"):
                                        comp_result = line[7:]
                                    elif line.strip():
                                        log(f"    {line.strip()}")
                        if _buf.strip():
                            log(f"    {_buf.strip()}")
                        proc.wait()
                        if proc.returncode != 0:
                            error_holder.append(
                                f"Download of '{comp}' failed (exit {proc.returncode})"
                            )
                            break
                        if comp_result:
                            result_holder[comp] = comp_result
                            # Persist each component immediately after it downloads.
                            def _save_comp(c: str = comp, p: str = comp_result) -> None:
                                import json as _json_save
                                pending_model_paths.setdefault(model_name, {})[c] = p
                                settings_state[MODEL_PATHS_KEY] = _json_save.dumps(pending_model_paths)
                                save_settings(settings_state)
                                _refresh_status(model_name)
                                # Update entry box if the registry has a StringVar for it
                                sv = _comp_path_vars_all.get(model_name, {}).get(c)
                                if sv is not None:
                                    sv.set(p)
                            dialog.after(0, _save_comp)
                    except Exception as exc:
                        error_holder.append(str(exc))
                        break

                dialog.after(0, _on_dl_done)

            def _on_dl_done() -> None:
                if error_holder:
                    log(f"[ERROR] {error_holder[0]}")
                    messagebox.showerror("Download failed", error_holder[0], parent=dialog)
                    return
                _refresh_status(model_name)
                log(f"━━━ Complete: {model_name} ━━━")
                messagebox.showinfo(
                    "Download complete",
                    f"'{model_name}' is ready.",
                    parent=dialog,
                )

            threading.Thread(target=_do_download, daemon=True).start()

        def browse_musubi() -> None:
            nonlocal selected_musubi_path, selected_musubi_python
            picked = filedialog.askdirectory(
                parent=dialog,
                title="Select Musubi-Tuner folder",
                initialdir=selected_musubi_path or str(Path.home()),
            )
            if picked:
                selected_musubi_path = picked
                musubi_display_var.set(picked)
                detected_python = resolve_musubi_python(Path(picked).expanduser())
                if detected_python is not None:
                    selected_musubi_python = str(detected_python)
                else:
                    selected_musubi_python = ""

        def browse_file(current_path: str, initial_dir_hint: str, title: str) -> str | None:
            initial_dir = initial_dir_hint
            if default_models_dir.exists() and default_models_dir.is_dir():
                initial_dir = str(default_models_dir)
            if current_path:
                current_parent = Path(current_path).expanduser().parent
                initial_dir = str(current_parent)
            picked = filedialog.askopenfilename(
                parent=dialog,
                title=title,
                initialdir=initial_dir or str(Path.home()),
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
            )
            return picked if picked else None

        def normalize_model_checkpoint_path(raw_path: str | None) -> str:
            if not raw_path:
                return ""
            candidate = Path(raw_path).expanduser()
            if candidate.is_file() and candidate.name.lower().endswith(".safetensors.index.json"):
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return str(candidate)
                weight_map = payload.get("weight_map", {})
                if not isinstance(weight_map, dict) or not weight_map:
                    return str(candidate)
                shard_names = sorted({str(v) for v in weight_map.values() if isinstance(v, str) and v.lower().endswith(".safetensors")})
                preferred = next((name for name in shard_names if re.search(r"-00001-of-\d+\.safetensors$", name, flags=re.IGNORECASE)), None)
                shard_name = preferred or (shard_names[0] if shard_names else None)
                if shard_name:
                    shard_path = candidate.parent / shard_name
                    if shard_path.is_file():
                        return str(shard_path)
            return str(candidate)

        def save_and_close() -> None:
            nonlocal result, settings_state
            # Force any focused entry widget to commit (triggers FocusOut → _save_path)
            dialog.focus_set()
            if not selected_musubi_path:
                messagebox.showerror("Missing folder", "Musubi-Tuner folder is not set.", parent=dialog)
                return

            try:
                save_every_default_value = int(train_save_every_default_var.get().strip())
                if save_every_default_value < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    "Training default 'Save every N steps' must be a positive integer.",
                    parent=dialog,
                )
                return

            musubi_path = Path(selected_musubi_path).expanduser()
            if not musubi_path.exists() or not musubi_path.is_dir():
                messagebox.showerror("Invalid folder", "Choose a valid Musubi-Tuner folder.", parent=dialog)
                return

            musubi_python_path = resolve_musubi_python(musubi_path)
            if musubi_python_path is None:
                messagebox.showerror(
                    "Python venv not found",
                    "App venv Python was not found. Run Setup.bat first.",
                    parent=dialog,
                )
                return

            import json as _json_save
            settings_state[MUSUBI_DIR_KEY] = str(musubi_path)
            settings_state[MUSUBI_PYTHON_KEY] = ""
            settings_state[MODEL_PATHS_KEY] = _json_save.dumps(pending_model_paths)
            # Backward compat: derive legacy keys from pending_model_paths
            _active_klein = settings_state.get(KLEIN_MODEL_VERSION_KEY, "klein-base-9b") or "klein-base-9b"
            _kpaths = pending_model_paths.get(_active_klein, {})
            settings_state[KLEIN_MODEL_VERSION_KEY] = _active_klein
            settings_state[KLEIN_DIT_KEY] = _kpaths.get("dit", "")
            settings_state[KLEIN_VAE_KEY] = _kpaths.get("vae", "")
            settings_state[KLEIN_TEXT_ENCODER_KEY] = _kpaths.get("text_encoder", "")
            _ltx_paths = pending_model_paths.get("ltx-2.3", {})
            settings_state[LTX_MODEL_VERSION_KEY] = "ltx-2.3" if _ltx_paths else ""
            settings_state[LTX_DIT_KEY] = _ltx_paths.get("dit", "")
            settings_state[LTX_VAE_KEY] = _ltx_paths.get("vae", "")
            settings_state[LTX_TEXT_ENCODER_KEY] = _ltx_paths.get("text_encoder", "")
            settings_state[DEFAULT_CAPTION_KEYWORD_KEY] = default_caption_keyword_var.get().strip()
            settings_state[ENABLE_COMPILE_OPTIMIZATIONS_KEY] = "1" if compile_optimizations_var.get() else "0"
            settings_state[ENABLE_CUDA_ALLOW_TF32_KEY] = "1" if cuda_allow_tf32_var.get() else "0"
            settings_state[ENABLE_CUDA_CUDNN_BENCHMARK_KEY] = "1" if cuda_cudnn_benchmark_var.get() else "0"
            settings_state[ENABLE_FP8_DIT_KEY] = "1" if fp8_dit_var.get() else "0"
            settings_state[ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY] = "1" if gc_cpu_offload_var.get() else "0"
            settings_state[TRAIN_ENABLE_LOGGING_KEY] = "1" if enable_training_logging_var.get() else "0"
            settings_state[TRAIN_LOG_BACKEND_KEY] = "tensorboard"
            settings_state[TRAIN_LOG_TRACKER_NAME_KEY] = train_log_tracker_name_var.get().strip()
            settings_state[TRAIN_STREAM_TO_LOGGER_KEY] = "1" if stream_to_logger_var.get() else "0"
            settings_state[TRAIN_AUTO_START_TENSORBOARD_KEY] = "1" if auto_start_tensorboard_var.get() else "0"
            settings_state[TRAIN_AUTO_CLEANUP_STATES_KEY] = "1" if auto_cleanup_states_var.get() else "0"
            settings_state[TRAIN_SAVE_EVERY_N_STEPS_KEY] = str(save_every_default_value)
            preferred_to_save = {
                family_name: var.get().strip()
                for family_name, var in preferred_preset_vars.items()
                if var.get().strip() and var.get().strip() != preset_none_label
            }
            settings_state[PREFERRED_PRESETS_BY_FAMILY_KEY] = _json_save.dumps(preferred_to_save)
            settings_state[MODEL_DOWNLOAD_LOCATION_KEY] = model_location_var.get()
            settings_state[HF_TOKEN_KEY] = hf_token_var.get().strip()
            _ep = extra_path_var.get().strip()
            settings_state[EXTRA_SEARCH_PATHS_KEY] = _json_save.dumps([_ep] if _ep else [])
            save_settings(settings_state)
            result = runtime_config_from_settings(settings_state)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        def reset_settings() -> None:
            nonlocal result, settings_state, settings_reset_requested
            confirmed = messagebox.askyesno(
                "Reset settings",
                "Delete settings.json and reset all saved settings?",
                parent=dialog,
            )
            if not confirmed:
                return

            try:
                if SETTINGS_FILE.exists():
                    SETTINGS_FILE.unlink()
            except OSError as exc:
                messagebox.showerror("Reset failed", f"Could not delete settings file:\n{exc}", parent=dialog)
                return

            settings_reset_requested = True
            settings_state = {}
            result = None
            dialog.destroy()
            root.after_idle(root.destroy)

        ttk.Button(musubi_section, text="Browse Folder", command=browse_musubi).grid(row=0, column=2, padx=(8, 0))
        button_row = ttk.Frame(footer)
        button_row.grid(row=0, column=0, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        ttk.Button(button_row, text="Reset Settings", command=reset_settings).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Cancel", command=cancel_and_close).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=2)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.update_idletasks()

        content_w = max(notebook.winfo_reqwidth(), footer.winfo_reqwidth()) + 44
        content_h = notebook.winfo_reqheight() + footer.winfo_reqheight() + 20
        max_w = max(760, dialog.winfo_screenwidth() - 80)
        max_h = max(480, dialog.winfo_screenheight() - 80)
        win_w = max(780, min(1080, content_w))
        win_w = min(win_w, max_w)
        win_h = min(max(520, content_h), max_h)
        dialog.geometry(f"{win_w}x{win_h}")
        center_window(dialog)
        dialog.deiconify()
        dialog.grab_set()
        dialog.focus_set()
        root.wait_window(dialog)

        if required and result is None:
            return None

        return result

    if runtime_config is None or not is_valid_musubi_tuner_dir(runtime_config.musubi_dir):
        proceed = prompt_to_clone_or_select_musubi()
        if not proceed:
            root.destroy()
            return 1

        if runtime_config is None or not is_valid_musubi_tuner_dir(runtime_config.musubi_dir):
            messagebox.showinfo(
                "First launch setup",
                "Musubi-Tuner location is required before this app can run. Set it in Settings now.",
                parent=root,
            )
            runtime_config = open_settings_dialog(required=True)
            if runtime_config is None or not is_valid_musubi_tuner_dir(runtime_config.musubi_dir):
                root.destroy()
                return 1

    maybe_autostart_tensorboard()

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=0)
    root.rowconfigure(2, weight=1, minsize=360)
    root.rowconfigure(3, weight=0, minsize=158)

    header = ttk.Frame(root, padding=8)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    training_path_var = tk.StringVar(value=f"Training folder: {runtime_config.training_dir}")
    ttk.Label(header, textvariable=training_path_var).grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.grid(row=1, column=0, sticky="ew")
    controls.columnconfigure(3, weight=1)

    def apply_settings_from_dialog(required: bool = False) -> bool:
        nonlocal runtime_config, dataset_order
        updated = open_settings_dialog(required=required)
        if updated is None:
            return False

        runtime_config = updated
        training_path_var.set(f"Training folder: {runtime_config.training_dir}")
        dataset_order = load_dataset_order(settings_state)
        rebuild_folder_list(force=True)
        load_job_queue_from_disk()
        refresh_job_queue_list()
        update_start_button_state()
        maybe_autostart_tensorboard()
        return True

    def create_dataset() -> None:
        def is_valid_dataset_name(name: str) -> bool:
            return DATASET_NAME_PATTERN.fullmatch(name) is not None

        dataset_name = simpledialog.askstring("Create Dataset", "Enter dataset name:", parent=root)
        if dataset_name is None:
            return

        dataset_name = dataset_name.strip()
        if not dataset_name:
            messagebox.showerror("Invalid name", "Dataset name cannot be empty.", parent=root)
            return

        if not is_valid_dataset_name(dataset_name):
            messagebox.showerror(
                "Invalid name",
                "Use only letters, numbers, '_' or '-'.",
                parent=root,
            )
            return

        datasets_root = ensure_datasets_root_dir()
        dataset_dir = datasets_root / dataset_name
        if dataset_dir.exists():
            messagebox.showerror("Name unavailable", f"Dataset '{dataset_name}' already exists.", parent=root)
            return

        images_dir = dataset_dir

        try:
            images_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            messagebox.showerror("Create failed", f"Could not create dataset structure:\n{exc}", parent=root)
            return

        selected_files = filedialog.askopenfilenames(
            parent=root,
            title="Select images and optional captions",
            filetypes=[
                ("Images and captions", "*.png *.jpg *.jpeg *.txt"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )

        copied_count = 0
        for raw_path in selected_files:
            source = Path(raw_path)
            if not source.exists() or not source.is_file():
                continue

            suffix = source.suffix.lower()
            if suffix not in VALID_IMAGE_EXTENSIONS and suffix != ".txt":
                continue

            destination = images_dir / source.name
            if destination.exists():
                stem = source.stem
                candidate_index = 1
                while destination.exists():
                    destination = images_dir / f"{stem}_{candidate_index}{source.suffix}"
                    candidate_index += 1

            try:
                shutil.copy2(source, destination)
                copied_count += 1
            except OSError:
                continue

        rebuild_folder_list(force=True)
        if copied_count:
            messagebox.showinfo(
                "Dataset created",
                f"Created dataset '{dataset_name}' and imported {copied_count} file(s).",
                parent=root,
            )
        else:
            messagebox.showinfo(
                "Dataset created",
                f"Created dataset '{dataset_name}' with an empty dataset folder.",
                parent=root,
            )

    def open_dataset_in_file_manager(dataset_name: str) -> None:
        dataset_dir = dataset_dir_path(dataset_name)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            messagebox.showerror("Open failed", f"Dataset folder not found:\n{dataset_dir}", parent=root)
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(dataset_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(dataset_dir)])
            else:
                subprocess.Popen(["xdg-open", str(dataset_dir)])
        except OSError as exc:
            messagebox.showerror("Open failed", f"Could not open folder:\n{exc}", parent=root)

    def open_metrics_viewer_dialog() -> None:
        nonlocal tensorboard_launch_in_progress

        if tensorboard_launch_in_progress:
            log("[Metrics Viewer] Launch already in progress. Please wait...")
            return

        tensorboard_launch_in_progress = True
        if metrics_viewer_button is not None:
            try:
                metrics_viewer_button.configure(state="disabled")
            except tk.TclError:
                pass

        log("[Metrics Viewer] Launching TensorBoard")

        try:
            if runtime_config is None:
                messagebox.showerror(
                    "Metrics Viewer",
                    "Runtime configuration is not available. Open Settings and verify Musubi paths.",
                    parent=root,
                )
                return
            if resolve_tensorboard_python() is None:
                messagebox.showerror(
                    "Metrics Viewer",
                    "TensorBoard is not installed in Musubi-Tuner Python.\n\n"
                    "Install it in Musubi Python:\n"
                    "pip install tensorboard",
                    parent=root,
                )
                return

            if is_tensorboard_running():
                log("[Metrics Viewer] TensorBoard is already running. Opening browser...")
            else:
                log("[Metrics Viewer] Starting TensorBoard... please wait a few seconds.")

            if not launch_tensorboard_background():
                log("[Metrics Viewer] Failed to start TensorBoard.")
                messagebox.showerror(
                    "Metrics Viewer",
                    "Could not start TensorBoard. Check the app log for details.",
                    parent=root,
                )
                return

            if is_tensorboard_running():
                log(f"[Metrics Viewer] TensorBoard is ready at {tensorboard_url}")
            else:
                log("[Metrics Viewer] TensorBoard is still starting. Opening browser now; refresh in a few seconds if needed.")

            opened = webbrowser.open(tensorboard_url, new=2)
            if opened:
                log("[Metrics Viewer] Opened TensorBoard in your default browser.")
                return

            try:
                if sys.platform == "win32":
                    os.startfile(tensorboard_url)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", tensorboard_url])
                else:
                    subprocess.Popen(["xdg-open", tensorboard_url])
                log("[Metrics Viewer] Opened TensorBoard using OS browser launcher.")
                return
            except OSError:
                log("[Metrics Viewer] Browser auto-open failed. Open the URL manually.")
                messagebox.showerror(
                    "Metrics Viewer",
                    f"TensorBoard is running, but the browser could not be opened automatically.\n\nOpen manually:\n{tensorboard_url}",
                    parent=root,
                )
        finally:
            tensorboard_launch_in_progress = False
            if metrics_viewer_button is not None:
                try:
                    metrics_viewer_button.configure(state="normal")
                except tk.TclError:
                    pass

    def add_images_to_dataset(dataset_name: str) -> None:
        dataset_dir = dataset_dir_path(dataset_name)
        images_dir = dataset_dir
        allowed_import_suffixes = VALID_IMAGE_EXTENSIONS | {".txt"}
        if not images_dir.exists():
            try:
                images_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Add images failed", f"Could not create dataset folder:\n{exc}", parent=root)
                return

        selected_files = filedialog.askopenfilenames(
            parent=root,
            title=f"Add images/captions to {dataset_name}",
            filetypes=[
                ("Images and captions", "*.png *.jpg *.jpeg *.txt"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not selected_files:
            return

        collisions = 0
        for raw_path in selected_files:
            source = Path(raw_path)
            if source.is_file() and source.suffix.lower() in allowed_import_suffixes:
                if (images_dir / source.name).exists():
                    collisions += 1

        overwrite_existing = False
        if collisions:
            choice = messagebox.askyesnocancel(
                "File name conflicts",
                (
                    f"{collisions} selected file(s) already exist in this dataset.\n\n"
                    "Yes = Replace all existing files\n"
                    "No = Skip existing files\n"
                    "Cancel = Abort"
                ),
                parent=root,
            )
            if choice is None:
                return
            overwrite_existing = bool(choice)

        skipped_errors = 0

        for raw_path in selected_files:
            source = Path(raw_path)
            if not source.exists() or not source.is_file() or source.suffix.lower() not in allowed_import_suffixes:
                continue

            destination = images_dir / source.name
            if destination.exists() and not overwrite_existing:
                continue

            try:
                shutil.copy2(source, destination)
            except OSError:
                skipped_errors += 1

        rebuild_folder_list(force=True)
        if skipped_errors:
            messagebox.showwarning(
                "Add files warning",
                f"Finished with {skipped_errors} copy error(s).",
                parent=root,
            )

    def open_edit_dataset_dialog(dataset_name: str) -> None:
        columns = 4
        tile_size_px = 300
        tile_gap_px = 4
        grid_width_px = (tile_size_px * columns) + (tile_gap_px * (columns + 1))
        # Include dialog frame padding and the vertical scrollbar lane.
        dialog_width_px = grid_width_px + 80
        tile_side_pad_px = tile_gap_px // 2

        dataset_dir = dataset_dir_path(dataset_name)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            messagebox.showerror("Edit dataset", f"Dataset folder not found:\n{dataset_dir}", parent=root)
            return

        image_paths = dataset_image_files(datasets_root_dir(), dataset_name)
        if not image_paths:
            messagebox.showinfo("Edit dataset", "No images found in this dataset.", parent=root)
            return

        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title(f"Edit Dataset: {dataset_name}")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(False, True)
        set_dark_title_bar(dialog)
        dialog.minsize(dialog_width_px, 760)
        dialog.geometry(f"{dialog_width_px}x920")

        outer = ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        ttk.Label(
            outer,
            text=f"{dataset_name} ({len(image_paths)} images)",
            style="TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        grid_host = ttk.Frame(outer)
        grid_host.grid(row=1, column=0, sticky="nsew")
        grid_host.columnconfigure(0, weight=1)
        grid_host.rowconfigure(0, weight=1)

        editor_canvas = tk.Canvas(grid_host, highlightthickness=0, bg=bg_panel)
        editor_scroll = ttk.Scrollbar(grid_host, orient="vertical", command=editor_canvas.yview, style="Dark.Vertical.TScrollbar")
        editor_inner = ttk.Frame(editor_canvas)
        editor_inner_id = editor_canvas.create_window((0, 0), window=editor_inner, anchor="nw")
        editor_canvas.configure(yscrollcommand=editor_scroll.set)

        editor_canvas.grid(row=0, column=0, sticky="nsew")
        editor_scroll.grid(row=0, column=1, sticky="ns")

        thumb_refs: list[ImageTk.PhotoImage] = []
        caption_path_by_widget: dict[tk.Text, Path] = {}
        pending_save_by_widget: dict[tk.Text, str] = {}

        def build_caption_thumb(image_path: Path) -> ImageTk.PhotoImage:
            thumb_size = (tile_size_px, tile_size_px)
            try:
                image = Image.open(image_path).convert("RGB")
                src_w, src_h = image.size
                dst_w, dst_h = thumb_size
                # Fit the full image inside the tile without cropping.
                scale = min(dst_w / src_w, dst_h / src_h)
                resized_w = max(1, int(round(src_w * scale)))
                resized_h = max(1, int(round(src_h * scale)))
                resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
                image = Image.new("RGB", thumb_size, color="#000000")
                paste_x = (dst_w - resized_w) // 2
                paste_y = (dst_h - resized_h) // 2
                image.paste(resized, (paste_x, paste_y))
            except Exception:
                image = Image.new("RGB", thumb_size, color="#000000")

            image_rgba = image.convert("RGBA")
            border_overlay = Image.new("RGBA", thumb_size, (0, 0, 0, 0))
            border_draw = ImageDraw.Draw(border_overlay)
            border_draw.rectangle((0, 0, thumb_size[0] - 1, thumb_size[1] - 1), outline=(255, 255, 255, 96), width=1)
            composited = Image.alpha_composite(image_rgba, border_overlay).convert("RGB")
            return ImageTk.PhotoImage(composited, master=root)

        def flush_caption_save(widget: tk.Text) -> None:
            after_id = pending_save_by_widget.pop(widget, None)
            if after_id is not None:
                try:
                    dialog.after_cancel(after_id)
                except Exception:
                    pass
            caption_path = caption_path_by_widget.get(widget)
            if caption_path is None:
                return
            content = widget.get("1.0", "end-1c")
            try:
                caption_path.write_text(content, encoding="utf-8")
            except OSError as exc:
                log(f"[Dataset Editor] Failed to save caption '{caption_path.name}': {exc}")

        def schedule_caption_save(event: tk.Event) -> None:
            widget = event.widget
            if not isinstance(widget, tk.Text):
                return
            if not widget.edit_modified():
                return
            widget.edit_modified(False)

            previous_after_id = pending_save_by_widget.get(widget)
            if previous_after_id is not None:
                try:
                    dialog.after_cancel(previous_after_id)
                except Exception:
                    pass
            pending_save_by_widget[widget] = dialog.after(220, lambda w=widget: flush_caption_save(w))

        def on_caption_focus_out(event: tk.Event) -> None:
            widget = event.widget
            if isinstance(widget, tk.Text):
                flush_caption_save(widget)

        def on_editor_inner_configure(_event: tk.Event) -> None:
            editor_canvas.configure(scrollregion=editor_canvas.bbox("all"))

        def on_editor_canvas_configure(event: tk.Event) -> None:
            editor_canvas.itemconfigure(editor_inner_id, width=event.width)

        def on_editor_mousewheel(event: tk.Event) -> str:
            delta = int(-event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            editor_canvas.yview_scroll(delta, "units")
            return "break"

        def on_editor_linux_up(_event: tk.Event) -> str:
            editor_canvas.yview_scroll(-1, "units")
            return "break"

        def on_editor_linux_down(_event: tk.Event) -> str:
            editor_canvas.yview_scroll(1, "units")
            return "break"

        def attach_autohide_scrollbar(text_widget: tk.Text, scrollbar_widget: ttk.Scrollbar) -> None:
            scroll_state = {"visible": False}

            def on_yscroll(first: str, last: str) -> None:
                scrollbar_widget.set(first, last)
                try:
                    first_value = float(first)
                    last_value = float(last)
                except (TypeError, ValueError):
                    first_value = 0.0
                    last_value = 1.0

                needs_scrollbar = first_value > 0.0 or last_value < 0.999
                if needs_scrollbar and not scroll_state["visible"]:
                    scrollbar_widget.pack(side="right", fill="y")
                    scroll_state["visible"] = True
                elif (not needs_scrollbar) and scroll_state["visible"]:
                    scrollbar_widget.pack_forget()
                    scroll_state["visible"] = False

            text_widget.configure(yscrollcommand=on_yscroll)

            def prime_scrollbar() -> None:
                first_value, last_value = text_widget.yview()
                on_yscroll(str(first_value), str(last_value))

            dialog.after_idle(prime_scrollbar)

        for column_index in range(columns):
            editor_inner.columnconfigure(column_index, weight=0, minsize=tile_size_px)

        for idx, image_path in enumerate(image_paths):
            caption_path = image_path.with_suffix(".txt")
            caption_text = ""
            if caption_path.exists() and caption_path.is_file():
                try:
                    caption_text = caption_path.read_text(encoding="utf-8")
                except OSError:
                    caption_text = ""

            item_frame = ttk.Frame(editor_inner, padding=(4, 4, 4, 6), style="TFrame")
            item_frame.grid(row=idx // columns, column=idx % columns, sticky="n", padx=tile_side_pad_px, pady=4)
            item_frame.columnconfigure(0, weight=1)

            photo = build_caption_thumb(image_path)
            thumb_refs.append(photo)
            image_label = ttk.Label(item_frame, image=photo, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")
            image_label.bind("<MouseWheel>", on_editor_mousewheel)
            image_label.bind("<Button-4>", on_editor_linux_up)
            image_label.bind("<Button-5>", on_editor_linux_down)

            caption_shell = tk.Frame(
                item_frame,
                width=tile_size_px,
                height=64,
                bg="#111826",
                highlightthickness=1,
                highlightbackground="#2a3a50",
                highlightcolor="#4a6ea3",
                bd=0,
            )
            caption_shell.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            caption_shell.grid_propagate(False)

            caption_widget = tk.Text(
                caption_shell,
                width=1,
                height=3,
                wrap="word",
                bg="#111826",
                fg=fg_text,
                insertbackground=fg_text,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                padx=6,
                pady=5,
            )
            caption_widget.pack(side="left", fill="both", expand=True)

            caption_scroll = ttk.Scrollbar(
                caption_shell,
                orient="vertical",
                command=caption_widget.yview,
                style="Dark.Vertical.TScrollbar",
            )
            attach_autohide_scrollbar(caption_widget, caption_scroll)
            caption_widget.insert("1.0", caption_text)
            caption_widget.edit_modified(False)
            caption_widget.bind("<<Modified>>", schedule_caption_save)
            caption_widget.bind("<FocusOut>", on_caption_focus_out)
            caption_widget.bind("<MouseWheel>", on_editor_mousewheel)
            caption_widget.bind("<Button-4>", on_editor_linux_up)
            caption_widget.bind("<Button-5>", on_editor_linux_down)
            caption_path_by_widget[caption_widget] = caption_path

        def close_editor() -> None:
            for text_widget in list(caption_path_by_widget.keys()):
                flush_caption_save(text_widget)
            dialog.destroy()

        actions = ttk.Frame(outer)
        actions.grid(row=2, column=0, sticky="e", pady=(10, 0))
        ttk.Button(actions, text="Close", command=close_editor).grid(row=0, column=0)

        editor_inner.bind("<Configure>", on_editor_inner_configure)
        editor_canvas.bind("<MouseWheel>", on_editor_mousewheel)
        editor_canvas.bind("<Button-4>", on_editor_linux_up)
        editor_canvas.bind("<Button-5>", on_editor_linux_down)
        editor_inner.bind("<MouseWheel>", on_editor_mousewheel)
        editor_inner.bind("<Button-4>", on_editor_linux_up)
        editor_inner.bind("<Button-5>", on_editor_linux_down)
        dialog.protocol("WM_DELETE_WINDOW", close_editor)

        center_window(dialog)
        dialog.deiconify()
        root.wait_window(dialog)

    def dataset_output_safetensors(dataset_name: str) -> list[Path]:
        output_dir = runtime_config.training_dir / dataset_name / "output"
        if not output_dir.exists() or not output_dir.is_dir():
            return []
        return sorted([p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors"])

    def compact_merge_selection_token(selected_files: list[str]) -> str:
        step_values: list[int] = []
        for raw_path in selected_files:
            match = re.search(r"step0*(\d+)", Path(raw_path).name, re.IGNORECASE)
            if match:
                step_values.append(int(match.group(1)))

        if not step_values:
            return f"n{len(selected_files)}"

        unique_steps = sorted(set(step_values))
        if len(unique_steps) <= 4:
            return "s" + "-".join(str(step) for step in unique_steps)

        return f"s{unique_steps[0]}-{unique_steps[-1]}x{len(unique_steps)}"

    def merge_preset_file_token(preset_name: str) -> str:
        preset_key = preset_name.strip().lower()
        if preset_key == "smooth":
            return "Smooth"
        if preset_key == "anti-overfit":
            return "NoOverfit"
        return "Balance"

    def merge_preset_tooltip_text() -> str:
        return (
            "Preset guide\n"
            "Balanced: General starting point for most training runs.\n"
            "Smooth: Prefer when training converged early and you want a smoother blend.\n"
            "Anti-overfit: Prefer when late checkpoints look overfit and too close to training images."
        )

    def merge_mode_tooltip_text() -> str:
        return (
            "Merge mode guide\n"
            "BETA: Uses a single constant decay rate across all checkpoints.\n"
            "BETA + BETA2: Interpolates decay from beta to beta2 across the merge order.\n"
            "SIGMA_REL: Uses Power Function EMA to compute decay schedule and reduce first-checkpoint bias."
        )

    def attach_hover_tooltip(widget: tk.Widget, text_provider: Callable[[], str] | str) -> None:
        tooltip_window: tk.Toplevel | None = None

        def tooltip_text() -> str:
            if callable(text_provider):
                return text_provider().strip()
            return str(text_provider).strip()

        def show_tooltip(_event: tk.Event | None = None) -> None:
            nonlocal tooltip_window
            text = tooltip_text()
            if not text:
                return
            hide_tooltip()
            tooltip_window = tk.Toplevel(widget)
            tooltip_window.wm_overrideredirect(True)
            tooltip_window.configure(bg="#1f1f1f")
            label = tk.Label(
                tooltip_window,
                text=text,
                justify="left",
                bg="#1f1f1f",
                fg=fg_text,
                relief="solid",
                bd=1,
                padx=8,
                pady=6,
            )
            label.pack()
            x = widget.winfo_rootx() + 8
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            tooltip_window.wm_geometry(f"+{x}+{y}")

        def hide_tooltip(_event: tk.Event | None = None) -> None:
            nonlocal tooltip_window
            if tooltip_window is not None and tooltip_window.winfo_exists():
                tooltip_window.destroy()
            tooltip_window = None

        widget.bind("<Enter>", show_tooltip, add="+")
        widget.bind("<Leave>", hide_tooltip, add="+")
        widget.bind("<ButtonPress>", hide_tooltip, add="+")
        widget.bind("<FocusOut>", hide_tooltip, add="+")

    def next_merged_output_path(
        dataset_name: str,
        output_dir: Path,
        merge_mode_suffix: str,
        preset_name: str,
        selected_files: list[str],
    ) -> Path:
        # Keep method name in the filename to make comparisons between merge modes easy.
        method_token = re.sub(r"[^A-Za-z0-9]+", "_", merge_mode_suffix.strip()).strip("_")
        preset_token = merge_preset_file_token(preset_name)
        selection_token = compact_merge_selection_token(selected_files)
        if method_token and preset_token:
            base_name = f"{dataset_name}_merged_{method_token}_{preset_token}_{selection_token}"
        elif method_token:
            base_name = f"{dataset_name}_merged_{method_token}_{selection_token}"
        else:
            base_name = f"{dataset_name}_merged_{selection_token}"
        candidate = output_dir / f"{base_name}.safetensors"
        if not candidate.exists():
            return candidate

        index = 2
        while True:
            candidate = output_dir / f"{base_name}_{index}.safetensors"
            if not candidate.exists():
                return candidate
            index += 1

    def post_hoc_ema_mode_args_for_preset(preset_name: str) -> dict[str, list[str]]:
        preset_key = preset_name.strip().lower()
        if preset_key == "smooth":
            return {
                "beta": ["--beta", "0.95"],
                "beta2": ["--beta", "0.95", "--beta2", "0.98"],
                "sigma_rel": ["--sigma_rel", "0.25"],
            }
        if preset_key == "anti-overfit":
            return {
                "beta": ["--beta", "0.8"],
                "beta2": ["--beta", "0.80", "--beta2", "0.90"],
                "sigma_rel": ["--sigma_rel", "0.15"],
            }
        return {
            "beta": ["--beta", "0.9"],
            "beta2": ["--beta", "0.90", "--beta2", "0.95"],
            "sigma_rel": ["--sigma_rel", "0.2"],
        }

    def ask_lora_merge_options(
        dataset_name: str,
        available_loras: list[Path],
    ) -> tuple[list[str], list[tuple[str, str, list[str], str]]] | None:
        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("LoRA Post-Hoc EMA Merge")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=bg_panel)
        set_dark_title_bar(dialog)
        dialog.minsize(380, 460)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text=f"LoRAs in output for {dataset_name}:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Select the LoRAs you want to merge.").grid(row=1, column=0, sticky="w", pady=(4, 0))

        selected_box_frame = ttk.Frame(frame)
        selected_box_frame.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        selected_box_frame.columnconfigure(0, weight=1)
        selected_box_frame.rowconfigure(0, weight=1)

        selected_list = tk.Listbox(
            selected_box_frame,
            selectmode="extended",
            exportselection=False,
            activestyle="none",
            width=1,
            bg="#1f1f1f",
            fg=fg_text,
            highlightthickness=1,
            highlightbackground=border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
            height=8,
        )
        selected_list.grid(row=0, column=0, sticky="nsew")
        selected_scroll = ttk.Scrollbar(
            selected_box_frame,
            orient="vertical",
            command=selected_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        selected_scroll.grid(row=0, column=1, sticky="ns")
        selected_list.configure(yscrollcommand=selected_scroll.set)

        for lora_path in available_loras:
            selected_list.insert("end", lora_path.name)

        ttk.Label(
            frame,
            text="Post-Hoc EMA smooths checkpoints from the same run into one more stable LoRA.",
        ).grid(row=3, column=0, sticky="ew", pady=(10, 0))

        options_frame = ttk.Frame(frame)
        options_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

        preset_section = ttk.LabelFrame(options_frame, text="Preset(s)", padding=6)
        preset_section.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        attach_hover_tooltip(preset_section, merge_preset_tooltip_text)

        mode_section = ttk.LabelFrame(options_frame, text="Mode(s)", padding=6)
        mode_section.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        attach_hover_tooltip(mode_section, merge_mode_tooltip_text)

        preset_balanced_var = tk.BooleanVar(value=False)
        preset_smooth_var = tk.BooleanVar(value=False)
        preset_anti_overfit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(preset_section, text="Balanced", variable=preset_balanced_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(preset_section, text="Smooth", variable=preset_smooth_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(preset_section, text="Anti-overfit", variable=preset_anti_overfit_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        mode_beta_var = tk.BooleanVar(value=False)
        mode_beta2_var = tk.BooleanVar(value=False)
        mode_sigma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_section, text="BETA (Default)", variable=mode_beta_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(mode_section, text="BETA + BETA2 (Interpolated)", variable=mode_beta2_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(mode_section, text="SIGMA_REL", variable=mode_sigma_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        button_row = ttk.Frame(frame)
        button_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)

        choice: tuple[list[str], list[tuple[str, str, list[str], str]]] | None = None

        def choose_and_close() -> None:
            nonlocal choice
            picked_indices = selected_list.curselection()
            selected_file_paths = [str(available_loras[i]) for i in picked_indices]
            if len(selected_file_paths) < 2:
                messagebox.showerror(
                    "Merge unavailable",
                    "Select at least 2 .safetensors files for Post-Hoc EMA merge.",
                    parent=dialog,
                )
                return

            selected_preset_names: list[str] = []
            if preset_balanced_var.get():
                selected_preset_names.append("Balanced")
            if preset_smooth_var.get():
                selected_preset_names.append("Smooth")
            if preset_anti_overfit_var.get():
                selected_preset_names.append("Anti-overfit")
            if not selected_preset_names:
                messagebox.showerror(
                    "Merge unavailable",
                    "Select at least one preset.",
                    parent=dialog,
                )
                return

            mode_defs: list[tuple[str, str, str]] = []
            if mode_beta_var.get():
                mode_defs.append(("BETA", "Beta", "beta"))
            if mode_beta2_var.get():
                mode_defs.append(("BETA2", "Beta2", "beta2"))
            if mode_sigma_var.get():
                mode_defs.append(("SIGMA_REL", "Sigma", "sigma_rel"))

            if not mode_defs:
                messagebox.showerror(
                    "Merge unavailable",
                    "Select at least one merge mode.",
                    parent=dialog,
                )
                return

            selected_jobs: list[tuple[str, str, list[str], str]] = []
            for preset_name in selected_preset_names:
                preset_args = post_hoc_ema_mode_args_for_preset(preset_name)
                for mode_label, mode_suffix, mode_key in mode_defs:
                    selected_jobs.append((mode_label, mode_suffix, preset_args[mode_key], preset_name))

            choice = (selected_file_paths, selected_jobs)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        go_button = ttk.Button(button_row, text="Go", command=choose_and_close)
        go_button.grid(row=0, column=0)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.bind("<Escape>", lambda _e: cancel_and_close())
        dialog.bind("<Return>", lambda _e: choose_and_close())

        dialog.update_idletasks()
        requested_width = max(390, dialog.winfo_reqwidth())
        requested_height = max(500, dialog.winfo_reqheight())
        dialog.geometry(f"{requested_width}x{requested_height}")
        center_window(dialog)
        dialog.deiconify()
        dialog.focus_set()
        selected_list.focus_set()
        root.wait_window(dialog)
        return choice

    def lora_post_hoc_ema_merge(dataset_name: str) -> None:
        output_dir = runtime_config.training_dir / dataset_name / "output"
        lora_post_hoc_ema_merge_for_output(dataset_name, output_dir)

    def lora_post_hoc_ema_merge_for_output(target_name: str, output_dir: Path, merge_output_dir: Path | None = None) -> None:
        if merge_output_dir is None:
            merge_output_dir = output_dir
        if not output_dir.exists() or not output_dir.is_dir():
            messagebox.showerror(
                "Merge unavailable",
                "No output folder was found for this job.",
                parent=root,
            )
            return

        available = sorted([p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors"])
        if not available:
            messagebox.showerror(
                "Merge unavailable",
                "No .safetensors files were found in this output folder.",
                parent=root,
            )
            return
        module_script = runtime_config.musubi_dir / "src" / "musubi_tuner" / "lora_post_hoc_ema.py"
        root_script = runtime_config.musubi_dir / "lora_post_hoc_ema.py"
        if not module_script.exists() and not root_script.exists():
            messagebox.showerror(
                "Merge unavailable",
                "Could not find lora_post_hoc_ema.py in Musubi-Tuner.",
                parent=root,
            )
            return

        merge_options = ask_lora_merge_options(target_name, available)
        if merge_options is None:
            return
        selected_files, selected_jobs = merge_options

        musubi_python = runtime_config.musubi_python
        if musubi_python is None or not musubi_python.is_file():
            messagebox.showerror(
                "Merge unavailable",
                (
                    "Musubi-Tuner Python was not found in its venv.\n"
                    "Expected: .venv/Scripts/python.exe inside your Musubi-Tuner folder."
                ),
                parent=root,
            )
            return

        run_env = os.environ.copy()
        musubi_src = str(runtime_config.musubi_dir / "src")
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = musubi_src if not existing_pythonpath else f"{musubi_src}{os.pathsep}{existing_pythonpath}"

        log("")
        created_paths: list[Path] = []
        for merge_mode_label, merge_mode_suffix, merge_mode_args, preset_name in selected_jobs:
            output_path = next_merged_output_path(
                target_name,
                merge_output_dir,
                merge_mode_suffix,
                preset_name,
                selected_files,
            )
            command: list[str]
            if module_script.exists():
                command = [
                    str(musubi_python),
                    "-m",
                    "musubi_tuner.lora_post_hoc_ema",
                    *selected_files,
                    "--output_file",
                    str(output_path),
                    *merge_mode_args,
                ]
            else:
                command = [
                    str(musubi_python),
                    str(root_script),
                    *selected_files,
                    "--output_file",
                    str(output_path),
                    *merge_mode_args,
                ]

            log(
                f"[Post-Hoc EMA] Merging {len(selected_files)} checkpoint(s) for '{target_name}' "
                f"using {merge_mode_label} / {preset_name}..."
            )
            result = subprocess.run(
                command,
                cwd=str(runtime_config.musubi_dir),
                env=run_env,
                capture_output=True,
                text=True,
            )

            stdout_text = result.stdout.strip()
            stderr_text = result.stderr.strip()
            if stdout_text:
                log(stdout_text)

            if result.returncode != 0:
                message = stderr_text if stderr_text else "lora_post_hoc_ema.py failed with no error output."
                log(f"[Post-Hoc EMA] Failed ({result.returncode}) while running {merge_mode_label} / {preset_name}.")
                if stderr_text:
                    log(stderr_text)
                messagebox.showerror("Post-Hoc EMA merge failed", message, parent=root)
                return

            log(f"[Post-Hoc EMA] Created ({merge_mode_suffix} / {preset_name}): {output_path}")
            if stderr_text:
                log(stderr_text)
            created_paths.append(output_path)

        if created_paths:
            created_text = "\n".join(str(path) for path in created_paths)
            messagebox.showinfo("Post-Hoc EMA merge complete", f"Created:\n{created_text}", parent=root)

        checkpoint_cache.pop(target_name, None)
        rebuild_folder_list(force=True)

    def open_lora_merge_tool_dialog() -> None:
        musubi_python = runtime_config.musubi_python
        if musubi_python is None or not musubi_python.is_file():
            messagebox.showerror(
                "Merge unavailable",
                "App venv Python was not found. Run Setup.bat first.",
                parent=root,
            )
            return

        module_script = runtime_config.musubi_dir / "src" / "musubi_tuner" / "lora_post_hoc_ema.py"
        root_script = runtime_config.musubi_dir / "lora_post_hoc_ema.py"
        if not module_script.exists() and not root_script.exists():
            messagebox.showerror(
                "Merge unavailable",
                "Could not find lora_post_hoc_ema.py in Musubi-Tuner.",
                parent=root,
            )
            return

        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("LoRA Post-Hoc EMA Merge")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(False, False)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        set_dark_title_bar(dialog)

        candidate_loras: list[Path] = []
        merge_loras: list[Path] = []

        frame = ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        candidate_section = ttk.LabelFrame(frame, text="LoRAs", padding=8)
        candidate_section.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        candidate_section.columnconfigure(0, weight=1)
        candidate_section.rowconfigure(0, weight=1)

        candidate_list = tk.Listbox(
            candidate_section,
            selectmode="extended",
            exportselection=False,
            height=10,
            activestyle="none",
            bg="#1f1f1f",
            fg=fg_text,
            highlightthickness=1,
            highlightbackground=border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
        )
        candidate_list.grid(row=0, column=0, sticky="nsew")
        candidate_scroll = ttk.Scrollbar(
            candidate_section,
            orient="vertical",
            command=candidate_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        candidate_scroll.grid(row=0, column=1, sticky="ns")
        candidate_list.configure(yscrollcommand=candidate_scroll.set)

        merge_section = ttk.LabelFrame(frame, text="Merge order", padding=8)
        merge_section.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        merge_section.columnconfigure(0, weight=1)
        merge_section.rowconfigure(0, weight=1)

        merge_list = tk.Listbox(
            merge_section,
            selectmode="extended",
            exportselection=False,
            height=10,
            activestyle="none",
            bg="#1f1f1f",
            fg=fg_text,
            highlightthickness=1,
            highlightbackground=border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
        )
        merge_list.grid(row=0, column=0, sticky="nsew")
        merge_scroll = ttk.Scrollbar(
            merge_section,
            orient="vertical",
            command=merge_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        merge_scroll.grid(row=0, column=1, sticky="ns")
        merge_list.configure(yscrollcommand=merge_scroll.set)

        mode_section = ttk.LabelFrame(frame, text="Merge options", padding=8)
        mode_section.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        mode_section.columnconfigure(0, weight=1)
        mode_section.columnconfigure(1, weight=1)

        ttk.Label(
            mode_section,
            text="Post-Hoc EMA smooths checkpoints from the same run into one more stable LoRA.",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        mode_beta_var = tk.BooleanVar(value=False)
        mode_beta2_var = tk.BooleanVar(value=False)
        mode_sigma_var = tk.BooleanVar(value=False)
        preset_balanced_var = tk.BooleanVar(value=False)
        preset_smooth_var = tk.BooleanVar(value=False)
        preset_anti_overfit_var = tk.BooleanVar(value=False)

        preset_section = ttk.LabelFrame(mode_section, text="Preset(s)", padding=6)
        preset_section.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(8, 0))
        attach_hover_tooltip(preset_section, merge_preset_tooltip_text)
        ttk.Checkbutton(preset_section, text="Balanced", variable=preset_balanced_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(preset_section, text="Smooth", variable=preset_smooth_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(preset_section, text="Anti-overfit", variable=preset_anti_overfit_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        mode_group = ttk.LabelFrame(mode_section, text="Mode(s)", padding=6)
        mode_group.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(8, 0))
        attach_hover_tooltip(mode_group, merge_mode_tooltip_text)
        ttk.Checkbutton(mode_group, text="BETA (Default)", variable=mode_beta_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(mode_group, text="BETA + BETA2 (Interpolated)", variable=mode_beta2_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(mode_group, text="SIGMA_REL", variable=mode_sigma_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        output_name_var = tk.StringVar(value="merged_lora")
        output_dir_var = tk.StringVar(value="")

        ttk.Label(mode_section, text="Output name:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(mode_section, textvariable=output_name_var, style="Flat.TEntry").grid(row=2, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(mode_section, text="Output folder:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Label(mode_section, textvariable=output_dir_var, style="PathDisplay.TLabel", anchor="w", padding=(6, 4)).grid(
            row=3,
            column=1,
            sticky="ew",
            pady=(8, 0),
        )

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)

        def refresh_candidate_list() -> None:
            candidate_list.delete(0, "end")
            for path in candidate_loras:
                candidate_list.insert("end", path.name)

        def refresh_merge_list() -> None:
            merge_list.delete(0, "end")
            for path in merge_loras:
                merge_list.insert("end", path.name)

        def _normalize_lora_paths(raw_paths: list[str]) -> list[Path]:
            normalized: list[Path] = []
            for raw_path in raw_paths:
                value = raw_path.strip().strip('"')
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.exists() or not path.is_file() or path.suffix.lower() != ".safetensors":
                    continue
                normalized.append(path)
            return normalized

        def add_loras_to_candidate_pool(paths: list[Path]) -> int:
            if not paths:
                return 0
            existing = {str(path.resolve()) for path in candidate_loras}
            added = 0
            for path in paths:
                key = str(path.resolve())
                if key in existing:
                    continue
                candidate_loras.append(path)
                existing.add(key)
                added += 1

            if added > 0:
                candidate_loras.sort(key=lambda p: p.name.lower())
                refresh_candidate_list()
            return added

        def add_paths_to_merge_list(paths: list[Path]) -> int:
            if not paths:
                return 0
            existing = {str(path.resolve()) for path in merge_loras}
            added = 0
            for path in paths:
                key = str(path.resolve())
                if key in existing:
                    continue
                merge_loras.append(path)
                existing.add(key)
                added += 1

            if added > 0:
                if merge_loras and not output_dir_var.get().strip():
                    output_dir_var.set(str(merge_loras[0].parent))
                refresh_merge_list()
            return added

        def add_raw_paths(raw_paths: list[str], to_merge: bool = False) -> int:
            paths = _normalize_lora_paths(raw_paths)
            if not paths:
                return 0
            add_loras_to_candidate_pool(paths)
            return add_paths_to_merge_list(paths) if to_merge else len(paths)

        def add_loras_to_pool() -> None:
            initial_dir = str(runtime_config.training_dir)
            if candidate_loras:
                initial_dir = str(candidate_loras[-1].parent)
            picked = filedialog.askopenfilenames(
                parent=dialog,
                title="Select LoRA files",
                initialdir=initial_dir,
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
            )
            if not picked:
                return

            add_raw_paths([str(p) for p in picked], to_merge=False)

        def add_to_merge_list() -> None:
            raw_selected_indices = list(candidate_list.curselection())
            if not raw_selected_indices:
                return

            selected_indices: list[int] = []
            for raw_index in raw_selected_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(candidate_loras):
                    selected_indices.append(index)

            if not selected_indices:
                return

            selected_paths = [candidate_loras[index] for index in selected_indices]
            add_paths_to_merge_list(selected_paths)

        def on_candidate_double_click(event: tk.Event) -> str:
            # Only treat double-clicks on an actual item row as "add to merge".
            clicked_index = candidate_list.nearest(event.y)
            if clicked_index < 0 or clicked_index >= len(candidate_loras):
                return "break"

            row_bbox = candidate_list.bbox(clicked_index)
            if row_bbox is None:
                return "break"
            _x, y, _w, h = row_bbox
            if not (y <= event.y < y + h):
                return "break"

            candidate_list.selection_clear(0, "end")
            candidate_list.selection_set(clicked_index)
            candidate_list.activate(clicked_index)
            add_to_merge_list()
            return "break"

        def try_enable_file_dnd() -> bool:
            if not tkdnd_available or DND_FILES is None:
                return False

            def process_drop_on_ui_thread(raw_paths: list[str], to_merge: bool, target_name: str) -> None:
                if not dialog.winfo_exists():
                    return

                def _apply_drop() -> None:
                    try:
                        added = add_raw_paths(raw_paths, to_merge=to_merge)
                        if added > 0:
                            destination = "merge list" if to_merge else "pool"
                            log(f"[LoRA Post-Hoc EMA Merge] Added {added} dropped LoRA file(s) to {destination}.")
                    except Exception as exc:
                        log(f"[LoRA Post-Hoc EMA Merge] Drop handling failed on {target_name}: {exc}")

                dialog.after(0, _apply_drop)

            def decode_dropped_paths(event_data: str) -> list[str]:
                if not event_data:
                    return []
                try:
                    split_values = dialog.tk.splitlist(event_data)
                except Exception:
                    split_values = [event_data]
                return [str(item) for item in split_values if str(item).strip()]

            def on_drop_to_pool(event: tk.Event) -> str:
                raw_paths = decode_dropped_paths(str(getattr(event, "data", "")))
                process_drop_on_ui_thread(raw_paths, to_merge=False, target_name="LoRAs")
                return "break"

            def on_drop_to_merge(event: tk.Event) -> str:
                raw_paths = decode_dropped_paths(str(getattr(event, "data", "")))
                process_drop_on_ui_thread(raw_paths, to_merge=True, target_name="Merge order")
                return "break"

            try:
                candidate_list.drop_target_register(DND_FILES)
                candidate_list.dnd_bind("<<Drop>>", on_drop_to_pool)
                merge_list.drop_target_register(DND_FILES)
                merge_list.dnd_bind("<<Drop>>", on_drop_to_merge)
                return True
            except Exception:
                return False

        def remove_from_merge_list() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices:
                return
            for index in reversed(selected_indices):
                merge_loras.pop(index)
            refresh_merge_list()

        def move_merge_up() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices or selected_indices[0] == 0:
                return
            for index in selected_indices:
                merge_loras[index - 1], merge_loras[index] = merge_loras[index], merge_loras[index - 1]
            refresh_merge_list()
            for index in [i - 1 for i in selected_indices]:
                merge_list.selection_set(index)

        def move_merge_down() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices or selected_indices[-1] >= len(merge_loras) - 1:
                return
            for index in reversed(selected_indices):
                merge_loras[index + 1], merge_loras[index] = merge_loras[index], merge_loras[index + 1]
            refresh_merge_list()
            for index in [i + 1 for i in selected_indices]:
                merge_list.selection_set(index)

        def clear_merge_list() -> None:
            merge_loras.clear()
            refresh_merge_list()

        def browse_output_folder() -> None:
            picked = filedialog.askdirectory(parent=dialog, title="Select output folder")
            if picked:
                output_dir_var.set(picked)

        def resolve_output_base() -> tuple[Path, str] | None:
            output_name = output_name_var.get().strip()
            if not output_name:
                messagebox.showerror("Missing value", "Output name is required.", parent=dialog)
                return None

            output_folder_raw = output_dir_var.get().strip()
            if not output_folder_raw:
                messagebox.showerror("Missing value", "Output folder is required.", parent=dialog)
                return None

            output_folder = Path(output_folder_raw).expanduser()
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Invalid folder", f"Could not use output folder:\n{exc}", parent=dialog)
                return None

            output_name = output_name[:-12] if output_name.lower().endswith(".safetensors") else output_name
            return output_folder, output_name

        def build_output_path(output_folder: Path, output_name: str, mode_suffix: str, preset_name: str) -> Path:
            preset_token = merge_preset_file_token(preset_name)
            return output_folder / f"{output_name}_{mode_suffix}_{preset_token}.safetensors"

        def run_merge() -> None:
            if len(merge_loras) < 2:
                messagebox.showerror("Merge unavailable", "Add at least 2 LoRAs to merge list.", parent=dialog)
                return

            selected_preset_names: list[str] = []
            if preset_balanced_var.get():
                selected_preset_names.append("Balanced")
            if preset_smooth_var.get():
                selected_preset_names.append("Smooth")
            if preset_anti_overfit_var.get():
                selected_preset_names.append("Anti-overfit")
            if not selected_preset_names:
                messagebox.showerror("Merge unavailable", "Select at least one preset.", parent=dialog)
                return

            mode_defs: list[tuple[str, str, str]] = []
            if mode_beta_var.get():
                mode_defs.append(("BETA", "Beta", "beta"))
            if mode_beta2_var.get():
                mode_defs.append(("BETA2", "Beta2", "beta2"))
            if mode_sigma_var.get():
                mode_defs.append(("SIGMA_REL", "Sigma", "sigma_rel"))
            if not mode_defs:
                messagebox.showerror("Merge unavailable", "Select at least one merge mode.", parent=dialog)
                return

            output_base = resolve_output_base()
            if output_base is None:
                return
            output_folder, output_name = output_base

            selected_jobs: list[tuple[str, str, list[str], str]] = []
            for preset_name in selected_preset_names:
                preset_args = post_hoc_ema_mode_args_for_preset(preset_name)
                for mode_label, mode_suffix, mode_key in mode_defs:
                    selected_jobs.append((mode_label, mode_suffix, preset_args[mode_key], preset_name))

            existing_outputs: list[Path] = []
            for _merge_mode_label, merge_mode_suffix, _merge_mode_args, preset_name in selected_jobs:
                candidate = build_output_path(output_folder, output_name, merge_mode_suffix, preset_name)
                if candidate.exists():
                    existing_outputs.append(candidate)
            if existing_outputs:
                existing_text = "\n".join(str(path) for path in existing_outputs)
                messagebox.showerror(
                    "Name already exists",
                    f"One or more output files already exist:\n{existing_text}\n\nChoose a different output name.",
                    parent=dialog,
                )
                return

            selected_files = [str(path) for path in merge_loras]

            run_env = os.environ.copy()
            musubi_src = str(runtime_config.musubi_dir / "src")
            existing_pythonpath = run_env.get("PYTHONPATH", "")
            run_env["PYTHONPATH"] = musubi_src if not existing_pythonpath else f"{musubi_src}{os.pathsep}{existing_pythonpath}"

            log("")
            created_paths: list[Path] = []
            for merge_mode_label, merge_mode_suffix, merge_mode_args, preset_name in selected_jobs:
                output_path = build_output_path(output_folder, output_name, merge_mode_suffix, preset_name)
                command: list[str]
                if module_script.exists():
                    command = [
                        str(musubi_python),
                        "-m",
                        "musubi_tuner.lora_post_hoc_ema",
                        *selected_files,
                        "--output_file",
                        str(output_path),
                        "--no_sort",
                        *merge_mode_args,
                    ]
                else:
                    command = [
                        str(musubi_python),
                        str(root_script),
                        *selected_files,
                        "--output_file",
                        str(output_path),
                        "--no_sort",
                        *merge_mode_args,
                    ]

                log(
                    f"[LoRA Post-Hoc EMA Merge] Merging {len(selected_files)} LoRA(s) "
                    f"using {merge_mode_label} / {preset_name}..."
                )
                log(f"[LoRA Post-Hoc EMA Merge] Output: {output_path}")

                result = subprocess.run(
                    command,
                    cwd=str(runtime_config.musubi_dir),
                    env=run_env,
                    capture_output=True,
                    text=True,
                )

                if result.stdout.strip():
                    log(result.stdout.strip())

                if result.returncode != 0:
                    log(
                        f"[LoRA Post-Hoc EMA Merge] Failed ({result.returncode}) "
                        f"while running {merge_mode_label} / {preset_name}."
                    )
                    if result.stderr.strip():
                        log(result.stderr.strip())
                    messagebox.showerror(
                        "Merge failed",
                        result.stderr.strip() or "lora_post_hoc_ema.py failed with no error output.",
                        parent=dialog,
                    )
                    return

                if result.stderr.strip():
                    log(result.stderr.strip())
                log(f"[LoRA Post-Hoc EMA Merge] Created ({merge_mode_suffix} / {preset_name}): {output_path}")
                created_paths.append(output_path)

            created_text = "\n".join(str(path) for path in created_paths)
            messagebox.showinfo("Merge complete", f"Created:\n{created_text}", parent=dialog)

        candidate_list.bind("<Double-Button-1>", on_candidate_double_click)

        candidate_actions = ttk.Frame(candidate_section)
        candidate_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(candidate_actions, text="Add LoRA", command=add_loras_to_pool).grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Button(candidate_actions, text="Add to Merge List >>", command=add_to_merge_list).grid(row=0, column=1, sticky="w")

        dnd_enabled = try_enable_file_dnd()
        dnd_hint = (
            "Tip: Drag .safetensors files onto LoRAs or Merge order lists."
            if dnd_enabled
            else "Tip: Drag-and-drop requires tkinterdnd2 in app Python (pip install tkinterdnd2). Use Add LoRA for now."
        )
        ttk.Label(candidate_section, text=dnd_hint, style="Dim.TLabel", wraplength=320).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        merge_actions = ttk.Frame(merge_section)
        merge_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(merge_actions, text="Remove", command=remove_from_merge_list).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(merge_actions, text="Up", command=move_merge_up).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(merge_actions, text="Down", command=move_merge_down).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(merge_actions, text="Clear", command=clear_merge_list).grid(row=0, column=3)

        ttk.Button(mode_section, text="Browse", command=browse_output_folder).grid(row=6, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Button(actions, text="Close", command=dialog.destroy).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Go", command=run_merge).grid(row=0, column=2)

        dialog.geometry("820x620")
        center_window(dialog)
        dialog.deiconify()
        root.wait_window(dialog)

    def archive_root_dir() -> Path:
        return datasets_root_dir().parent / "Archive"

    def archive_dataset(dataset_name: str) -> None:
        src = dataset_dir_path(dataset_name)
        if not src.exists() or not src.is_dir():
            messagebox.showerror("Archive failed", f"Dataset folder not found:\n{src}", parent=root)
            return
        dest_parent = archive_root_dir() / "Datasets"
        dest = dest_parent / dataset_name
        if not messagebox.askyesno(
            "Archive dataset",
            f"Move dataset '{dataset_name}' to Archive/Datasets?\n\nThe folder will be moved out of the Datasets directory.",
            parent=root,
        ):
            return
        try:
            dest_parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Merge: copy every file from src into dest, overwriting same-named files.
                # Files only in dest are kept untouched.
                for src_file in src.iterdir():
                    if src_file.is_file():
                        dst_file = dest / src_file.name
                        shutil.copy2(str(src_file), str(dst_file))
                shutil.rmtree(src)
            else:
                shutil.move(str(src), str(dest))
        except OSError as exc:
            messagebox.showerror("Archive failed", f"Could not archive dataset folder:\n{exc}", parent=root)
            return
        rebuild_folder_list(force=True)
        log(f"[Archive] Dataset '{dataset_name}' archived to: {dest}")

    def archive_selected_datasets() -> None:
        names = selected_dataset_names()
        if not names:
            messagebox.showinfo(
                "Archive Datasets",
                "No datasets are selected. Check the datasets you want to archive first.",
                parent=root,
            )
            return
        label = "\n".join(f"  • {n}" for n in names)
        if not messagebox.askyesno(
            "Archive Datasets",
            f"Move {len(names)} dataset(s) to Archive/Datasets?\n\n{label}\n\nThe folders will be moved out of the Datasets directory.",
            parent=root,
        ):
            return
        dest_parent = archive_root_dir() / "Datasets"
        errors: list[str] = []
        for name in names:
            src = dataset_dir_path(name)
            if not src.exists() or not src.is_dir():
                errors.append(f"{name}: folder not found")
                continue
            dest = dest_parent / name
            try:
                dest_parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    for src_file in src.iterdir():
                        if src_file.is_file():
                            shutil.copy2(str(src_file), str(dest / src_file.name))
                    shutil.rmtree(src)
                else:
                    shutil.move(str(src), str(dest))
                log(f"[Archive] Dataset '{name}' archived to: {dest}")
            except OSError as exc:
                errors.append(f"{name}: {exc}")
        rebuild_folder_list(force=True)
        if errors:
            messagebox.showerror(
                "Archive Datasets",
                f"Some datasets could not be archived:\n" + "\n".join(errors),
                parent=root,
            )

    def open_restore_datasets_dialog() -> None:
        archive_datasets_dir = archive_root_dir() / "Datasets"
        if not archive_datasets_dir.exists() or not archive_datasets_dir.is_dir():
            messagebox.showinfo("Restore Datasets", "No archived datasets found.", parent=root)
            return

        archived_names = sorted(
            child.name for child in archive_datasets_dir.iterdir() if child.is_dir()
        )
        if not archived_names:
            messagebox.showinfo("Restore Datasets", "No archived datasets found.", parent=root)
            return

        THUMB_PX = 120
        COLS = 4
        selected_vars: dict[str, tk.BooleanVar] = {}
        thumb_refs: list[ImageTk.PhotoImage] = []

        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("Restore Archived Datasets")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(True, True)
        set_dark_title_bar(dialog)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=0)

        # ── scrollable grid host ──────────────────────────────────────────
        scroll_host = ttk.Frame(dialog)
        scroll_host.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_host, bg=bg_panel, bd=0, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(scroll_host, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vscroll.set)

        inner = ttk.Frame(canvas, padding=4)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_scroll(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event: tk.Event) -> None:
            canvas.itemconfigure(inner_id, width=event.width)

        inner.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _sync_width)

        def _on_mousewheel(event: tk.Event) -> str:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        dialog.bind("<MouseWheel>", _on_mousewheel)

        # ── build thumbnail grid ──────────────────────────────────────────
        for idx_n, name in enumerate(archived_names):
            col = idx_n % COLS
            row = idx_n // COLS

            archive_folder = archive_datasets_dir / name
            images = sorted(
                p for p in archive_folder.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ) if archive_folder.is_dir() else []
            first_img = images[0] if images else None

            # build thumbnail
            try:
                if first_img:
                    img = Image.open(first_img).convert("RGB")
                    img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
                    bg_img = Image.new("RGB", (THUMB_PX, THUMB_PX), (45, 45, 45))
                    offset = ((THUMB_PX - img.width) // 2, (THUMB_PX - img.height) // 2)
                    bg_img.paste(img, offset)
                    photo = ImageTk.PhotoImage(bg_img, master=root)
                else:
                    placeholder = Image.new("RGB", (THUMB_PX, THUMB_PX), (45, 45, 45))
                    photo = ImageTk.PhotoImage(placeholder, master=root)
            except Exception:
                placeholder = Image.new("RGB", (THUMB_PX, THUMB_PX), (45, 45, 45))
                photo = ImageTk.PhotoImage(placeholder, master=root)
            thumb_refs.append(photo)

            var = tk.BooleanVar(value=False)
            selected_vars[name] = var

            card = tk.Frame(inner, bg=bg_card, bd=1, relief="solid", cursor="hand2")
            card.grid(row=row, column=col, padx=4, pady=4, sticky="n")

            thumb_label = tk.Label(card, image=photo, bg=bg_card, cursor="hand2")
            thumb_label.pack(padx=4, pady=(4, 2))

            name_label = tk.Label(
                card, text=name, fg=fg_text, bg=bg_card,
                font=("Segoe UI", 8), wraplength=THUMB_PX + 8, justify="center", cursor="hand2",
            )
            name_label.pack(padx=4, pady=(0, 2))

            count_text = f"{len(images)} image{'s' if len(images) != 1 else ''}"
            tk.Label(
                card, text=count_text, fg=fg_muted, bg=bg_card,
                font=("Segoe UI", 7),
            ).pack(padx=4, pady=(0, 4))

            def _toggle(n: str = name, c: tk.Frame = card) -> None:
                v = selected_vars[n]
                v.set(not v.get())
                c.configure(bg="#1e3a5a" if v.get() else bg_card)
                for w in c.winfo_children():
                    try:
                        w.configure(bg="#1e3a5a" if v.get() else bg_card)
                    except tk.TclError:
                        pass

            for widget in (card, thumb_label, name_label):
                widget.bind("<Button-1>", lambda _e, n=name, c=card: _toggle(n, c))

        # ── footer ────────────────────────────────────────────────────────
        footer = ttk.Frame(dialog, padding=(10, 8, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        sel_label_var = tk.StringVar(value="0 selected")

        def _update_sel_label(*_: object) -> None:
            count = sum(1 for v in selected_vars.values() if v.get())
            sel_label_var.set(f"{count} selected")

        for v in selected_vars.values():
            v.trace_add("write", _update_sel_label)

        ttk.Label(footer, textvariable=sel_label_var, foreground=fg_muted).grid(row=0, column=0, sticky="w")

        def _apply() -> None:
            to_restore = [n for n, v in selected_vars.items() if v.get()]
            if not to_restore:
                messagebox.showwarning("No selection", "Select at least one dataset to restore.", parent=dialog)
                return

            errors: list[str] = []
            restored: list[str] = []
            for name in to_restore:
                src = archive_datasets_dir / name
                dst = datasets_root_dir() / name
                if dst.exists():
                    errors.append(f"'{name}' — destination already exists, skipped.")
                    continue
                try:
                    datasets_root_dir().mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    restored.append(name)
                except OSError as exc:
                    errors.append(f"'{name}' — {exc}")

            if errors:
                messagebox.showwarning(
                    "Restore completed with errors",
                    "Some datasets could not be restored:\n\n" + "\n".join(errors),
                    parent=dialog,
                )
            dialog.destroy()
            rebuild_folder_list(force=True)
            if restored:
                log(f"[Restore] Restored {len(restored)} dataset(s): {', '.join(restored)}")

        btn_row = ttk.Frame(footer)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btn_row, text="Restore Selected", command=_apply).grid(row=0, column=1)

        dialog.update_idletasks()
        cols_w = COLS * (THUMB_PX + 24) + 44
        win_w = max(cols_w, 420)
        win_h = min(700, max(380, (len(archived_names) // COLS + 1) * (THUMB_PX + 60) + 120))
        dialog.geometry(f"{win_w}x{win_h}")
        center_window(dialog)
        dialog.deiconify()
        root.wait_window(dialog)

    def archive_job(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        job = job_queue[idx]
        job_name = job.get("job_name", "unnamed")
        training_name = job.get("training_name", "").strip() or job_name
        training_dir = Path(job.get("training_dir", "")).expanduser()
        if not training_dir.exists() and training_name:
            training_dir = training_job_dir_path(training_name).expanduser()
        dest_parent = archive_root_dir() / "Jobs"
        dest = dest_parent / training_dir.name
        overwrite = False
        if dest.exists():
            overwrite = messagebox.askyesno(
                "Archive job",
                f"An archived job named '{training_dir.name}' already exists.\n\nOverwrite it?",
                parent=root,
            )
            if not overwrite:
                return
        if not messagebox.askyesno(
            "Archive job",
            f"Move job '{job_name}' to Archive/Jobs?\n\nThe job will be removed from the queue and its folder moved to the archive.",
            parent=root,
        ):
            return
        removed = job_queue.pop(idx)
        remove_job_from_disk(removed)
        save_job_order()
        if training_dir.exists() and training_dir.is_dir():
            try:
                dest_parent.mkdir(parents=True, exist_ok=True)
                if overwrite and dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(training_dir), str(dest))
            except OSError as exc:
                messagebox.showerror("Archive failed", f"Could not move job folder:\n{exc}", parent=root)
        refresh_job_queue_list()
        update_start_button_state()
        log(f"[Archive] Job '{job_name}' archived to: {dest}")

    def show_thumbnail_context_menu(event: tk.Event, dataset_name: str) -> str:
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Edit Dataset", command=lambda: open_edit_dataset_dialog(dataset_name))
        menu.add_command(label="Open Dataset", command=lambda: open_dataset_in_file_manager(dataset_name))
        menu.add_command(label="Add Images", command=lambda: add_images_to_dataset(dataset_name))
        menu.add_separator()
        menu.add_command(label="Archive Dataset", command=lambda: archive_dataset(dataset_name))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    paned = ttk.PanedWindow(root, orient="vertical")
    paned.grid(row=2, column=0, sticky="nsew", padx=8)
    paned.bind("<Configure>", schedule_main_window_position_save)
    paned.bind("<ButtonRelease-1>", schedule_main_window_position_save)

    top_pane = ttk.Frame(paned)
    top_pane.columnconfigure(0, weight=1)
    top_pane.rowconfigure(0, weight=1)
    top_pane.rowconfigure(1, weight=0)
    paned.add(top_pane, weight=3)

    list_container = ttk.LabelFrame(top_pane, text="Datasets (click thumbnail to select)", padding=8)
    list_container.grid(row=0, column=0, sticky="nsew")
    list_container.columnconfigure(0, weight=1)
    list_container.columnconfigure(1, weight=0, minsize=12)
    list_container.rowconfigure(0, weight=1)

    canvas = tk.Canvas(list_container, highlightthickness=0, bg=bg_panel)
    scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview, style="Dataset.Vertical.TScrollbar")
    inner = ttk.Frame(canvas, style="TFrame")

    def update_scrollbar_visibility() -> None:
        bbox = canvas.bbox("all")
        content_height = 0 if bbox is None else max(0, bbox[3] - bbox[1])
        viewport_height = max(0, canvas.winfo_height())
        should_show = content_height > (viewport_height + 1)
        if should_show:
            scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            scrollbar.grid_remove()

    def on_inner_configure(_event: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        update_scrollbar_visibility()

    inner.bind("<Configure>", on_inner_configure)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    dataset_actions_bar = ttk.Frame(top_pane, padding=(0, 8, 0, 10))
    dataset_actions_bar.grid(row=1, column=0, sticky="ew")
    dataset_actions_bar.columnconfigure(0, weight=1)

    bottom_pane = ttk.Frame(paned)
    bottom_pane.columnconfigure(0, weight=1)
    bottom_pane.rowconfigure(0, weight=1)
    paned.add(bottom_pane, weight=2)

    start_bar = ttk.Frame(bottom_pane, padding=(0, 0, 0, 8))
    start_bar.grid(row=1, column=0, sticky="ew")
    start_bar.columnconfigure(0, weight=1)

    queue_container = ttk.LabelFrame(bottom_pane, text="", padding=8)
    queue_container.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
    queue_container.columnconfigure(0, weight=1)
    queue_container.rowconfigure(1, weight=1)

    queue_header = ttk.Frame(queue_container)
    queue_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    queue_header.columnconfigure(0, weight=1)
    ttk.Label(queue_header, text="Queue:", style="TLabel").grid(row=0, column=0, sticky="w")
    ttk.Button(
        queue_header,
        text="Reload",
        style="QueueAction.TButton",
        command=lambda: (
            load_job_queue_from_disk(),
            refresh_job_queue_list(),
            sync_queue_row_action_buttons(),
            update_start_button_state(),
        ),
    ).grid(row=0, column=1, sticky="e")

    queue_table_border = tk.Frame(queue_container, bg="#2a4a72", bd=0, highlightthickness=0)
    queue_table_border.grid(row=1, column=0, columnspan=2, sticky="nsew")
    queue_table_border.columnconfigure(0, weight=1)
    queue_table_border.rowconfigure(0, weight=1)

    queue_list = ttk.Treeview(
        queue_table_border,
        columns=("run", "thumb", "name", "source", "status", "actions"),
        show="tree headings",
        selectmode="browse",
        height=6,
        style="Queue.Treeview",
    )
    queue_list.heading("#0", text="", anchor="center")
    queue_list.heading("run", text="", anchor="center")
    queue_list.heading("thumb", text="", anchor="center")
    queue_list.heading("name", text="   LoRA Name", anchor="w")
    queue_list.heading("source", text="   Source Dataset", anchor="w")
    queue_list.heading("status", text="Status", anchor="center")
    queue_list.heading("actions", text="", anchor="center")
    queue_list.column("#0", width=28, minwidth=26, stretch=False, anchor="center")
    queue_list.column("run", width=40, minwidth=38, stretch=False, anchor="center")
    queue_list.column("thumb", width=76, minwidth=68, stretch=False, anchor="center")
    queue_list.column("name", width=156, minwidth=120, stretch=True, anchor="w")
    queue_list.column("source", width=118, minwidth=90, stretch=True, anchor="w")
    queue_list.column("status", width=96, minwidth=82, stretch=False, anchor="center")
    queue_list.column("actions", width=36, minwidth=32, stretch=False, anchor="center")
    queue_list.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
    queue_list.tag_configure("row_even", background="#1c2534")
    queue_list.tag_configure("row_odd", background="#17202e")
    queue_list.tag_configure("row_running", background="#163326", foreground="#a7f3cc")
    queue_list.tag_configure("row_even_disabled", background="#191e28", foreground="#5a6474")
    queue_list.tag_configure("row_odd_disabled", background="#141923", foreground="#5a6474")
    queue_list.tag_configure("status_done", foreground="#4ade80")
    queue_list.tag_configure("status_failed", foreground="#f87171")
    queue_list.tag_configure("status_running", foreground="#a7f3cc")
    queue_list.tag_configure("status_resume", foreground="#fbbf24")
    queue_list.tag_configure("status_paused", foreground="#fb923c")

    queue_scroll = ttk.Scrollbar(
        queue_table_border,
        orient="vertical",
        command=queue_list.yview,
        style="Dark.Vertical.TScrollbar",
    )
    queue_scroll.grid(row=0, column=1, sticky="ns", pady=1, padx=(0, 1))
    queue_list.configure(yscrollcommand=queue_scroll.set)

    log_container = ttk.Frame(root, height=150)
    log_container.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
    log_container.grid_propagate(False)
    log_container.columnconfigure(0, weight=1)
    log_container.rowconfigure(0, weight=1)

    log_box = tk.Text(log_container, height=10, wrap="none")
    log_box.grid(row=0, column=0, sticky="nsew")
    log_scroll_y = ttk.Scrollbar(
        log_container,
        orient="vertical",
        command=log_box.yview,
        style="Dark.Vertical.TScrollbar",
    )
    log_scroll_y.grid(row=0, column=1, sticky="ns")
    log_scroll_x = ttk.Scrollbar(
        log_container,
        orient="horizontal",
        command=log_box.xview,
        style="Dark.Horizontal.TScrollbar",
    )
    log_scroll_x.grid(row=1, column=0, sticky="ew")
    log_box.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set)
    log_box.configure(bg="#0e1319", fg=fg_text, insertbackground=fg_text, relief="flat", borderwidth=0)
    log_progress_active = False
    log_progress_mark = "log_progress_line_start"

    def _show_log_context_menu(event: "tk.Event[tk.Text]") -> None:
        menu = tk.Menu(
            root, tearoff=0,
            bg=bg_card, fg=fg_text,
            activebackground="#3a3f4b", activeforeground=fg_text,
            bd=0, relief="flat",
        )
        menu.add_command(label="Clear console", command=lambda: log_box.delete("1.0", "end"))
        menu.tk_popup(event.x_root, event.y_root)

    log_box.bind("<Button-3>", _show_log_context_menu)

    def is_log_scrolled_to_bottom() -> bool:
        _first, last = log_box.yview()
        return last >= 0.999

    def log(message: str) -> None:
        def append_line() -> None:
            nonlocal log_progress_active
            if not root.winfo_exists():
                return

            at_bottom = is_log_scrolled_to_bottom()
            if message.startswith("\r"):
                progress_text = message[1:]
                if not log_progress_active:
                    log_box.mark_set(log_progress_mark, "end")
                    log_box.mark_gravity(log_progress_mark, tk.LEFT)
                    log_box.insert("end", progress_text + "\n")
                    log_progress_active = True
                else:
                    log_box.delete(log_progress_mark, f"{log_progress_mark} lineend+1c")
                    log_box.insert(log_progress_mark, progress_text + "\n")
            else:
                log_progress_active = False
                log_box.insert("end", message + "\n")

            if at_bottom:
                log_box.see("end")
            root.update_idletasks()

        if threading.current_thread() is threading.main_thread():
            append_line()
        else:
            root.after(0, append_line)

    def bool_to_flag(value: bool) -> str:
        return "1" if value else "0"

    def flag_to_bool(value: str, default: bool = False) -> bool:
        return is_truthy(value, default=default)

    def jobs_storage_dir() -> Path:
        return runtime_config.training_dir

    def jobs_order_file_path() -> Path:
        return jobs_storage_dir() / JOBS_ORDER_FILE_NAME

    def presets_storage_dir() -> Path:
        return download_workspace_root() / "Presets"

    def _slugify_preset_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        token = token.strip("-._")
        return token or "preset"

    def job_preset_file_path(family_name: str, preset_name: str) -> Path:
        family_token = _slugify_preset_token(family_name)
        preset_token = _slugify_preset_token(preset_name)
        return presets_storage_dir() / f"{family_token}__{preset_token}{JOB_PRESET_FILE_SUFFIX}"

    def job_settings_file_path(training_name: str) -> Path:
        return training_job_dir_path(training_name) / JOB_SETTINGS_FILE_NAME

    def ensure_jobs_storage() -> None:
        jobs_storage_dir().mkdir(parents=True, exist_ok=True)

    def save_job_order() -> None:
        ensure_jobs_storage()
        order = [job.get("training_name", "").strip() for job in job_queue if job.get("training_name", "").strip()]
        jobs_order_file_path().write_text(json.dumps(order, indent=2), encoding="utf-8")

    def load_job_presets_from_disk() -> dict[str, dict[str, object]]:
        cleaned: dict[str, dict[str, object]] = {}

        def _ingest_record(record: dict[str, object], source_path: Path) -> None:
            model_name = str(record.get("model", "")).strip()
            family_name = str(record.get("family", "")).strip()
            preset_name = str(record.get("name", "")).strip()
            values = record.get("values", {})
            if not isinstance(values, dict):
                values = {}
            if not preset_name or (not model_name and not family_name):
                return
            key_scope = family_name or model_name
            key = f"{key_scope}::{preset_name}"
            cleaned[key] = {
                "model": model_name,
                "family": family_name,
                "name": preset_name,
                "values": {str(k): str(v) for k, v in values.items() if isinstance(k, str)},
                "file": str(source_path),
            }

        presets_dir = presets_storage_dir()
        if presets_dir.exists() and presets_dir.is_dir():
            for path in sorted(presets_dir.glob(f"*{JOB_PRESET_FILE_SUFFIX}"), key=lambda p: p.name.casefold()):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(raw, dict):
                    _ingest_record(raw, path)

        return cleaned

    def save_job_preset_to_disk(model_name: str, family_name: str, preset_name: str, values: dict[str, str]) -> Path:
        presets_storage_dir().mkdir(parents=True, exist_ok=True)
        target = job_preset_file_path(family_name, preset_name)
        payload = {
            "model": model_name,
            "family": family_name,
            "name": preset_name,
            "values": values,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def save_job_to_disk(job: dict[str, str]) -> None:
        ensure_jobs_storage()
        training_name = job.get("training_name", "").strip() or job.get("job_name", "").strip()
        if not training_name:
            return
        job["id"] = training_name
        job["training_name"] = training_name
        job["training_dir"] = str(training_job_dir_path(training_name))
        job["output_dir"] = str((training_job_dir_path(training_name) / "output").expanduser())
        settings_path = job_settings_file_path(training_name)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    def remove_job_from_disk(job: dict[str, str]) -> None:
        try:
            training_name = job.get("training_name", "").strip() or job.get("job_name", "").strip()
            if not training_name:
                return
            path = job_settings_file_path(training_name)
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def existing_job_names() -> set[str]:
        return {job.get("job_name", "").strip().lower() for job in job_queue if job.get("job_name", "").strip()}

    def discovered_training_names() -> set[str]:
        discovered: set[str] = set()
        root = jobs_storage_dir()
        if not root.exists() or not root.is_dir():
            return discovered
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name.lower() in {"logs", "__pycache__"}:
                continue
            if (
                (child / "dataset.toml").exists()
                or (child / "output").is_dir()
                or (child / "cache").is_dir()
                or (child / "logs").is_dir()
            ):
                discovered.add(child.name.strip().lower())
        return discovered

    def unique_job_name(base_name: str) -> str:
        base = base_name.strip() or "job"
        existing = existing_job_names() | discovered_training_names()
        if base.lower() not in existing:
            return base
        suffix = 2
        while True:
            candidate = f"{base}_{suffix}"
            if candidate.lower() not in existing:
                return candidate
            suffix += 1

    def job_output_names(job: dict[str, str]) -> set[str]:
        names = {
            job.get("job_name", "").strip(),
            job.get("training_name", "").strip(),
        }
        return {name for name in names if name}

    def job_resume_progress(job: dict[str, str]) -> tuple[int, bool]:
        def _recorded_completed_step() -> int:
            training_dir_raw = job.get("training_dir", "").strip()
            if training_dir_raw:
                metadata_path = Path(training_dir_raw).expanduser() / JOB_PROGRESS_FILE_NAME
            else:
                metadata_path = Path(job.get("output_dir", "")).expanduser().parent / JOB_PROGRESS_FILE_NAME

            if not metadata_path.exists() or not metadata_path.is_file():
                return 0

            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                return 0

            if not isinstance(payload, dict):
                return 0

            raw_value = payload.get("completed_step", 0)
            try:
                completed_step = int(raw_value)
            except (TypeError, ValueError):
                return 0

            return completed_step if completed_step > 0 else 0

        output_dir = Path(job.get("output_dir", "")).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            recorded_step = _recorded_completed_step()
            return recorded_step, recorded_step > 0

        output_names = job_output_names(job)
        if not output_names:
            recorded_step = _recorded_completed_step()
            return recorded_step, recorded_step > 0

        latest_step = _recorded_completed_step()
        has_resume_state = latest_step > 0
        ckpt_patterns = [
            re.compile(rf"^{re.escape(name)}(?:_Klein)?(?:-resume)?-step(\d+)\.safetensors$", re.IGNORECASE)
            for name in output_names
        ]
        state_patterns = [
            re.compile(rf"^{re.escape(name)}(?:_Klein)?(?:-resume)?-step(\d+)-state$", re.IGNORECASE)
            for name in output_names
        ]
        loose_state_names = {
            f"{name}-state".lower() for name in output_names
        } | {
            f"{name}_Klein-state".lower() for name in output_names
        } | {
            f"{name}-resume-state".lower() for name in output_names
        } | {
            f"{name}_Klein-resume-state".lower() for name in output_names
        }

        for path in output_dir.iterdir():
            path_name = path.name
            path_name_lower = path_name.lower()

            if path.is_file() and path.suffix.lower() == ".safetensors":
                for pattern in ckpt_patterns:
                    match = pattern.match(path_name)
                    if match is None:
                        continue
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
                    has_resume_state = True
                    break

            if not path.is_dir():
                continue

            for pattern in state_patterns:
                match = pattern.match(path_name)
                if match is None:
                    continue
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                has_resume_state = True
                break

            if path_name_lower in loose_state_names and (path / "scheduler.bin").exists():
                has_resume_state = True

        return latest_step, has_resume_state

    def _element_root_base_from_name(path: Path) -> str | None:
        name = path.name
        match = re.match(r"^(?P<base>.+?)(?:-resume)?-step\d{8}(?:-state)?(?:\.safetensors)?$", name, re.IGNORECASE)
        if match is not None:
            return match.group("base")

        if path.is_file() and path.suffix.lower() == ".safetensors":
            match = re.match(r"^(?P<base>.+?)(?:-resume)?\.safetensors$", name, re.IGNORECASE)
            if match is not None:
                return match.group("base")

        if path.is_dir():
            match = re.match(r"^(?P<base>.+?)(?:-resume)?-state$", name, re.IGNORECASE)
            if match is not None:
                return match.group("base")

        return None

    def detect_job_element_base_mismatch(job: dict[str, str]) -> tuple[bool, str | None]:
        output_dir = Path(job.get("output_dir", "")).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            return False, None

        expected_base = (job.get("training_name", "").strip() or job.get("job_name", "").strip())
        if not expected_base:
            return False, None

        base_counts: dict[str, int] = {}
        expected_count = 0
        for item in output_dir.iterdir():
            base = _element_root_base_from_name(item)
            if base is None:
                continue
            base_counts[base] = base_counts.get(base, 0) + 1
            if base.lower() == expected_base.lower():
                expected_count += 1

        if not base_counts:
            return False, None

        # If expected training base artifacts exist, treat extra files (for example merged outputs)
        # as non-breaking and keep this job healthy.
        if expected_count > 0:
            return False, None

        non_target_bases = [(base, count) for base, count in base_counts.items() if base.lower() != expected_base.lower()]
        if not non_target_bases:
            return False, None

        source_base = max(non_target_bases, key=lambda pair: pair[1])[0]
        return True, source_base

    def rename_job_elements_to_training_name(job: dict[str, str], source_base: str) -> tuple[int, int]:
        output_dir = Path(job.get("output_dir", "")).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            return 0, 0

        target_base = (job.get("training_name", "").strip() or job.get("job_name", "").strip())
        if not target_base:
            return 0, 0

        renamed = 0
        conflicts = 0

        for item in sorted(output_dir.iterdir(), key=lambda p: p.name.lower()):
            old_name = item.name
            new_name = old_name

            replacements = [
                (f"{source_base}-resume-step", f"{target_base}-resume-step"),
                (f"{source_base}-step", f"{target_base}-step"),
                (f"{source_base}-resume-state", f"{target_base}-resume-state"),
                (f"{source_base}-state", f"{target_base}-state"),
                (f"{source_base}-resume.safetensors", f"{target_base}-resume.safetensors"),
                (f"{source_base}.safetensors", f"{target_base}.safetensors"),
            ]

            for old_prefix, new_prefix in replacements:
                if old_name.startswith(old_prefix):
                    new_name = new_prefix + old_name[len(old_prefix):]
                    break

            if new_name == old_name:
                continue

            target_path = output_dir / new_name
            if target_path.exists():
                conflicts += 1
                continue

            try:
                item.rename(target_path)
                renamed += 1
            except OSError:
                conflicts += 1

        return renamed, conflicts

    def infer_dataset_name_from_training_dir(job_dir: Path) -> str:
        dataset_toml = job_dir / "dataset.toml"
        if dataset_toml.exists() and dataset_toml.is_file():
            try:
                content = dataset_toml.read_text(encoding="utf-8", errors="replace")
            except OSError:
                content = ""
            match = re.search(r'^\s*image_directory\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
            if match is not None:
                try:
                    image_dir = Path(match.group(1).strip()).expanduser()
                    if not image_dir.is_absolute():
                        image_dir = (dataset_toml.parent / image_dir).resolve()
                    if image_dir.name.lower() == "images" and image_dir.parent.name:
                        return image_dir.parent.name
                    if image_dir.name:
                        return image_dir.name
                except Exception:
                    pass
        return ""

    def build_recovered_job_from_training_dir(job_dir: Path) -> dict[str, str]:
        training_name = job_dir.name.strip()
        dataset_name = infer_dataset_name_from_training_dir(job_dir).strip() or training_name
        output_dir = (job_dir / "output").expanduser()
        return {
            "id": training_name,
            "dataset_name": dataset_name,
            "training_name": training_name,
            "training_dir": str(job_dir),
            "job_name": training_name,
            "model": "Klein",
            "output_dir": str(output_dir),
            "resolution": str(DEFAULT_RESOLUTION),
            "network_dim": str(DEFAULT_NETWORK_DIM),
            "network_alpha": str(DEFAULT_NETWORK_ALPHA),
            "optimizer_type": "prodigy",
            "learning_rate": DEFAULT_LEARNING_RATE,
            "train_steps": str(DEFAULT_TRAIN_STEPS),
            "save_every_n_steps": str(
                get_positive_int_setting(
                    settings_state,
                    TRAIN_SAVE_EVERY_N_STEPS_KEY,
                    DEFAULT_SAVE_EVERY_N_STEPS,
                    minimum=1,
                )
            ),
            "enable_compile": settings_state.get(ENABLE_COMPILE_OPTIMIZATIONS_KEY, "0"),
            "enable_tf32": settings_state.get(ENABLE_CUDA_ALLOW_TF32_KEY, "1"),
            "enable_cudnn": settings_state.get(ENABLE_CUDA_CUDNN_BENCHMARK_KEY, "1"),
            "enable_fp8": settings_state.get(ENABLE_FP8_DIT_KEY, "0"),
            "enable_gc": settings_state.get(ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY, "0"),
            "enable_logging": bool_to_flag(is_truthy(settings_state.get(TRAIN_ENABLE_LOGGING_KEY), default=True)),
            "tracker_name": training_name,
            "stream_output": bool_to_flag(is_truthy(settings_state.get(TRAIN_STREAM_TO_LOGGER_KEY), default=False)),
            "auto_cleanup": bool_to_flag(is_truthy(settings_state.get(TRAIN_AUTO_CLEANUP_STATES_KEY), default=True)),
            "hold": "0",
            "status": "queued",
        }

    def detect_job_status(job: dict[str, str]) -> str:
        hold = flag_to_bool(job.get("hold", "0"))
        output_dir = Path(job.get("output_dir", "")).expanduser()
        output_names = job_output_names(job)
        has_name_mismatch, _source_base = detect_job_element_base_mismatch(job)
        if has_name_mismatch:
            return "broken"

        target_steps = get_positive_int_setting(job, "train_steps", DEFAULT_TRAIN_STEPS)
        progress_step, has_resume_data = job_resume_progress(job)
        has_finished_output = any((output_dir / f"{name}.safetensors").exists() for name in output_names)

        # A final output file alone is not enough when the user increases target steps later.
        # If we can see resume progress and it is below the new target, keep this resumable.
        if has_resume_data and progress_step >= target_steps:
            return "done"
        if has_finished_output and not has_resume_data:
            return "done"

        if hold:
            return "paused"

        status = job.get("status", "queued").strip().lower()
        if status == "running":
            return "running" if run_in_progress else "queued"
        if status == "cancelled":
            return "resume" if has_resume_data else "queued"
        if has_resume_data:
            return "resume"
        if status in {"queued", "failed", "resume"}:
            return status
        return "queued"

    def persist_queue_state() -> None:
        for job in job_queue:
            save_job_to_disk(job)
        save_job_order()

    def load_job_queue_from_disk() -> None:
        ensure_jobs_storage()
        job_queue.clear()

        loaded_jobs: dict[str, dict[str, str]] = {}
        for child in sorted(jobs_storage_dir().iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            training_name = child.name.strip()
            if training_name.lower() in {"logs", "__pycache__"}:
                continue

            settings_path = child / JOB_SETTINGS_FILE_NAME
            if settings_path.exists() and settings_path.is_file():
                try:
                    raw = json.loads(settings_path.read_text(encoding="utf-8"))
                except Exception:
                    raw = None
                if not isinstance(raw, dict):
                    raw = None
            else:
                raw = None

            if raw is None:
                if not (
                    (child / "dataset.toml").exists()
                    or (child / "output").is_dir()
                    or (child / "cache").is_dir()
                    or (child / "logs").is_dir()
                ):
                    continue
                normalized = build_recovered_job_from_training_dir(child)
            else:
                normalized = {str(k): str(v) for k, v in raw.items()}

            job_id = training_name
            normalized["id"] = training_name
            normalized["training_name"] = training_name
            normalized["training_dir"] = str(child)
            normalized["output_dir"] = str((child / "output").expanduser())
            normalized["job_name"] = training_name
            if not normalized.get("dataset_name", "").strip():
                normalized["dataset_name"] = infer_dataset_name_from_training_dir(child).strip() or training_name
            normalized["status"] = detect_job_status(normalized)
            loaded_jobs[job_id] = normalized

        order: list[str] = []
        order_path = jobs_order_file_path()
        if order_path.exists():
            try:
                raw_order = json.loads(order_path.read_text(encoding="utf-8"))
                if isinstance(raw_order, list):
                    order = [str(item) for item in raw_order]
            except Exception:
                order = []

        for job_id in order:
            job = loaded_jobs.pop(job_id, None)
            if job is not None:
                job_queue.append(job)

        for job_id in sorted(loaded_jobs.keys()):
            job_queue.append(loaded_jobs[job_id])

        persist_queue_state()

    def selected_dataset_names() -> list[str]:
        return [name for name, var in vars_by_name.items() if var.get()]

    def selected_queue_index() -> int | None:
        selection = queue_list.selection()
        if not selection:
            return None
        try:
            index = int(selection[0])
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= len(job_queue):
            return None
        return index

    def set_queue_selection(index: int) -> None:
        if index < 0 or index >= len(job_queue):
            return
        item_id = str(index)
        queue_list.selection_set(item_id)
        queue_list.focus(item_id)
        queue_list.see(item_id)

    def refresh_job_queue_list() -> None:
        queue_list.delete(*queue_list.get_children())
        queue_thumb_by_item.clear()
        for index, job in enumerate(job_queue):
            status = detect_job_status(job)
            if job.get("status", "") != status:
                job["status"] = status
                save_job_to_disk(job)
            hold = flag_to_bool(job.get("hold", "0"))
            if hold:
                row_tags: list[str] = ["row_even_disabled" if index % 2 == 0 else "row_odd_disabled", "status_paused"]
            else:
                if status == "running":
                    row_tags = ["row_running", "status_running"]
                else:
                    row_tags = ["row_even" if index % 2 == 0 else "row_odd"]
                if status == "done":
                    row_tags.append("status_done")
                elif status == "failed":
                    row_tags.append("status_failed")
                elif status == "broken":
                    row_tags.append("status_failed")
                elif status == "resume":
                    row_tags.append("status_resume")
                elif status == "paused":
                    row_tags.append("status_paused")
            if status == "running":
                queue_thumb_state = "in_progress"
            elif status == "done":
                queue_thumb_state = "done"
            else:
                queue_thumb_state = "pending"
            queue_thumb = make_thumbnail(first_image_path(job.get("dataset_name", "").strip()), queue_thumb_state, 40)
            item_id = str(index)
            queue_thumb_by_item[item_id] = queue_thumb
            queue_list.insert(
                "",
                "end",
                iid=item_id,
                text="",
                values=(
                    "",
                    "",
                    "   " + job.get("job_name", "unnamed"),
                    "   " + job.get("dataset_name", "?"),
                    (
                        "IN PROGRESS"
                        if status == "running"
                        else ("RESUME" if status == "resume" else ("BROKEN" if status == "broken" else status.upper()))
                    ),
                    "",
                ),
                tags=tuple(row_tags),
            )
        build_queue_row_drag_handles()
        build_queue_row_checkbox_labels()
        build_queue_row_thumb_labels()
        build_queue_row_action_buttons()
        build_queue_row_dividers()
        build_queue_col_dividers()

    def move_selected_queue_item(direction: int) -> None:
        idx = selected_queue_index()
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(job_queue):
            return
        job_queue.insert(new_idx, job_queue.pop(idx))
        refresh_job_queue_list()
        set_queue_selection(new_idx)
        save_job_order()
        update_start_button_state()

    def toggle_hold_job(index: int) -> None:
        idx = index
        if idx < 0 or idx >= len(job_queue):
            return
        if detect_job_status(job_queue[idx]) == "done":
            return
        current = flag_to_bool(job_queue[idx].get("hold", "0"))
        next_hold = not current
        job_queue[idx]["hold"] = bool_to_flag(next_hold)
        if next_hold and job_queue[idx].get("status", "") in {"queued", "failed", "running", "resume"}:
            job_queue[idx]["status"] = "paused"
        elif (not next_hold) and job_queue[idx].get("status", "") == "paused":
            job_queue[idx]["status"] = "queued"
        save_job_to_disk(job_queue[idx])
        refresh_job_queue_list()
        set_queue_selection(idx)
        update_start_button_state()

    def toggle_hold_selected_job() -> None:
        idx = selected_queue_index()
        if idx is None:
            return
        toggle_hold_job(idx)

    def remove_selected_job() -> None:
        idx = selected_queue_index()
        if idx is None:
            return
        delete_job_with_confirmation(idx)

    def clear_queue() -> None:
        if not job_queue:
            return
        if not messagebox.askyesno("Clear queue", "Remove all queued jobs?", parent=root):
            return
        for job in job_queue:
            remove_job_from_disk(job)
        job_queue.clear()
        save_job_order()
        refresh_job_queue_list()
        update_start_button_state()

    def show_queue_context_menu(event: tk.Event) -> str:
        clicked_item = queue_list.identify_row(event.y)
        if not clicked_item:
            return "break"
        try:
            clicked = int(clicked_item)
        except ValueError:
            return "break"
        if clicked < 0 or clicked >= len(job_queue):
            return "break"
        set_queue_selection(clicked)

        def fix_clicked_job_names() -> None:
            fix_job_element_names(clicked)

        def clear_clicked_job_cache() -> None:
            clear_job_cache_with_confirmation(clicked)

        clicked_status = detect_job_status(job_queue[clicked])
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Open Output Folder", command=lambda: open_job_output_folder(clicked))
        menu.add_command(label="LoRA Post-Hoc EMA Merge", command=lambda: merge_job_output_loras(clicked))
        menu.add_command(label="Duplicate Job", command=lambda: duplicate_job(clicked))
        menu.add_command(label="Edit Job", command=lambda: open_create_job_dialog(existing_job=job_queue[clicked]))
        menu.add_command(label="Clear Job Cache", command=clear_clicked_job_cache)
        if clicked_status == "broken":
            menu.add_command(label="Fix LoRA Names", command=fix_clicked_job_names)
        menu.add_separator()
        menu.add_command(label="Archive Job", command=lambda: archive_job(clicked))
        menu.add_command(label="Delete Job", command=lambda: delete_job_with_confirmation(clicked))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def open_job_output_folder(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        output_dir = Path(job_queue[idx].get("output_dir", "")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(output_dir))
        except OSError as exc:
            messagebox.showerror("Open output failed", f"Could not open output folder:\n{exc}", parent=root)

    def merge_job_output_loras(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        job = job_queue[idx]
        job_name = job.get("job_name", "job")
        output_dir = Path(job.get("output_dir", "")).expanduser()
        merged_output_dir = output_dir / "merged"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        lora_post_hoc_ema_merge_for_output(job_name, output_dir, merged_output_dir)

    def fix_job_element_names(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        job = job_queue[idx]
        has_mismatch, source_base = detect_job_element_base_mismatch(job)
        if not has_mismatch or source_base is None:
            messagebox.showinfo("Fix LoRA Names", "No naming mismatch detected for this job.", parent=root)
            return

        target_base = (job.get("training_name", "").strip() or job.get("job_name", "").strip())
        if not target_base:
            return

        if not messagebox.askyesno(
            "Fix LoRA Names",
            (
                "Detected output element names that do not match this job folder.\n\n"
                f"From: {source_base}\n"
                f"To:   {target_base}\n\n"
                "Rename elements now?"
            ),
            parent=root,
        ):
            return

        renamed = 0
        conflicts = 0
        for _attempt in range(5):
            has_mismatch, next_source_base = detect_job_element_base_mismatch(job)
            if not has_mismatch or next_source_base is None:
                break
            renamed_count, conflict_count = rename_job_elements_to_training_name(job, next_source_base)
            renamed += renamed_count
            conflicts += conflict_count
            if renamed_count == 0:
                break
        refresh_job_queue_list()
        sync_queue_row_action_buttons()
        update_start_button_state()

        if renamed > 0 and conflicts == 0:
            messagebox.showinfo(
                "Fix LoRA Names",
                f"Renamed {renamed} element(s) to match '{target_base}'.",
                parent=root,
            )
            return
        if renamed > 0:
            messagebox.showwarning(
                "Fix LoRA Names",
                f"Renamed {renamed} element(s), with {conflicts} conflict(s).",
                parent=root,
            )
            return
        messagebox.showwarning(
            "Fix LoRA Names",
            "No elements were renamed. Existing files may already use target names.",
            parent=root,
        )

    def duplicate_job(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        source = dict(job_queue[idx])
        source_name = source.get("job_name", "job")
        source_dataset = source.get("dataset_name", "").strip()
        new_name = unique_job_name(f"{source_name}_copy")
        if not source_dataset:
            messagebox.showerror("Duplicate failed", "Source job has no dataset assigned.", parent=root)
            return

        resolution_value = get_positive_int_setting(source, "resolution", DEFAULT_RESOLUTION, minimum=64)
        raw_datasets = source.get("datasets_json", "")
        if raw_datasets:
            try:
                dup_datasets: list[dict] = json.loads(raw_datasets)
            except Exception:
                dup_datasets = [{"name": source_dataset, "num_repeats": 1}]
        else:
            dup_datasets = [{"name": source_dataset, "num_repeats": 1}]
        try:
            training_dir_path, output_dir_path, created_captions = ensure_training_job_structure(
                training_name=new_name,
                datasets=dup_datasets,
                resolution=resolution_value,
                default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                model_name=source.get("model", ""),
            )
        except Exception as exc:
            messagebox.showerror("Duplicate failed", str(exc), parent=root)
            return

        duplicate = {
            **source,
            "id": new_name,
            "job_name": new_name,
            "training_name": new_name,
            "training_dir": str(training_dir_path),
            "output_dir": str(output_dir_path),
            "status": "queued",
            "hold": "0",
        }
        insert_at = idx + 1
        job_queue.insert(insert_at, duplicate)
        save_job_to_disk(duplicate)
        save_job_order()
        refresh_job_queue_list()
        set_queue_selection(insert_at)
        update_start_button_state()
        log(
            f"[Queue] Duplicated job: {source_name} -> {new_name} (source dataset: {source_dataset}, captions added: {created_captions})"
        )

    def _job_cache_dirs(job: dict) -> list[Path]:
        training_name = job.get("training_name", "").strip() or job.get("job_name", "").strip()
        if not training_name:
            return []
        training_dir = Path(job.get("training_dir", "")).expanduser()
        if not training_dir.exists() or not training_dir.is_dir():
            training_dir = training_job_dir_path(training_name).expanduser()
        if not training_dir.exists() or not training_dir.is_dir():
            return []
        cache_dirs: list[Path] = []
        for child in training_dir.iterdir():
            if not child.is_dir():
                continue
            child_name = child.name.lower()
            if child_name == "cache" or child_name.startswith("cache_"):
                cache_dirs.append(child)
        return sorted(cache_dirs)

    def clear_job_cache_with_confirmation(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        job = job_queue[idx]
        job_name = job.get("job_name", "unnamed")
        cache_dirs = _job_cache_dirs(job)
        if not cache_dirs:
            messagebox.showinfo("Clear job cache", f"No cache folders found for '{job_name}'.", parent=root)
            return

        if not messagebox.askyesno(
            "Clear job cache",
            (
                f"Delete cached latents/text encodes for '{job_name}'?\n\n"
                "This removes files inside cache folders so they can be regenerated on next run."
            ),
            parent=root,
        ):
            return

        deleted_files = 0
        deleted_dirs = 0
        for cache_dir in cache_dirs:
            for child in list(cache_dir.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                        deleted_dirs += 1
                    else:
                        child.unlink(missing_ok=True)
                        deleted_files += 1
                except OSError as exc:
                    messagebox.showerror(
                        "Clear job cache failed",
                        f"Could not remove:\n{child}\n\n{exc}",
                        parent=root,
                    )
                    return

        log(
            f"[Queue] Cleared cache for {job_name}: {deleted_files} file(s), {deleted_dirs} folder(s) removed."
        )
        messagebox.showinfo(
            "Clear job cache",
            f"Cleared cache for '{job_name}'.\nRemoved {deleted_files} file(s) and {deleted_dirs} folder(s).",
            parent=root,
        )

    def delete_job_with_confirmation(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return
        job = job_queue[idx]
        job_name = job.get("job_name", "unnamed")
        training_name = job.get("training_name", "").strip() or job_name
        training_dir = Path(job.get("training_dir", "")).expanduser()
        if not training_dir.exists() and training_name:
            training_dir = training_job_dir_path(training_name).expanduser()

        if not messagebox.askyesno(
            "Delete job",
            (
                f"Delete job '{job_name}'?\n\n"
                "This will remove it from the queue and delete its job folder "
                "(output/cache/logs/config) if it exists."
            ),
            parent=root,
        ):
            return

        removed = job_queue.pop(idx)
        remove_job_from_disk(removed)
        save_job_order()

        if training_dir.exists() and training_dir.is_dir():
            try:
                shutil.rmtree(training_dir)
            except OSError as exc:
                messagebox.showerror("Delete job folder failed", f"Could not delete job folder:\n{exc}", parent=root)

        refresh_job_queue_list()
        update_start_button_state()
        log(f"[Queue] Deleted job: {job_name}")

    def on_queue_press(event: tk.Event) -> None:
        nonlocal queue_drag_index, queue_drag_moved, queue_drag_allowed
        if len(job_queue) == 0:
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return
        clicked_item = queue_list.identify_row(event.y)
        if not clicked_item:
            queue_list.selection_set([])
            root.after_idle(sync_all_row_overlays)
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return
        try:
            clicked = int(clicked_item)
        except ValueError:
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return
        if clicked < 0 or clicked >= len(job_queue):
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return

        set_queue_selection(clicked)

        clicked_col = queue_list.identify_column(event.x)
        if clicked_col == "#1":
            toggle_hold_job(clicked)
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return

        queue_drag_allowed = clicked_col in {"#0", "#2", "#3", "#4", "#5"}
        queue_drag_index = clicked if queue_drag_allowed else None
        queue_drag_moved = False

    def on_queue_motion(event: tk.Event) -> None:
        nonlocal queue_drag_index, queue_drag_moved, queue_drag_allowed
        if not queue_drag_allowed:
            return
        if queue_drag_index is None:
            return
        if len(job_queue) <= 1:
            return
        target_item = queue_list.identify_row(event.y)
        if not target_item:
            return
        try:
            target = int(target_item)
        except ValueError:
            return
        if target < 0 or target >= len(job_queue) or target == queue_drag_index:
            return
        job_queue.insert(target, job_queue.pop(queue_drag_index))
        queue_drag_index = target
        queue_drag_moved = True
        refresh_job_queue_list()
        set_queue_selection(target)

    def on_queue_release(_event: tk.Event) -> None:
        nonlocal queue_drag_index, queue_drag_moved, queue_drag_allowed
        if queue_drag_moved:
            save_job_order()
            update_start_button_state()
        queue_drag_index = None
        queue_drag_moved = False
        queue_drag_allowed = False

    def open_create_job_dialog(existing_job: dict[str, str] | None = None) -> None:
        if existing_job is None:
            selected = selected_dataset_names()
            if not selected:
                messagebox.showinfo("No source dataset selected", "Select at least one dataset card first.", parent=root)
                return
            initial_datasets: list[dict] = [{"name": n, "num_repeats": 1} for n in selected]
            dataset_name = selected[0]
        else:
            dataset_name = existing_job.get("dataset_name", "").strip()
            if not dataset_name:
                messagebox.showerror("Invalid job", "Job has no dataset name.", parent=root)
                return
            raw_datasets = existing_job.get("datasets_json", "")
            if raw_datasets:
                try:
                    initial_datasets = json.loads(raw_datasets)
                except Exception:
                    initial_datasets = [{"name": dataset_name, "num_repeats": 1}]
            else:
                initial_datasets = [{"name": dataset_name, "num_repeats": 1}]

        dialog = tk.Toplevel(root)
        dialog.withdraw()
        dialog.title("Edit Job" if existing_job is not None else "Create Job")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(False, False)
        set_dark_title_bar(dialog)

        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        outer = ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # ── Header: LoRA name + model ──────────────────────────────────────
        header_frame = ttk.Frame(outer)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)

        def _fit_create_job_dialog_to_content() -> None:
            if not dialog.winfo_exists():
                return
            dialog.update_idletasks()
            target_width = max(820, dialog.winfo_reqwidth())
            target_height = dialog.winfo_reqheight()
            if dialog.state() == "withdrawn":
                dialog.geometry(f"{target_width}x{target_height}")
            else:
                pos_x = dialog.winfo_x()
                pos_y = dialog.winfo_y()
                dialog.geometry(f"{target_width}x{target_height}+{pos_x}+{pos_y}")

        _model_to_family: dict[str, str] = {
            mn: fam
            for fam, models in DOWNLOAD_MODEL_FAMILIES.items()
            for mn in models
        }

        _KLEIN_MODELS = {"klein-base-9b", "klein-9b", "klein-base-4b", "klein-4b"}

        def _family_label(mn: str) -> str:
            if mn in _KLEIN_MODELS:
                return "Klein"
            fam = _model_to_family.get(mn, "")
            if fam == "FLUX.2":
                return "Flux2"
            return fam or mn

        model_var = tk.StringVar(value=(existing_job or {}).get("model", "klein-base-9b"))
        ltx_mode_var = tk.StringVar(value=(existing_job or {}).get("ltx_mode", "video"))
        default_job_name = f"{dataset_name}_{_family_label(model_var.get())}"
        if existing_job is not None:
            default_job_name = existing_job.get("job_name", default_job_name)
        job_name_var = tk.StringVar(value=default_job_name)
        _job_name_user_edited = [existing_job is not None]  # track manual edits

        # Build available model list from saved paths
        import json as _json_cj
        _mpaths_cj: dict[str, dict[str, str]] = {}
        try:
            _mpaths_cj = _json_cj.loads(settings_state.get(MODEL_PATHS_KEY, "{}"))
        except Exception:
            pass
        _all_mn = [mn for fam in DOWNLOAD_MODEL_FAMILIES.values() for mn in fam]
        _avail_models = [mn for mn in _all_mn if _mpaths_cj.get(mn, {}).get("dit")]
        # Backward compat: legacy klein key
        if not _avail_models and settings_state.get(KLEIN_DIT_KEY, "").strip():
            _avail_models = ["klein-base-9b"]
        if not _avail_models:
            _avail_models = ["klein-base-9b"]
        if model_var.get() not in _avail_models:
            model_var.set(_avail_models[0])
        _display_values = [DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn) for mn in _avail_models]
        _ltx_mode_display_to_value = {"Image Training": "video"}
        _ltx_mode_value_to_display = {
            value: key for key, value in _ltx_mode_display_to_value.items()
        }
        _ltx_image_lora_target_choices = (
            "t2v",
            "v2v",
            "video_sa",
            "video_sa_ff",
            "video_sa_ca_ff",
            "full",
            "lycoris",
        )

        def _normalize_ltx_mode_ui(value: str) -> str:
            mode = (value or "video").strip().lower()
            if mode in {"video", "v", "image"}:
                return "video"
            if mode in {"av", "va"}:
                return "av"
            if mode in {"audio", "a"}:
                return "audio"
            return "video"

        ltx_mode_var.set(_normalize_ltx_mode_ui(ltx_mode_var.get()))

        _preferred_presets_raw = settings_state.get(PREFERRED_PRESETS_BY_FAMILY_KEY, "")
        try:
            _preferred_presets_loaded = json.loads(_preferred_presets_raw) if _preferred_presets_raw else {}
        except Exception:
            _preferred_presets_loaded = {}
        preferred_preset_by_family: dict[str, str] = {}
        if isinstance(_preferred_presets_loaded, dict):
            preferred_preset_by_family = {
                str(k): str(v)
                for k, v in _preferred_presets_loaded.items()
                if isinstance(k, str) and isinstance(v, str)
            }

        ttk.Label(header_frame, text="LoRA name:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        _lora_name_entry = ttk.Entry(header_frame, textvariable=job_name_var, style="Flat.TEntry")
        _lora_name_entry.grid(row=0, column=1, sticky="ew")
        _lora_name_entry.bind("<Key>", lambda _e: _job_name_user_edited.__setitem__(0, True))
        ttk.Label(header_frame, text="Model:").grid(row=0, column=2, sticky="w", padx=(12, 8))
        _model_display_var = tk.StringVar(value=DOWNLOAD_MODEL_DISPLAY_NAMES.get(model_var.get(), model_var.get()))

        def _on_model_display_change(*_a: object) -> None:
            disp = _model_display_var.get()
            for mn, dn in DOWNLOAD_MODEL_DISPLAY_NAMES.items():
                if dn == disp:
                    model_var.set(mn)
                    break
            else:
                model_var.set(disp)
            # Auto-update job name suffix if the user hasn't manually edited it
            if not _job_name_user_edited[0]:
                current = job_name_var.get()
                # Replace suffix after the last underscore with the new family label
                base = current.rsplit("_", 1)[0] if "_" in current else current
                job_name_var.set(f"{base}_{_family_label(model_var.get())}")
            _sync_model_specific_controls()
            _refresh_preset_combo()

        _model_display_var.trace_add("write", _on_model_display_change)
        ttk.Combobox(header_frame, textvariable=_model_display_var, values=_display_values, state="readonly", width=28).grid(row=0, column=3, sticky="w")

        # ── Notebook ───────────────────────────────────────────────────────
        notebook = ttk.Notebook(outer)
        notebook.grid(row=1, column=0, sticky="nsew")

        # ── Tab 1: Training ────────────────────────────────────────────────
        training_tab = ttk.Frame(notebook, padding=10)
        training_tab.columnconfigure(1, weight=1)
        training_tab.columnconfigure(3, weight=1)
        notebook.add(training_tab, text="  Training  ")

        train_optimizer_var = tk.StringVar(
            value=(existing_job or {}).get("optimizer_type", "prodigy")
        )
        train_optimizer_args_var = tk.StringVar(
            value=(existing_job or {}).get("optimizer_args", "")
        )
        train_learning_rate_var = tk.StringVar(
            value=(existing_job or {}).get("learning_rate", DEFAULT_LEARNING_RATE)
        )
        train_steps_var = tk.StringVar(
            value=(existing_job or {}).get("train_steps", str(DEFAULT_TRAIN_STEPS))
        )
        _default_save_every_from_settings = str(
            get_positive_int_setting(
                settings_state,
                TRAIN_SAVE_EVERY_N_STEPS_KEY,
                DEFAULT_SAVE_EVERY_N_STEPS,
                minimum=1,
            )
        )
        train_save_every_var = tk.StringVar(
            value=(existing_job or {}).get("save_every_n_steps", _default_save_every_from_settings)
        )
        train_network_dim_var = tk.StringVar(
            value=(existing_job or {}).get("network_dim", str(DEFAULT_NETWORK_DIM))
        )
        train_network_alpha_var = tk.StringVar(
            value=(existing_job or {}).get("network_alpha", str(DEFAULT_NETWORK_ALPHA))
        )
        lr_scheduler_var = tk.StringVar(value=(existing_job or {}).get("lr_scheduler", "constant"))
        lr_warmup_steps_var = tk.StringVar(value=(existing_job or {}).get("lr_warmup_steps", "0"))
        gradient_accumulation_steps_var = tk.StringVar(value=(existing_job or {}).get("gradient_accumulation_steps", "1"))
        blocks_to_swap_var = tk.StringVar(value=(existing_job or {}).get("blocks_to_swap", "0"))
        timestep_sampling_var = tk.StringVar(value=(existing_job or {}).get("timestep_sampling", "sigma"))
        ltx_lora_target_preset_var = tk.StringVar(value=(existing_job or {}).get("ltx_lora_target_preset", "full"))
        ltx_first_frame_conditioning_p_var = tk.StringVar(value=(existing_job or {}).get("ltx_first_frame_conditioning_p", "0.5"))

        job_presets = load_job_presets_from_disk()
        preset_none_label = "---------"
        preset_name_var = tk.StringVar(value=preset_none_label)
        compile_var = tk.BooleanVar(
            value=flag_to_bool((existing_job or {}).get("enable_compile", bool_to_flag(is_truthy(settings_state.get(ENABLE_COMPILE_OPTIMIZATIONS_KEY), default=False))))
        )
        tf32_var = tk.BooleanVar(
            value=flag_to_bool((existing_job or {}).get("enable_tf32", bool_to_flag(is_truthy(settings_state.get(ENABLE_CUDA_ALLOW_TF32_KEY), default=True))))
        )
        cudnn_var = tk.BooleanVar(
            value=flag_to_bool((existing_job or {}).get("enable_cudnn", bool_to_flag(is_truthy(settings_state.get(ENABLE_CUDA_CUDNN_BENCHMARK_KEY), default=True))))
        )
        fp8_var = tk.BooleanVar(
            value=flag_to_bool((existing_job or {}).get("enable_fp8", bool_to_flag(is_truthy(settings_state.get(ENABLE_FP8_DIT_KEY), default=False))))
        )
        gc_var = tk.BooleanVar(
            value=flag_to_bool((existing_job or {}).get("enable_gc", bool_to_flag(is_truthy(settings_state.get(ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY), default=False))))
        )

        def _collect_preset_values() -> dict[str, str]:
            return {
                "optimizer_type": train_optimizer_var.get().strip().lower(),
                "optimizer_args": train_optimizer_args_var.get().strip(),
                "learning_rate": train_learning_rate_var.get().strip(),
                "train_steps": train_steps_var.get().strip(),
                "save_every_n_steps": train_save_every_var.get().strip(),
                "network_dim": train_network_dim_var.get().strip(),
                "network_alpha": train_network_alpha_var.get().strip(),
                "resolution": train_resolution_var.get().strip(),
                "batch_size": train_batch_var.get().strip(),
                "lr_scheduler": lr_scheduler_var.get().strip(),
                "lr_warmup_steps": lr_warmup_steps_var.get().strip(),
                "gradient_accumulation_steps": gradient_accumulation_steps_var.get().strip(),
                "blocks_to_swap": blocks_to_swap_var.get().strip(),
                "timestep_sampling": timestep_sampling_var.get().strip(),
                "ltx_mode": _normalize_ltx_mode_ui(ltx_mode_var.get()),
                "ltx_lora_target_preset": ltx_lora_target_preset_var.get().strip(),
                "ltx_first_frame_conditioning_p": ltx_first_frame_conditioning_p_var.get().strip(),
            }

        def _apply_preset_values(values: dict[str, str]) -> None:
            if "optimizer_type" in values:
                train_optimizer_var.set(values["optimizer_type"])
            if "optimizer_args" in values:
                train_optimizer_args_var.set(values["optimizer_args"])
            if "learning_rate" in values:
                train_learning_rate_var.set(values["learning_rate"])
            if "train_steps" in values:
                train_steps_var.set(values["train_steps"])
            if "save_every_n_steps" in values:
                train_save_every_var.set(values["save_every_n_steps"])
            if "network_dim" in values:
                train_network_dim_var.set(values["network_dim"])
            if "network_alpha" in values:
                train_network_alpha_var.set(values["network_alpha"])
            if "resolution" in values:
                train_resolution_var.set(values["resolution"])
            if "batch_size" in values:
                train_batch_var.set(values["batch_size"])
            if "lr_scheduler" in values:
                lr_scheduler_var.set(values["lr_scheduler"])
            if "lr_warmup_steps" in values:
                lr_warmup_steps_var.set(values["lr_warmup_steps"])
            if "gradient_accumulation_steps" in values:
                gradient_accumulation_steps_var.set(values["gradient_accumulation_steps"])
            if "blocks_to_swap" in values:
                blocks_to_swap_var.set(values["blocks_to_swap"])
            if "timestep_sampling" in values:
                timestep_sampling_var.set(values["timestep_sampling"])
            if "ltx_mode" in values:
                ltx_mode_var.set(_normalize_ltx_mode_ui(values["ltx_mode"]))
            if "ltx_lora_target_preset" in values:
                ltx_lora_target_preset_var.set(values["ltx_lora_target_preset"])
            if "ltx_first_frame_conditioning_p" in values:
                ltx_first_frame_conditioning_p_var.set(values["ltx_first_frame_conditioning_p"])

        def _preset_names_for_model(model_name: str) -> list[str]:
            selected_family = _model_to_family.get(model_name, "")
            names = {
                str(payload.get("name", "")).strip()
                for payload in job_presets.values()
                if isinstance(payload, dict)
                and (
                    str(payload.get("family", "")).strip() == selected_family
                    or str(payload.get("model", "")).strip() == model_name
                    or (
                        not str(payload.get("family", "")).strip()
                        and _model_to_family.get(str(payload.get("model", "")).strip(), "") == selected_family
                    )
                )
            }
            return sorted([name for name in names if name], key=str.casefold)

        def _preset_payload_for_model_name(model_name: str, preset_name: str) -> dict[str, object] | None:
            selected_family = _model_to_family.get(model_name, "")
            for payload in job_presets.values():
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("name", "")).strip() != preset_name:
                    continue
                payload_family = str(payload.get("family", "")).strip()
                payload_model = str(payload.get("model", "")).strip()
                if payload_family == selected_family:
                    return payload
                if payload_model == model_name:
                    return payload
                if not payload_family and _model_to_family.get(payload_model, "") == selected_family:
                    return payload
            return None

        preset_section = ttk.LabelFrame(training_tab, text="Preset", padding=8)
        preset_section.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        preset_section.columnconfigure(1, weight=1)
        ttk.Label(preset_section, text="Preset:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        preset_combo = ttk.Combobox(preset_section, textvariable=preset_name_var, state="readonly")
        preset_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        def _refresh_preset_combo() -> None:
            nonlocal job_presets
            job_presets = load_job_presets_from_disk()
            names = [preset_none_label] + _preset_names_for_model(model_var.get().strip())
            preset_combo.configure(values=names)
            if preset_name_var.get() not in names:
                preset_name_var.set(preset_none_label)

        def _on_preset_selected(*_args: object) -> None:
            selected_name = preset_name_var.get().strip()
            if selected_name == preset_none_label:
                return
            payload = _preset_payload_for_model_name(model_var.get().strip(), selected_name)
            if not isinstance(payload, dict):
                return
            values = payload.get("values", {})
            if isinstance(values, dict):
                _apply_preset_values({str(k): str(v) for k, v in values.items() if isinstance(k, str)})
                _sync_resolution_controls()
                _sync_model_specific_controls()

        preset_name_var.trace_add("write", _on_preset_selected)

        def _save_preset() -> None:
            current_model = model_var.get().strip()
            current_family = _model_to_family.get(current_model, "") or current_model
            initial_name = preset_name_var.get().strip()
            if not initial_name or initial_name == preset_none_label:
                initial_name = f"{current_family} preset"
            preset_name = simpledialog.askstring(
                "Save preset",
                "Preset name:",
                initialvalue=initial_name,
                parent=dialog,
            )
            if preset_name is None:
                return
            preset_name = preset_name.strip()
            if not preset_name:
                messagebox.showerror("Invalid preset", "Preset name is required.", parent=dialog)
                return
            if _preset_payload_for_model_name(current_model, preset_name) is not None:
                if not messagebox.askyesno(
                    "Overwrite preset",
                    f"Preset '{preset_name}' already exists for family {current_family}. Overwrite it?",
                    parent=dialog,
                ):
                    return
            save_job_preset_to_disk(current_model, current_family, preset_name, _collect_preset_values())
            _refresh_preset_combo()
            preset_name_var.set(preset_name)

        def _reload_presets() -> None:
            _refresh_preset_combo()

        def _delete_preset() -> None:
            selected_name = preset_name_var.get().strip()
            if not selected_name or selected_name == preset_none_label:
                messagebox.showerror("Delete preset", "Select a preset to delete.", parent=dialog)
                return

            current_model = model_var.get().strip()
            payload = _preset_payload_for_model_name(current_model, selected_name)
            if not isinstance(payload, dict):
                messagebox.showerror("Delete preset", "Preset could not be found on disk.", parent=dialog)
                return

            payload_family = str(payload.get("family", "")).strip() or _model_to_family.get(current_model, "")
            file_path_raw = str(payload.get("file", "")).strip()
            target_path = Path(file_path_raw) if file_path_raw else job_preset_file_path(payload_family, selected_name)

            if not messagebox.askyesno(
                "Delete preset",
                f"Delete preset '{selected_name}' for family {payload_family}?",
                parent=dialog,
            ):
                return

            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError as exc:
                messagebox.showerror("Delete preset", f"Could not delete preset:\n{exc}", parent=dialog)
                return

            _refresh_preset_combo()
            preset_name_var.set(preset_none_label)

        _reload_preset_button = ttk.Button(preset_section, text="\u21bb", command=_reload_presets, width=3)
        _reload_preset_button.grid(row=0, column=2, sticky="e")
        _save_preset_button = ttk.Button(preset_section, text="Save preset", command=_save_preset)
        _save_preset_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        _delete_preset_button = ttk.Button(preset_section, text="\U0001F5D1", command=_delete_preset, width=3)
        _delete_preset_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        attach_hover_tooltip(_reload_preset_button, "Reload presets from disk")
        attach_hover_tooltip(_delete_preset_button, "Delete selected preset")

        def _attach_field_tooltip(label_widget: tk.Widget, input_widget: tk.Widget, text: str) -> None:
            attach_hover_tooltip(label_widget, text)
            attach_hover_tooltip(input_widget, text)

        options = ttk.LabelFrame(training_tab, text="Training settings", padding=8)
        options.grid(row=1, column=0, columnspan=4, sticky="ew")
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        _optimizer_label = ttk.Label(options, text="Optimizer type:")
        _optimizer_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        train_optimizer_combo = ttk.Combobox(
            options, textvariable=train_optimizer_var, values=OPTIMIZER_TYPE_CHOICES, state="readonly"
        )
        train_optimizer_combo.grid(row=0, column=1, sticky="ew")
        _steps_label = ttk.Label(options, text="Training steps:")
        _steps_label.grid(row=0, column=2, sticky="w", padx=(12, 8))
        _steps_entry = ttk.Entry(options, textvariable=train_steps_var, style="Flat.TEntry")
        _steps_entry.grid(row=0, column=3, sticky="ew")

        _learning_rate_label = ttk.Label(options, text="Learning rate:")
        _learning_rate_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        train_learning_rate_entry = ttk.Entry(options, textvariable=train_learning_rate_var, style="Flat.TEntry")
        train_learning_rate_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        _network_dim_label = ttk.Label(options, text="LoRA network dim:")
        _network_dim_label.grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _network_dim_combo = ttk.Combobox(options, textvariable=train_network_dim_var, values=TRAIN_DIM_ALPHA_CHOICES, state="readonly")
        _network_dim_combo.grid(
            row=1, column=3, sticky="ew", pady=(6, 0)
        )
        _save_steps_label = ttk.Label(options, text="Save every N steps:")
        _save_steps_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _save_steps_entry = ttk.Entry(options, textvariable=train_save_every_var, style="Flat.TEntry")
        _save_steps_entry.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        _network_alpha_label = ttk.Label(options, text="LoRA network alpha:")
        _network_alpha_label.grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _network_alpha_combo = ttk.Combobox(options, textvariable=train_network_alpha_var, values=TRAIN_DIM_ALPHA_CHOICES, state="readonly")
        _network_alpha_combo.grid(
            row=2, column=3, sticky="ew", pady=(6, 0)
        )

        _optimizer_args_label = ttk.Label(options, text="Optimizer args:")
        _optimizer_args_label.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _optimizer_args_entry = ttk.Entry(options, textvariable=train_optimizer_args_var, style="Flat.TEntry")
        _optimizer_args_entry.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(6, 0))

        _attach_field_tooltip(
            _optimizer_label,
            train_optimizer_combo,
            "Optimizer algorithm for LoRA training. Prodigy uses lr=1 in this UI; other optimizers use the entered learning rate.",
        )
        _attach_field_tooltip(
            _steps_label,
            _steps_entry,
            "Total update steps to run. Higher values train longer and can improve fit at the cost of overfit risk.",
        )
        _attach_field_tooltip(
            _learning_rate_label,
            train_learning_rate_entry,
            "Main step size for updates. For Prodigy, this value is ignored and Musubi uses 1 automatically.",
        )
        _attach_field_tooltip(
            _optimizer_args_label,
            _optimizer_args_entry,
            "Optional optimizer key=value args separated by spaces. Example (Prodigy): safeguard_warmup=True use_bias_correction=True weight_decay=0.01 betas=(0.9,0.99)",
        )
        _attach_field_tooltip(
            _network_dim_label,
            _network_dim_combo,
            "LoRA rank (capacity). Higher rank can capture more detail but uses more VRAM and may overfit faster.",
        )
        _attach_field_tooltip(
            _save_steps_label,
            _save_steps_entry,
            "Checkpoint interval. Saves model/state every N steps so you can resume or pick the best checkpoint.",
        )
        _attach_field_tooltip(
            _network_alpha_label,
            _network_alpha_combo,
            "LoRA scaling value. Commonly kept equal to dim for balanced behavior.",
        )

        common_advanced = ttk.LabelFrame(training_tab, text="Advanced settings", padding=8)
        common_advanced.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        common_advanced.columnconfigure(1, weight=1)
        common_advanced.columnconfigure(3, weight=1)

        _lr_scheduler_label = ttk.Label(common_advanced, text="LR scheduler:")
        _lr_scheduler_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        _lr_scheduler_combo = ttk.Combobox(
            common_advanced,
            textvariable=lr_scheduler_var,
            values=("constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"),
            state="readonly",
        )
        _lr_scheduler_combo.grid(row=0, column=1, sticky="ew")
        _lr_warmup_label = ttk.Label(common_advanced, text="LR warmup steps:")
        _lr_warmup_label.grid(row=0, column=2, sticky="w", padx=(12, 8))
        _lr_warmup_entry = ttk.Entry(common_advanced, textvariable=lr_warmup_steps_var, style="Flat.TEntry")
        _lr_warmup_entry.grid(row=0, column=3, sticky="ew")

        _grad_accum_label = ttk.Label(common_advanced, text="Grad accumulation:")
        _grad_accum_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _grad_accum_entry = ttk.Entry(common_advanced, textvariable=gradient_accumulation_steps_var, style="Flat.TEntry")
        _grad_accum_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        _blocks_to_swap_label = ttk.Label(common_advanced, text="Blocks to swap:")
        _blocks_to_swap_label.grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _blocks_to_swap_entry = ttk.Entry(common_advanced, textvariable=blocks_to_swap_var, style="Flat.TEntry")
        _blocks_to_swap_entry.grid(row=1, column=3, sticky="ew", pady=(6, 0))

        _timestep_sampling_label = ttk.Label(common_advanced, text="Timestep sampling:")
        _timestep_sampling_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _timestep_sampling_combo = ttk.Combobox(
            common_advanced,
            textvariable=timestep_sampling_var,
            values=("sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "logsnr", "shifted_logit_normal"),
            state="readonly",
        )
        _timestep_sampling_combo.grid(row=2, column=1, sticky="ew", pady=(6, 0))

        _attach_field_tooltip(
            _lr_scheduler_label,
            _lr_scheduler_combo,
            "Learning-rate decay strategy over training. constant keeps LR fixed; warmup/linear/cosine vary it by step.",
        )
        _attach_field_tooltip(
            _lr_warmup_label,
            _lr_warmup_entry,
            "Number of initial steps used to ramp LR up. Helpful when using warmup schedulers.",
        )
        _attach_field_tooltip(
            _grad_accum_label,
            _grad_accum_entry,
            "Accumulate gradients across N steps before optimizer update. Increases effective batch size.",
        )
        _attach_field_tooltip(
            _blocks_to_swap_label,
            _blocks_to_swap_entry,
            "Model-specific memory/perf tuning knob from Musubi scripts. Keep at profile default unless needed.",
        )
        _attach_field_tooltip(
            _timestep_sampling_label,
            _timestep_sampling_combo,
            "How timesteps are sampled during flow-matching training. Recommended values are model-family dependent.",
        )

        model_specific = ttk.LabelFrame(training_tab, text="Model-specific settings", padding=8)
        model_specific.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        model_specific.columnconfigure(1, weight=1)
        model_specific.columnconfigure(3, weight=1)

        ltx_specific_frame = ttk.Frame(model_specific)
        ltx_specific_frame.grid(row=0, column=0, columnspan=4, sticky="ew")
        ltx_specific_frame.columnconfigure(1, weight=1)
        ltx_specific_frame.columnconfigure(3, weight=1)

        ttk.Label(ltx_specific_frame, text="LTX mode:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _ltx_mode_display_var = tk.StringVar(
            value=_ltx_mode_value_to_display.get(_normalize_ltx_mode_ui(ltx_mode_var.get()), "Image Training")
        )

        def _on_ltx_mode_display_change(*_args: object) -> None:
            ltx_mode_var.set(_ltx_mode_display_to_value.get(_ltx_mode_display_var.get(), "video"))

        _ltx_mode_display_var.trace_add("write", _on_ltx_mode_display_change)
        _ltx_mode_combo = ttk.Combobox(
            ltx_specific_frame,
            textvariable=_ltx_mode_display_var,
            values=list(_ltx_mode_display_to_value.keys()),
            state="readonly",
        )
        _ltx_mode_combo.grid(row=0, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(ltx_specific_frame, text="LoRA target preset:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _ltx_lora_target_combo = ttk.Combobox(
            ltx_specific_frame,
            textvariable=ltx_lora_target_preset_var,
            values=_ltx_image_lora_target_choices,
            state="readonly",
        )
        _ltx_lora_target_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(ltx_specific_frame, text="First-frame conditioning p:").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _ltx_first_frame_entry = ttk.Entry(ltx_specific_frame, textvariable=ltx_first_frame_conditioning_p_var, style="Flat.TEntry")
        _ltx_first_frame_entry.grid(row=1, column=3, sticky="ew", pady=(6, 0))

        attach_hover_tooltip(
            _ltx_lora_target_combo,
            (
                "LoRA target preset controls which parts of the LTX model receive LoRA adapters.\n"
                "For image-training workflows, presets are limited to video/image-relevant targets."
            ),
        )
        attach_hover_tooltip(
            _ltx_first_frame_entry,
            (
                "First-frame conditioning probability.\n"
                "Higher values bias training to preserve frame-0 identity/composition guidance."
            ),
        )

        flags = ttk.LabelFrame(training_tab, text="Advanced flags", padding=8)
        flags.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(flags, text="Enable Torch Compile", variable=compile_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(flags, text="Enable Allow TF32", variable=tf32_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(flags, text="Enable cuDNN Benchmark", variable=cudnn_var).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Checkbutton(flags, text="Enable FP8 (Low VRAM)", variable=fp8_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(flags, text="Enable CPU Gradient Checkpointing (Low RAM)", variable=gc_var).grid(
            row=1, column=1, columnspan=2, sticky="w", padx=(12, 0), pady=(6, 0)
        )

        def sync_optimizer_controls() -> None:
            optimizer_value = train_optimizer_var.get().strip().lower()
            if optimizer_value == "prodigy":
                if not train_optimizer_args_var.get().strip():
                    train_optimizer_args_var.set(DEFAULT_PRODIGY_OPTIMIZER_ARGS)
            else:
                if train_optimizer_args_var.get().strip() == DEFAULT_PRODIGY_OPTIMIZER_ARGS:
                    train_optimizer_args_var.set("")

        def _sync_model_specific_controls() -> None:
            model_name = model_var.get().strip()
            is_ltx = _model_to_family.get(model_name, "") == "LTX"
            if is_ltx:
                model_specific.grid()
                ltx_specific_frame.grid()
                _ltx_mode_combo.configure(state="disabled")
                ltx_mode_var.set("video")
                _ltx_mode_display_var.set(_ltx_mode_value_to_display["video"])
                if ltx_lora_target_preset_var.get().strip() not in _ltx_image_lora_target_choices:
                    ltx_lora_target_preset_var.set("full")
            else:
                model_specific.grid_remove()
                ltx_specific_frame.grid_remove()
            _fit_create_job_dialog_to_content()

        train_optimizer_var.trace_add("write", lambda *_args: sync_optimizer_controls())
        sync_optimizer_controls()
        _sync_model_specific_controls()
        _refresh_preset_combo()

        # ── Tab 2: Datasets ────────────────────────────────────────────────
        datasets_tab = ttk.Frame(notebook, padding=10)
        datasets_tab.columnconfigure(0, weight=1)
        datasets_tab.rowconfigure(1, weight=1)
        notebook.add(datasets_tab, text="  Datasets  ")

        # Resolution + Batch row (both written to [general] section of dataset.toml)
        res_row = ttk.Frame(datasets_tab)
        res_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        _saved_res = int((existing_job or {}).get("resolution", str(DEFAULT_RESOLUTION)))
        _saved_res_str = str(_saved_res) if _saved_res in RESOLUTION_CHOICES else str(RESOLUTION_CHOICES[RESOLUTION_CHOICES.index(1024)])
        train_resolution_var = tk.StringVar(value=_saved_res_str)
        _ltx_resolution_choices = (1280, 1920)
        _ltx_resolution_choice_values = [str(r) for r in _ltx_resolution_choices]
        _default_resolution_choice_values = [str(r) for r in RESOLUTION_CHOICES]
        _saved_batch = (existing_job or {}).get("batch_size", "1")
        train_batch_var = tk.StringVar(value=str(_saved_batch))
        ttk.Label(res_row, text="Resolution:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        _resolution_combo = ttk.Combobox(
            res_row, textvariable=train_resolution_var,
            values=_default_resolution_choice_values,
            state="readonly", width=7,
        )
        _resolution_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(res_row, text="Batch size:").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Entry(res_row, textvariable=train_batch_var, style="Flat.TEntry", width=5).grid(row=0, column=3, sticky="w")

        def _sync_resolution_controls() -> None:
            model_name = model_var.get().strip()
            is_ltx = _model_to_family.get(model_name, "") == "LTX"
            allowed_values = _ltx_resolution_choice_values if is_ltx else _default_resolution_choice_values
            _resolution_combo.configure(values=allowed_values)

            current_value = train_resolution_var.get().strip()
            if current_value not in allowed_values:
                train_resolution_var.set("1920" if is_ltx else _saved_res_str)

        _family_default_profiles: dict[str, dict[str, str]] = {
            "FLUX.2": {
                "optimizer_type": "prodigy",
                "optimizer_args": DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "train_steps": str(DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(DEFAULT_NETWORK_DIM),
                "network_alpha": str(DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "flux2_shift",
                "resolution": str(DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "LTX": {
                "optimizer_type": "adamw8bit",
                "optimizer_args": "",
                "learning_rate": DEFAULT_LEARNING_RATE,
                "train_steps": "400",
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": "16",
                "network_alpha": "16",
                "lr_scheduler": "constant_with_warmup",
                "lr_warmup_steps": "100",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "1",
                "timestep_sampling": "shifted_logit_normal",
                "resolution": "1920",
                "batch_size": "1",
                "ltx_lora_target_preset": "full",
                "ltx_first_frame_conditioning_p": "0.5",
            },
            "Wan": {
                "optimizer_type": "prodigy",
                "optimizer_args": DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "train_steps": str(DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(DEFAULT_NETWORK_DIM),
                "network_alpha": str(DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "shift",
                "resolution": str(DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "Z-Image": {
                "optimizer_type": "prodigy",
                "optimizer_args": DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "train_steps": str(DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(DEFAULT_NETWORK_DIM),
                "network_alpha": str(DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "shift",
                "resolution": str(DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "Qwen-Image": {
                "optimizer_type": "prodigy",
                "optimizer_args": DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "train_steps": str(DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(DEFAULT_NETWORK_DIM),
                "network_alpha": str(DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "qwen_shift",
                "resolution": str(DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
        }

        def _apply_family_defaults_if_needed() -> None:
            if existing_job is not None:
                return
            if preset_name_var.get().strip() != preset_none_label:
                return

            model_name = model_var.get().strip()
            family_name = _model_to_family.get(model_name, "")

            preferred_name = preferred_preset_by_family.get(family_name, "").strip()
            if preferred_name:
                preferred_payload = _preset_payload_for_model_name(model_name, preferred_name)
                if isinstance(preferred_payload, dict):
                    preferred_values = preferred_payload.get("values", {})
                    if isinstance(preferred_values, dict):
                        _apply_preset_values({str(k): str(v) for k, v in preferred_values.items() if isinstance(k, str)})
                        preset_name_var.set(preferred_name)
                        return
                # Preferred preset may have been deleted/renamed; fall back to family defaults.
                preferred_preset_by_family.pop(family_name, None)

            profile = _family_default_profiles.get(family_name)
            if not profile:
                return

            train_optimizer_var.set(profile["optimizer_type"])
            train_optimizer_args_var.set(profile.get("optimizer_args", ""))
            train_learning_rate_var.set(profile["learning_rate"])
            train_steps_var.set(profile["train_steps"])
            train_save_every_var.set(profile["save_every_n_steps"])
            train_network_dim_var.set(profile["network_dim"])
            train_network_alpha_var.set(profile["network_alpha"])
            lr_scheduler_var.set(profile["lr_scheduler"])
            lr_warmup_steps_var.set(profile["lr_warmup_steps"])
            gradient_accumulation_steps_var.set(profile["gradient_accumulation_steps"])
            blocks_to_swap_var.set(profile["blocks_to_swap"])
            timestep_sampling_var.set(profile["timestep_sampling"])
            train_resolution_var.set(profile["resolution"])
            train_batch_var.set(profile["batch_size"])
            if "ltx_lora_target_preset" in profile:
                ltx_lora_target_preset_var.set(profile["ltx_lora_target_preset"])
            if "ltx_first_frame_conditioning_p" in profile:
                ltx_first_frame_conditioning_p_var.set(profile["ltx_first_frame_conditioning_p"])

        def _on_model_change_for_defaults(*_args: object) -> None:
            _sync_resolution_controls()
            _apply_family_defaults_if_needed()
            _sync_resolution_controls()

        def _on_preset_change_for_defaults(*_args: object) -> None:
            _apply_family_defaults_if_needed()

        model_var.trace_add("write", _on_model_change_for_defaults)
        preset_name_var.trace_add("write", _on_preset_change_for_defaults)
        _sync_resolution_controls()
        _apply_family_defaults_if_needed()
        _sync_resolution_controls()

        # Dataset list
        list_host = ttk.LabelFrame(datasets_tab, text="Datasets", padding=8)
        list_host.grid(row=1, column=0, sticky="nsew")
        list_host.columnconfigure(0, weight=1)
        list_host.rowconfigure(0, weight=1)

        ds_canvas = tk.Canvas(list_host, bg=bg_panel, highlightthickness=0, height=120)
        ds_canvas.grid(row=0, column=0, sticky="nsew")
        ds_scrollbar = ttk.Scrollbar(list_host, orient="vertical", command=ds_canvas.yview, style="Dark.Vertical.TScrollbar")
        ds_scrollbar.grid(row=0, column=1, sticky="ns")
        ds_canvas.configure(yscrollcommand=ds_scrollbar.set)

        ds_inner = ttk.Frame(ds_canvas)
        ds_inner_id = ds_canvas.create_window((0, 0), window=ds_inner, anchor="nw")
        ds_inner.columnconfigure(0, weight=1)

        def _sync_ds_scroll(_e: object = None) -> None:
            ds_canvas.configure(scrollregion=ds_canvas.bbox("all"))

        def _sync_ds_canvas_width(e: tk.Event) -> None:
            ds_canvas.itemconfigure(ds_inner_id, width=e.width)

        ds_inner.bind("<Configure>", _sync_ds_scroll)
        ds_canvas.bind("<Configure>", _sync_ds_canvas_width)

        # dataset_entries: list of {"name", "num_repeats_var", "frame"}
        dataset_entries: list[dict] = []

        def _rebuild_ds_rows() -> None:
            for child in ds_inner.winfo_children():
                child.destroy()
            for idx, entry in enumerate(dataset_entries):
                row_frame = ttk.Frame(ds_inner, padding=(4, 3, 4, 3), style="Card.TFrame")
                row_frame.grid(row=idx, column=0, sticky="ew", pady=2)
                row_frame.columnconfigure(0, weight=1)
                entry["frame"] = row_frame
                ttk.Label(row_frame, text=entry["name"], style="CardTitle.TLabel").grid(
                    row=0, column=0, sticky="w", padx=(4, 12)
                )
                ttk.Label(row_frame, text="Repeats:", style="CardMeta.TLabel").grid(
                    row=0, column=1, sticky="e", padx=(0, 6)
                )
                ttk.Entry(row_frame, textvariable=entry["num_repeats_var"], style="Flat.TEntry", width=5).grid(
                    row=0, column=2, sticky="e", padx=(0, 8)
                )

                def _make_remove(e: dict = entry) -> None:
                    dataset_entries.remove(e)
                    _rebuild_ds_rows()
                    _refresh_add_combo()

                ttk.Button(row_frame, text="✕", style="QueueAction.TButton", command=_make_remove, width=2).grid(
                    row=0, column=3, sticky="e"
                )

        def _available_datasets() -> list[str]:
            return sorted(scan_training_folders(datasets_root_dir()), key=str.casefold)

        def _refresh_add_combo() -> None:
            already = {e["name"] for e in dataset_entries}
            available = [n for n in _available_datasets() if n not in already]
            add_combo["values"] = available
            if available and add_combo.get() not in available:
                add_combo.set(available[0])
            elif not available:
                add_combo.set("")

        # Populate from initial_datasets
        for _ds in initial_datasets:
            dataset_entries.append({"name": _ds["name"], "num_repeats_var": tk.StringVar(value=str(_ds.get("num_repeats", 1))), "frame": None})
        _rebuild_ds_rows()

        # Add dataset row
        add_row = ttk.Frame(datasets_tab, padding=(0, 8, 0, 0))
        add_row.grid(row=2, column=0, sticky="ew")
        add_row.columnconfigure(1, weight=1)
        ttk.Label(add_row, text="Add dataset:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        add_combo = ttk.Combobox(add_row, state="readonly", width=24)
        add_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        _refresh_add_combo()

        def _add_dataset() -> None:
            name = add_combo.get().strip()
            if not name or any(e["name"] == name for e in dataset_entries):
                return
            dataset_entries.append({"name": name, "num_repeats_var": tk.StringVar(value="1"), "frame": None})
            _rebuild_ds_rows()
            _refresh_add_combo()

        ttk.Button(add_row, text="Add", command=_add_dataset).grid(row=0, column=2, sticky="w")

        # ── create_job / save ──────────────────────────────────────────────
        def create_job() -> None:
            job_name = job_name_var.get().strip()
            if not job_name:
                messagebox.showerror("Missing value", "LoRA name is required.", parent=dialog)
                return
            if not is_valid_folder_name(job_name):
                messagebox.showerror(
                    "Invalid name",
                    "LoRA name must be a valid folder name. Spaces and '-' are allowed.",
                    parent=dialog,
                )
                return
            if not dataset_entries:
                messagebox.showerror("No datasets", "Add at least one dataset in the Datasets tab.", parent=dialog)
                return

            resolution_value = int(train_resolution_var.get())

            try:
                batch_size_value = int(train_batch_var.get().strip())
                if batch_size_value < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "Batch size must be a positive integer.", parent=dialog)
                return

            try:
                _ = int(train_steps_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "Steps must be numeric.", parent=dialog)
                return

            try:
                save_every_n_steps_value = int(train_save_every_var.get().strip())
                if save_every_n_steps_value < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "Save every N steps must be a positive integer.", parent=dialog)
                return

            train_optimizer = train_optimizer_var.get().strip().lower()
            if train_optimizer not in set(OPTIMIZER_TYPE_CHOICES):
                messagebox.showerror(
                    "Invalid value",
                    "Optimizer type must be one of: " + ", ".join(OPTIMIZER_TYPE_CHOICES),
                    parent=dialog,
                )
                return
            train_optimizer_args = train_optimizer_args_var.get().strip()
            if train_optimizer == "prodigy" and not train_optimizer_args:
                train_optimizer_args = DEFAULT_PRODIGY_OPTIMIZER_ARGS
                train_optimizer_args_var.set(train_optimizer_args)

            train_learning_rate = train_learning_rate_var.get().strip()
            if train_optimizer == "prodigy":
                train_learning_rate = "1"
            if not train_learning_rate:
                messagebox.showerror("Invalid value", "Learning rate is required.", parent=dialog)
                return
            try:
                learning_rate_number = float(train_learning_rate)
            except ValueError:
                messagebox.showerror("Invalid value", "Learning rate must be numeric (example: 1e-4).", parent=dialog)
                return
            if learning_rate_number <= 0:
                messagebox.showerror("Invalid value", "Learning rate must be greater than 0.", parent=dialog)
                return

            lr_scheduler_value = lr_scheduler_var.get().strip().lower() or "constant"
            try:
                lr_warmup_steps_value = int(lr_warmup_steps_var.get().strip())
                if lr_warmup_steps_value < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "LR warmup steps must be a non-negative integer.", parent=dialog)
                return

            try:
                grad_accum_value = int(gradient_accumulation_steps_var.get().strip())
                if grad_accum_value < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "Gradient accumulation must be a positive integer.", parent=dialog)
                return

            try:
                blocks_to_swap_value = int(blocks_to_swap_var.get().strip())
                if blocks_to_swap_value < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "Blocks to swap must be a non-negative integer.", parent=dialog)
                return

            timestep_sampling_value = timestep_sampling_var.get().strip().lower() or "sigma"
            ltx_lora_target_preset_value = ltx_lora_target_preset_var.get().strip().lower() or "t2v"
            try:
                ltx_first_frame_conditioning_p_value = float(ltx_first_frame_conditioning_p_var.get().strip())
                if ltx_first_frame_conditioning_p_value < 0 or ltx_first_frame_conditioning_p_value > 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid value", "First-frame conditioning p must be a number between 0 and 1.", parent=dialog)
                return

            # Validate and collect per-dataset config
            datasets_config: list[dict] = []
            for entry in dataset_entries:
                try:
                    repeats = int(entry["num_repeats_var"].get().strip())
                    if repeats < 1:
                        raise ValueError
                except ValueError:
                    messagebox.showerror(
                        "Invalid value",
                        f"Repeats for '{entry['name']}' must be a positive integer.",
                        parent=dialog,
                    )
                    return
                datasets_config.append({"name": entry["name"], "num_repeats": repeats})

            existing_index: int | None = None
            if existing_job is not None:
                try:
                    existing_index = job_queue.index(existing_job)
                except ValueError:
                    existing_index = None

            for idx, queued_job in enumerate(job_queue):
                if existing_index is not None and idx == existing_index:
                    continue
                if queued_job.get("job_name", "").strip().lower() == job_name.lower():
                    messagebox.showerror("Duplicate name", "LoRA name already exists in queue.", parent=dialog)
                    return

            existing_training_name = ""
            if existing_job is not None:
                existing_training_name = (
                    (existing_job or {}).get("training_name", "").strip()
                    or (existing_job or {}).get("job_name", "").strip()
                )

            training_name = job_name
            renamed_training_folder = False

            if existing_job is not None and existing_training_name and training_name != existing_training_name:
                source_training_dir = training_job_dir_path(existing_training_name).expanduser()
                target_training_dir = training_job_dir_path(training_name).expanduser()

                if target_training_dir.exists():
                    messagebox.showerror(
                        "Duplicate name",
                        (
                            "A job folder with this LoRA name already exists:\n"
                            f"{target_training_dir}\n\n"
                            "Choose a different name."
                        ),
                        parent=dialog,
                    )
                    return

                if source_training_dir.exists() and source_training_dir.is_dir():
                    try:
                        shutil.move(str(source_training_dir), str(target_training_dir))
                        renamed_training_folder = True
                    except OSError as exc:
                        messagebox.showerror(
                            "Rename failed",
                            f"Could not rename job folder:\n{exc}",
                            parent=dialog,
                        )
                        return

            try:
                training_dir_path, output_root, created_captions = ensure_training_job_structure(
                    training_name=training_name,
                    datasets=datasets_config,
                    resolution=resolution_value,
                    batch_size=batch_size_value,
                    default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                    model_name=model_var.get().strip(),
                )
            except Exception as exc:
                messagebox.showerror("Create job failed", str(exc), parent=dialog)
                return

            primary_dataset = datasets_config[0]["name"]
            tracker_name = settings_state.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip() or job_name

            new_job = {
                "id": training_name,
                "dataset_name": primary_dataset,
                "datasets_json": json.dumps(datasets_config),
                "training_name": training_name,
                "training_dir": str(training_dir_path),
                "job_name": job_name,
                "model": model_var.get().strip() or "Klein",
                "ltx_mode": _normalize_ltx_mode_ui(ltx_mode_var.get()),
                "output_dir": str(output_root),
                "resolution": str(resolution_value),
                "batch_size": str(batch_size_value),
                "save_every_n_steps": str(save_every_n_steps_value),
                "network_dim": train_network_dim_var.get().strip(),
                "network_alpha": train_network_alpha_var.get().strip(),
                "optimizer_type": train_optimizer,
                "optimizer_args": train_optimizer_args,
                "learning_rate": train_learning_rate,
                "train_steps": train_steps_var.get().strip(),
                "lr_scheduler": lr_scheduler_value,
                "lr_warmup_steps": str(lr_warmup_steps_value),
                "gradient_accumulation_steps": str(grad_accum_value),
                "blocks_to_swap": str(blocks_to_swap_value),
                "timestep_sampling": timestep_sampling_value,
                "ltx_lora_target_preset": ltx_lora_target_preset_value,
                "ltx_first_frame_conditioning_p": str(ltx_first_frame_conditioning_p_value),
                "enable_compile": bool_to_flag(compile_var.get()),
                "enable_tf32": bool_to_flag(tf32_var.get()),
                "enable_cudnn": bool_to_flag(cudnn_var.get()),
                "enable_fp8": bool_to_flag(fp8_var.get()),
                "enable_gc": bool_to_flag(gc_var.get()),
                "enable_logging": bool_to_flag(is_truthy(settings_state.get(TRAIN_ENABLE_LOGGING_KEY), default=True)),
                "tracker_name": tracker_name,
                "stream_output": bool_to_flag(is_truthy(settings_state.get(TRAIN_STREAM_TO_LOGGER_KEY), default=False)),
                "auto_cleanup": bool_to_flag(is_truthy(settings_state.get(TRAIN_AUTO_CLEANUP_STATES_KEY), default=True)),
                "hold": (existing_job or {}).get("hold", "0"),
                "status": "queued",
            }

            new_job["status"] = detect_job_status(new_job)

            auto_fixed_elements = 0
            if existing_job is not None:
                for _attempt in range(5):
                    has_mismatch, source_base = detect_job_element_base_mismatch(new_job)
                    if not has_mismatch or source_base is None:
                        break
                    renamed_count, _conflicts = rename_job_elements_to_training_name(new_job, source_base)
                    auto_fixed_elements += renamed_count
                    if renamed_count == 0:
                        break

            if existing_job is None:
                job_queue.append(new_job)
            else:
                if existing_index is None:
                    job_queue.append(new_job)
                else:
                    job_queue[existing_index] = new_job

            save_job_to_disk(new_job)
            save_job_order()
            if renamed_training_folder:
                load_job_queue_from_disk()
            refresh_job_queue_list()
            update_start_button_state()

            ds_names = ", ".join(d["name"] for d in datasets_config)
            if existing_job is None:
                log(f"[Queue] Created job: {job_name} (datasets: {ds_names}, training: {training_name}, captions added: {created_captions})")
                for var in vars_by_name.values():
                    var.set(False)
                for name in list(card_frame_by_name.keys()):
                    apply_card_style(name)
            else:
                if auto_fixed_elements > 0:
                    log(f"[Queue] Updated job: {job_name} (datasets: {ds_names}, training: {training_name}, captions added: {created_captions}, renamed elements: {auto_fixed_elements})")
                else:
                    log(f"[Queue] Updated job: {job_name} (datasets: {ds_names}, training: {training_name}, captions added: {created_captions})")
            dialog.destroy()

        # ── Footer buttons ─────────────────────────────────────────────────
        buttons = ttk.Frame(outer, padding=(0, 10, 0, 0))
        buttons.grid(row=2, column=0, sticky="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Save" if existing_job is not None else "Create Job", command=create_job).grid(row=0, column=1)

        _fit_create_job_dialog_to_content()
        center_window(dialog)
        dialog.deiconify()
        root.wait_window(dialog)

    def first_image_path(dataset_name: str) -> Path | None:
        cached = first_image_cache.get(dataset_name)
        if dataset_name in first_image_cache:
            return cached

        image_candidates = dataset_image_files(datasets_root_dir(), dataset_name)
        chosen = image_candidates[0] if image_candidates else None
        first_image_cache[dataset_name] = chosen
        return chosen

    def make_thumbnail(image_path: Path | None, run_state: str, thumb_px: int) -> ImageTk.PhotoImage:
        thumb_size = (thumb_px, thumb_px)
        cache_path = "__none__"
        cache_mtime_ns = 0
        if image_path is not None:
            cache_path = str(image_path)
            try:
                cache_mtime_ns = image_path.stat().st_mtime_ns
            except OSError:
                cache_mtime_ns = 0

        cache_key = (cache_path, run_state, thumb_px, cache_mtime_ns)
        cached_thumb = thumbnail_cache.get(cache_key)
        if cached_thumb is not None:
            return cached_thumb

        if image_path is None:
            image = Image.new("RGB", thumb_size, color="#3a3a3a")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                src_w, src_h = image.size
                dst_w, dst_h = thumb_size

                # Fill the entire thumbnail area and center-crop any overflow.
                scale = max(dst_w / src_w, dst_h / src_h)
                resized_w = max(1, int(round(src_w * scale)))
                resized_h = max(1, int(round(src_h * scale)))
                image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

                crop_x = max(0, (resized_w - dst_w) // 2)
                crop_y = max(0, (resized_h - dst_h) // 2)
                image = image.crop((crop_x, crop_y, crop_x + dst_w, crop_y + dst_h))
            except Exception:
                image = Image.new("RGB", thumb_size, color="#3a3a3a")

        if run_state == "done":
            ghost_overlay = Image.new("RGB", thumb_size, color="#141414")
            image = Image.blend(image, ghost_overlay, alpha=0.45)

        # Add a true 50% white border around every thumbnail.
        image_rgba = image.convert("RGBA")
        border_overlay = Image.new("RGBA", thumb_size, (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(border_overlay)
        border_draw.rectangle(
            (0, 0, thumb_size[0] - 1, thumb_size[1] - 1),
            outline=(255, 255, 255, 128),
            width=1,
        )
        image = Image.alpha_composite(image_rgba, border_overlay).convert("RGB")

        # Composite badge icons directly onto the image so alpha works correctly.
        badge_size = max(18, thumb_px // 8)
        badge_margin = 6
        icons_dir = Path(__file__).resolve().parent / "icons"

        def _load_badge_pil(icon_name: str) -> Image.Image | None:
            p = icons_dir / f"{icon_name}.png"
            if not p.exists():
                return None
            try:
                img = Image.open(p).convert("RGBA")
                if img.size != (badge_size, badge_size):
                    img = img.resize((badge_size, badge_size), Image.Resampling.LANCZOS)
                return img
            except Exception:
                return None

        image = image.convert("RGBA")
        if run_state == "done":
            badge = _load_badge_pil("ok")
            if badge is not None:
                image.paste(badge, (thumb_px - badge_margin - badge_size, badge_margin), badge)

        photo = ImageTk.PhotoImage(image, master=root)
        thumbnail_cache[cache_key] = photo
        return photo

    def toggle_dataset(name: str) -> None:
        if run_state_by_name.get(name) == "done":
            return
        vars_by_name[name].set(not vars_by_name[name].get())
        apply_card_style(name)

    def apply_card_style(name: str) -> None:
        card = card_frame_by_name.get(name)
        if card is None:
            return
        if drag_dataset_name == name:
            card.configure(style="DragSourceCard.TFrame")
            return
        if drag_hover_dataset_name == name:
            card.configure(style="DropTargetCard.TFrame")
            return
        run_state = run_state_by_name.get(name, "pending")
        if run_state == "done":
            card.configure(style="DoneCard.TFrame")
            return
        selected = vars_by_name.get(name).get() if name in vars_by_name else False
        card.configure(style=("SelectedCard.TFrame" if selected else "Card.TFrame"))

    def on_card_press(name: str) -> None:
        nonlocal drag_dataset_name, drag_hover_dataset_name, drag_moved, drag_start_x, drag_start_y
        drag_dataset_name = name
        drag_hover_dataset_name = None
        drag_moved = False
        drag_start_x = root.winfo_pointerx()
        drag_start_y = root.winfo_pointery()

    def on_card_motion() -> None:
        # Dataset drag reorder is intentionally disabled; queue supports drag ordering.
        return

    def dataset_name_from_widget(widget: tk.Misc | None) -> str | None:
        current: tk.Misc | None = widget
        while current is not None:
            for dataset_name, card_widget in card_frame_by_name.items():
                if current == card_widget:
                    return dataset_name
            parent_name = current.winfo_parent()
            if not parent_name:
                break
            try:
                current = current.nametowidget(parent_name)
            except Exception:
                break
        return None

    def on_card_release(target_name: str) -> str:
        nonlocal drag_dataset_name, drag_hover_dataset_name, drag_moved, drag_start_x, drag_start_y
        if drag_dataset_name is None:
            return "break"

        source_name = drag_dataset_name
        hovered_name = drag_hover_dataset_name
        moved = drag_moved
        drag_dataset_name = None
        drag_hover_dataset_name = None
        drag_moved = False
        drag_start_x = None
        drag_start_y = None
        hide_drag_preview()
        apply_card_style(source_name)
        if hovered_name is not None:
            apply_card_style(hovered_name)

        if moved:
            return "break"

        toggle_dataset(source_name)
        apply_card_style(source_name)
        update_start_button_state()
        return "break"

    def on_card_double_click(name: str) -> str:
        open_edit_dataset_dialog(name)
        return "break"

    def show_drag_preview(name: str) -> None:
        nonlocal drag_preview, drag_preview_photo
        hide_drag_preview()

        thumb = card_thumb_by_name.get(name)
        if thumb is None:
            return

        drag_preview = tk.Toplevel(root)
        drag_preview.overrideredirect(True)
        try:
            drag_preview.attributes("-topmost", True)
            drag_preview.attributes("-alpha", 0.85)
        except Exception:
            pass
        drag_preview.configure(bg="#1d1d1d")

        drag_preview_photo = thumb
        preview_frame = tk.Frame(drag_preview, bg="#2a2a2a", bd=1, relief="solid")
        preview_frame.pack(padx=0, pady=0)
        preview_image = tk.Label(preview_frame, image=drag_preview_photo, bg="#2a2a2a", bd=0)
        preview_image.pack(padx=4, pady=(4, 2))
        preview_text = tk.Label(preview_frame, text=name, fg="#e6e6e6", bg="#2a2a2a")
        preview_text.pack(padx=4, pady=(0, 4))
        move_drag_preview()

    def move_drag_preview() -> None:
        if drag_preview is None:
            return
        x = root.winfo_pointerx() + 14
        y = root.winfo_pointery() + 14
        drag_preview.geometry(f"+{x}+{y}")

    def hide_drag_preview() -> None:
        nonlocal drag_preview, drag_preview_photo
        if drag_preview is not None:
            try:
                drag_preview.destroy()
            except Exception:
                pass
        drag_preview = None
        drag_preview_photo = None

    def rebuild_folder_list(force: bool = False) -> None:
        nonlocal dataset_order
        selected_before = {name for name, var in vars_by_name.items() if var.get()}

        if force:
            first_image_cache.clear()
            thumbnail_cache.clear()
            checkpoint_cache.clear()

        for widget in card_widgets:
            widget.destroy()
        card_widgets.clear()
        vars_by_name.clear()
        run_state_by_name.clear()
        card_frame_by_name.clear()
        card_thumb_by_name.clear()

        ensure_datasets_root_dir()
        names = sorted(scan_training_folders(datasets_root_dir()), key=str.casefold)
        dataset_order = list(names)
        stale_names = [name for name in list(first_image_cache.keys()) if name not in names]
        for stale_name in stale_names:
            first_image_cache.pop(stale_name, None)
            checkpoint_cache.pop(stale_name, None)

        if not names:
            empty_label = ttk.Label(inner, text="No datasets found.")
            empty_label.grid(row=0, column=0, sticky="w")
            card_widgets.append(empty_label)
            update_start_button_state()
            return

        gap = ui_config["card_gap"]
        card_width = ui_config["card_width"]
        thumb_px = ui_config["thumbnail_size"]
        card_height = ui_config["card_height"]
        columns = 4

        for col in range(max(1, len(names))):
            inner.columnconfigure(col, minsize=0, weight=0)
        for col in range(columns):
            inner.columnconfigure(col, minsize=card_width + gap, weight=0)

        for idx, name in enumerate(names):
            train_state = "pending"
            run_state_by_name[name] = train_state

            var = tk.BooleanVar(value=(name in selected_before))
            vars_by_name[name] = var

            card_style = "DoneCard.TFrame" if train_state == "done" else ("SelectedCard.TFrame" if var.get() else "Card.TFrame")
            card = ttk.Frame(inner, padding=6, style=card_style, width=card_width, height=card_height)
            grid_row = idx // columns
            grid_col = idx % columns
            card.grid(row=grid_row, column=grid_col, sticky="nw", padx=4, pady=4)
            card.grid_propagate(False)
            card.columnconfigure(0, weight=1)
            card_frame_by_name[name] = card

            image_path = first_image_path(name)
            thumb = make_thumbnail(image_path, train_state, thumb_px)
            card_thumb_by_name[name] = thumb

            title_style = "DoneCardTitle.TLabel" if train_state == "done" else "CardTitle.TLabel"
            meta_style = "DoneCardMeta.TLabel" if train_state == "done" else "CardMeta.TLabel"

            image_label = ttk.Label(card, image=thumb, style=title_style, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")

            title_label = ttk.Label(card, text=name, style=title_style, anchor="center")
            title_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            image_count = len(dataset_image_files(datasets_root_dir(), name))
            status_text = f"{image_count} IMG"
            status_label = ttk.Label(card, text=status_text, style=meta_style, anchor="center")
            status_label.grid(row=2, column=0, sticky="ew", pady=(2, 8))

            click_targets: list[tk.Widget] = [card, image_label, title_label, status_label]

            for clickable in click_targets:
                clickable.bind("<ButtonPress-1>", lambda _e, n=name: on_card_press(n))
                clickable.bind("<B1-Motion>", lambda _e: on_card_motion())
                clickable.bind("<ButtonRelease-1>", lambda _e, n=name: on_card_release(n))
                clickable.bind("<Double-Button-1>", lambda _e, n=name: on_card_double_click(n))
                clickable.bind("<Button-3>", lambda e, n=name: show_thumbnail_context_menu(e, n))

            card_widgets.append(card)

        update_start_button_state()

    def request_relayout(canvas_width: int | None = None) -> None:
        nonlocal resize_after_id, last_canvas_width
        width = canvas_width if canvas_width is not None else canvas.winfo_width()
        if width <= 1:
            return
        if width == last_canvas_width:
            return
        last_canvas_width = width

        if resize_after_id is not None:
            root.after_cancel(resize_after_id)
        resize_after_id = root.after(ui_config["relayout_debounce_ms"], rebuild_folder_list)

    def is_widget_in_dataset_panel(widget: tk.Misc | None) -> bool:
        current: tk.Misc | None = widget
        while current is not None:
            if current == list_container:
                return True
            parent_name = current.winfo_parent()
            if not parent_name:
                break
            try:
                current = current.nametowidget(parent_name)
            except Exception:
                break
        return False

    def on_mousewheel(event: tk.Event) -> str | None:
        hovered = root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        if not is_widget_in_dataset_panel(hovered):
            return None
        delta = int(-event.delta / 120)
        if delta == 0:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")
        return "break"

    def on_mousewheel_linux_up(_event: tk.Event) -> str | None:
        hovered = root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        if not is_widget_in_dataset_panel(hovered):
            return None
        canvas.yview_scroll(-1, "units")
        return "break"

    def on_mousewheel_linux_down(_event: tk.Event) -> str | None:
        hovered = root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        if not is_widget_in_dataset_panel(hovered):
            return None
        canvas.yview_scroll(1, "units")
        return "break"

    root.bind_all("<MouseWheel>", on_mousewheel)
    root.bind_all("<Button-4>", on_mousewheel_linux_up)
    root.bind_all("<Button-5>", on_mousewheel_linux_down)

    def update_start_button_state() -> None:
        if run_in_progress:
            if run_cancel_event is not None and run_cancel_event.is_set():
                run_button.configure(text="Cancelling...", style="StartDisabled.TButton")
                run_button.state(["disabled"])
            else:
                run_button.configure(text="Queue In Progress (Press to Cancel)", style="StartInProgress.TButton")
                run_button.state(["!disabled"])
            return

        run_button.configure(text="START QUEUE")
        has_runnable_jobs = any(
            (not flag_to_bool(job.get("hold", "0"))) and job.get("status", "queued") in {"queued", "failed", "resume"}
            for job in job_queue
        )
        if has_runnable_jobs:
            run_button.configure(style="StartEnabled.TButton")
            run_button.state(["!disabled"])
        else:
            run_button.configure(style="StartDisabled.TButton")
            run_button.state(["disabled"])

    def run_queue() -> None:
        nonlocal run_in_progress, run_cancel_event
        if run_in_progress:
            if run_cancel_event is not None and not run_cancel_event.is_set():
                should_cancel = messagebox.askyesno(
                    "Cancel Queue",
                    "Stop current job and cancel all remaining queued jobs?",
                )
                if should_cancel:
                    run_cancel_event.set()
                    log("Cancellation requested. Stopping remaining queued jobs...")
                    update_start_button_state()
            return

        runnable_indices = [
            idx
            for idx, job in enumerate(job_queue)
            if (not flag_to_bool(job.get("hold", "0"))) and job.get("status", "queued") in {"queued", "failed", "paused", "resume"}
        ]
        if not runnable_indices:
            messagebox.showinfo("Queue is empty", "Add jobs and ensure at least one job is not on hold.", parent=root)
            return

        if runtime_config is None or runtime_config.dit is None or runtime_config.vae is None or runtime_config.text_encoder is None:
            missing = []
            if runtime_config is None or runtime_config.dit is None:
                missing.append("Model (DiT)")
            if runtime_config is None or runtime_config.vae is None:
                missing.append("VAE")
            if runtime_config is None or runtime_config.text_encoder is None:
                missing.append("Text Encoder")
            messagebox.showerror(
                "Model paths not configured",
                "The following model paths are not set:\n\n"
                + "\n".join(f"  \u2022 {m}" for m in missing)
                + "\n\nOpen Settings and configure the paths before starting training.",
                parent=root,
            )
            return

        run_cancel_event = threading.Event()
        run_in_progress = True
        update_start_button_state()
        model_error_popup_shown = False

        def friendly_model_config_error(details: str) -> tuple[str, str] | None:
            text = (details or "").strip()
            lowered = text.lower()

            if "memoryerror" in lowered or "out of memory" in lowered:
                return (
                    "Text Encoder Load Failed",
                    "Failed to load Klein Text Encoder due to insufficient memory or a corrupted checkpoint shard.\n\n"
                    "This can also happen if this Musubi build is given a .safetensors.index.json path instead of a shard file.\n\n"
                    "Verify Klein > Text Encoder is correct, and if needed reduce memory pressure before running again.\n"
                    "Tip: close other GPU/RAM-heavy apps and retry.",
                )

            if (
                "klein text encoder" in lowered
                or "model.embed_tokens.weight" in lowered
                or "flux_2_cache_text_encoder_outputs.py" in lowered
            ):
                return (
                    "Invalid Text Encoder",
                    "Selected Klein Text Encoder is invalid or incompatible.\n\n"
                    "Open Settings and set Klein > Text Encoder to the correct file, typically:\n"
                    "Models/klein/text_encoder/model.safetensors.index.json"
                )

            if "klein vae" in lowered or "flux_2_cache_latents.py" in lowered:
                return (
                    "Invalid VAE",
                    "Selected Klein VAE is invalid or incompatible.\n\n"
                    "Open Settings and verify Klein > VAE points to the correct Klein VAE checkpoint."
                )

            if "training launch failed due to invalid or incompatible klein model files" in lowered or "klein model" in lowered:
                return (
                    "Invalid Model Configuration",
                    "Selected Klein model files are invalid or incompatible.\n\n"
                    "Open Settings and verify Klein > Model, VAE, and Text Encoder paths."
                )

            return None

        def refresh_ui_now_from_worker() -> None:
            done = threading.Event()

            def do_refresh() -> None:
                try:
                    if not root.winfo_exists():
                        return
                    rebuild_folder_list(force=True)
                    update_start_button_state()
                finally:
                    done.set()

            try:
                root.after(0, do_refresh)
            except Exception:
                return
            done.wait(timeout=10)

        def background_train() -> None:
            nonlocal model_error_popup_shown
            try:
                log("")
                log("Queue is in progress...")
                failed_jobs: list[str] = []
                for queue_index in runnable_indices:
                    if run_cancel_event is not None and run_cancel_event.is_set():
                        break

                    job = job_queue[queue_index]
                    job_name = job.get("job_name", f"job_{queue_index + 1}")
                    dataset_name = job.get("dataset_name", "")

                    def mark_running() -> None:
                        if queue_index < len(job_queue):
                            job_queue[queue_index]["status"] = "running"
                            save_job_to_disk(job_queue[queue_index])
                            refresh_job_queue_list()

                    root.after(0, mark_running)

                    model_name = job.get("model", "klein-base-9b") or "klein-base-9b"
                    job_run_fn = _run_job_for_model(model_name)
                    if job_run_fn is None:
                        log(f"[Queue] Unsupported model '{model_name}' for job {job_name}.")
                        exit_code = JOB_EXIT_FAILED
                    else:
                        job_runtime_config = runtime_config_for_model(settings_state, model_name) or runtime_config
                        training_name = job.get("training_name", "").strip() or job_name
                        dataset_source_name = job.get("dataset_name", "").strip()
                        resolution_value = get_positive_int_setting(job, "resolution", DEFAULT_RESOLUTION, minimum=64)
                        batch_size_value = max(1, int(job.get("batch_size", "1") or "1"))
                        raw_job_datasets = job.get("datasets_json", "")
                        if raw_job_datasets:
                            try:
                                runner_datasets: list[dict] = json.loads(raw_job_datasets)
                            except Exception:
                                runner_datasets = [{"name": dataset_source_name, "num_repeats": 1}]
                        else:
                            runner_datasets = [{"name": dataset_source_name, "num_repeats": 1}]
                        job_error_details = ""

                        try:
                            _training_dir, output_dir, captions_added = ensure_training_job_structure(
                                training_name=training_name,
                                datasets=runner_datasets,
                                resolution=resolution_value,
                                batch_size=batch_size_value,
                                default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                                model_name=model_name,
                            )
                            if captions_added > 0:
                                ds_label = ", ".join(d["name"] for d in runner_datasets)
                                log(f"[Queue] Added {captions_added} missing caption file(s) for dataset(s): {ds_label}.")
                        except Exception as exc:
                            log(f"[Queue] Job setup failed for {job_name}: {exc}")
                            exit_code = 1
                            output_dir = Path(job.get("output_dir", str(training_job_dir_path(training_name) / "output"))).expanduser()
                        else:
                            job["training_name"] = training_name
                            job["training_dir"] = str(training_job_dir_path(training_name))
                            job["output_dir"] = str(output_dir)
                            save_job_to_disk(job)

                            def capture_job_error(message: str) -> None:
                                nonlocal job_error_details
                                job_error_details = message

                            run_job_kwargs = {
                                "dataset_name": training_name,
                                "output_name": job_name,
                                "output_dir": output_dir,
                                "default_caption_keyword": settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                                "resolution": resolution_value,
                                "network_dim": get_positive_int_setting(job, "network_dim", DEFAULT_NETWORK_DIM),
                                "network_alpha": get_positive_int_setting(job, "network_alpha", DEFAULT_NETWORK_ALPHA),
                                "optimizer_type": job.get("optimizer_type", "prodigy"),
                                "optimizer_args": str(job.get("optimizer_args", "") or ""),
                                "learning_rate": job.get("learning_rate", DEFAULT_LEARNING_RATE),
                                "train_steps": get_positive_int_setting(job, "train_steps", DEFAULT_TRAIN_STEPS),
                                "save_every_n_steps": get_positive_int_setting(job, "save_every_n_steps", DEFAULT_SAVE_EVERY_N_STEPS, minimum=1),
                                "enable_compile_optimizations": flag_to_bool(job.get("enable_compile", "0")),
                                "enable_cuda_allow_tf32": flag_to_bool(job.get("enable_tf32", "1")),
                                "enable_cuda_cudnn_benchmark": flag_to_bool(job.get("enable_cudnn", "1")),
                                "enable_fp8_dit": flag_to_bool(job.get("enable_fp8", "0")),
                                "enable_gradient_checkpointing_cpu_offload": flag_to_bool(job.get("enable_gc", "0")),
                                "enable_training_logging": is_truthy(settings_state.get(TRAIN_ENABLE_LOGGING_KEY), default=True),
                                "training_log_backend": get_train_log_backend_setting(settings_state),
                                "training_log_tracker_name": settings_state.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
                                "stream_training_output": is_truthy(settings_state.get(TRAIN_STREAM_TO_LOGGER_KEY), default=False),
                                "auto_cleanup_states": is_truthy(settings_state.get(TRAIN_AUTO_CLEANUP_STATES_KEY), default=True),
                                "logger": log,
                                "do_prep_dataset": True,
                                "do_cache_latents": True,
                                "do_cache_text": True,
                                "do_train": True,
                                "cancel_requested": (lambda: run_cancel_event is not None and run_cancel_event.is_set()),
                                "on_error": capture_job_error,
                            }
                            if job_run_fn is _run_job_ltx:
                                run_job_kwargs["ltx_mode"] = str(job.get("ltx_mode", "video"))
                                run_job_kwargs["lr_scheduler"] = str(job.get("lr_scheduler", "constant"))
                                run_job_kwargs["lr_warmup_steps"] = get_non_negative_int_setting(job, "lr_warmup_steps", 0)
                                run_job_kwargs["gradient_accumulation_steps"] = get_positive_int_setting(job, "gradient_accumulation_steps", 1, minimum=1)
                                run_job_kwargs["blocks_to_swap"] = get_non_negative_int_setting(job, "blocks_to_swap", 0)
                                run_job_kwargs["timestep_sampling"] = str(job.get("timestep_sampling", "sigma") or "sigma")
                                run_job_kwargs["ltx_lora_target_preset"] = str(job.get("ltx_lora_target_preset", "t2v") or "t2v")
                                try:
                                    run_job_kwargs["ltx_first_frame_conditioning_p"] = float(
                                        str(job.get("ltx_first_frame_conditioning_p", "0.1") or "0.1")
                                    )
                                except ValueError:
                                    run_job_kwargs["ltx_first_frame_conditioning_p"] = 0.1

                            exit_code = job_run_fn(job_runtime_config, **run_job_kwargs)

                            if (
                                exit_code != 0
                                and not (run_cancel_event is not None and run_cancel_event.is_set())
                                and not model_error_popup_shown
                            ):
                                popup_payload = friendly_model_config_error(job_error_details)
                                if popup_payload is not None:
                                    popup_title, popup_message = popup_payload
                                    model_error_popup_shown = True

                                    def show_model_error_popup() -> None:
                                        if root.winfo_exists():
                                            messagebox.showerror(popup_title, popup_message, parent=root)

                                    root.after(0, show_model_error_popup)

                    def mark_done() -> None:
                        if queue_index < len(job_queue):
                            if exit_code == JOB_EXIT_SUCCESS:
                                next_status = "done"
                            elif exit_code == JOB_EXIT_CANCELLED:
                                next_status = "cancelled"
                            else:
                                next_status = "failed"
                            job_queue[queue_index]["status"] = next_status
                            save_job_to_disk(job_queue[queue_index])
                            refresh_job_queue_list()
                            update_start_button_state()

                    root.after(0, mark_done)

                    if (
                        exit_code not in {JOB_EXIT_SUCCESS, JOB_EXIT_CANCELLED}
                        and not (run_cancel_event is not None and run_cancel_event.is_set())
                    ):
                        log(
                            "[Queue] Tip: If this started after changing dataset repeats/resolution/model settings, "
                            "clear this job's caches and rerun (right-click job -> Clear Job Cache)."
                        )
                        failed_jobs.append(job_name)

                    if not (run_cancel_event is not None and run_cancel_event.is_set()):
                        refresh_ui_now_from_worker()

                if run_cancel_event is not None and run_cancel_event.is_set():
                    log("Queue cancelled by user.")
                elif failed_jobs:
                    log(f"Queue completed with failures: {', '.join(failed_jobs)}")
                else:
                    log("Queue completed.")
            except Exception as exc:
                log(f"Queue failed unexpectedly: {exc}")
                log(traceback.format_exc())
            finally:
                def finish_ui() -> None:
                    nonlocal run_in_progress, run_cancel_event
                    if not root.winfo_exists():
                        return
                    rebuild_folder_list(force=True)
                    run_in_progress = False
                    run_cancel_event = None
                    refresh_job_queue_list()
                    update_start_button_state()

                root.after(0, finish_ui)

        threading.Thread(target=background_train, daemon=True).start()

    scan_button = ttk.Button(controls, text="Scan Datasets", command=lambda: rebuild_folder_list(force=True))
    restore_datasets_button = ttk.Button(controls, text="Restore Datasets", command=open_restore_datasets_dialog)
    archive_datasets_button = ttk.Button(controls, text="Archive Datasets", command=archive_selected_datasets)
    create_dataset_button = ttk.Button(controls, text="Create Dataset", command=create_dataset)
    metrics_viewer_button = ttk.Button(controls, text="TensorBoard", command=open_metrics_viewer_dialog)
    lora_merge_tool_button = ttk.Button(controls, text="LoRA EMA Merge", command=open_lora_merge_tool_dialog)
    _settings_icon_path = Path(__file__).resolve().parent / "icons" / "settings.png"
    _settings_icon_img: ImageTk.PhotoImage | None = None
    try:
        with Image.open(_settings_icon_path) as _img:
            _raw = _img.convert("RGBA").resize((18, 18), Image.LANCZOS)
        _settings_icon_img = ImageTk.PhotoImage(_raw, master=root)
    except Exception:
        pass
    settings_button = ttk.Button(controls, command=apply_settings_from_dialog)
    if _settings_icon_img is not None:
        try:
            settings_button.configure(image=_settings_icon_img, padding=(4, 2))
            settings_button._icon_ref = _settings_icon_img  # type: ignore[attr-defined]
        except Exception:
            settings_button.configure(text="Settings")
    else:
        settings_button.configure(text="Settings")
    create_job_large_button = ttk.Button(dataset_actions_bar, text="Create Job", command=open_create_job_dialog)
    run_button = ttk.Button(start_bar, text="START QUEUE", command=run_queue, style="StartDisabled.TButton")

    scan_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    archive_datasets_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    restore_datasets_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    create_dataset_button.grid(row=0, column=4, padx=(0, 8), sticky="e")
    metrics_viewer_button.grid(row=0, column=5, padx=(0, 8), sticky="e")
    lora_merge_tool_button.grid(row=0, column=6, padx=(0, 8), sticky="e")
    settings_button.grid(row=0, column=7, sticky="e")

    create_job_large_button.configure(style="TButton")
    create_job_large_button.grid(row=0, column=0, sticky="ew")

    queue_list.bind("<Button-3>", show_queue_context_menu)
    queue_list.bind("<ButtonPress-1>", on_queue_press)
    queue_list.bind("<B1-Motion>", on_queue_motion)
    queue_list.bind("<ButtonRelease-1>", on_queue_release)

    def row_background_for_item(item_id: str) -> str:
        if item_id in queue_list.selection():
            return "#1e4a7a"
        tags = set(queue_list.item(item_id, "tags"))
        if "row_running" in tags:
            return "#163326"
        if "row_even_disabled" in tags:
            return "#191e28"
        if "row_odd_disabled" in tags:
            return "#141923"
        if "row_even" in tags:
            return "#1c2534"
        return "#17202e"

    def clear_queue_row_drag_handles() -> None:
        for handle_label in queue_row_drag_handles.values():
            handle_label.destroy()
        queue_row_drag_handles.clear()

    def place_queue_row_drag_handles() -> None:
        for item_id, handle_label in queue_row_drag_handles.items():
            cell_bbox = queue_list.bbox(item_id, "#0")
            if not cell_bbox:
                handle_label.place_forget()
                continue
            x, y, width, height = cell_bbox
            if width <= 0 or height <= 0:
                handle_label.place_forget()
                continue
            row_bg = row_background_for_item(item_id)
            handle_label.configure(bg=row_bg)
            handle_label.place(x=x, y=y, width=width, height=height)

    def build_queue_row_drag_handles() -> None:
        clear_queue_row_drag_handles()
        for item_id in queue_list.get_children():
            handle_label = tk.Label(
                queue_list,
                text="☰",
                font=("Segoe UI", 10, "bold"),
                fg="#8aa6c8",
                bg="#1c2534",
                bd=0,
                padx=0,
                pady=0,
                relief="flat",
                highlightthickness=0,
                cursor="fleur",
                anchor="center",
            )
            bind_thumb_overlay_events(handle_label)
            queue_row_drag_handles[item_id] = handle_label
        place_queue_row_drag_handles()

    def forward_overlay_mouse_event(event: tk.Event, sequence: str) -> str:
        widget = event.widget
        x = widget.winfo_x() + event.x
        y = widget.winfo_y() + event.y
        queue_list.event_generate(sequence, x=x, y=y)
        return "break"

    def bind_thumb_overlay_events(widget: tk.Widget) -> None:
        widget.bind("<ButtonPress-1>", lambda e: forward_overlay_mouse_event(e, "<ButtonPress-1>"))
        widget.bind("<B1-Motion>", lambda e: forward_overlay_mouse_event(e, "<B1-Motion>"))
        widget.bind("<ButtonRelease-1>", lambda e: forward_overlay_mouse_event(e, "<ButtonRelease-1>"))
        widget.bind("<Double-1>", lambda e: forward_overlay_mouse_event(e, "<Double-1>"))

    def clear_queue_row_checkbox_labels() -> None:
        for label in queue_row_checkbox_labels.values():
            label.destroy()
        queue_row_checkbox_labels.clear()

    def place_queue_row_checkbox_labels() -> None:
        for item_id, cb_label in queue_row_checkbox_labels.items():
            cell_bbox = queue_list.bbox(item_id, "run")
            if not cell_bbox:
                cb_label.place_forget()
                continue
            x, y, width, height = cell_bbox
            if width <= 0 or height <= 0:
                cb_label.place_forget()
                continue
            row_bg = row_background_for_item(item_id)
            cb_label.configure(bg=row_bg)
            cb_label.place(x=x, y=y, width=width, height=height)

    def build_queue_row_checkbox_labels() -> None:
        clear_queue_row_checkbox_labels()
        for item_id in queue_list.get_children():
            try:
                index = int(item_id)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(job_queue):
                continue
            job = job_queue[index]
            status = detect_job_status(job)
            hold = flag_to_bool(job.get("hold", "0"))
            row_bg = row_background_for_item(item_id)
            if status == "done":
                cb_text = "☒"
                cb_fg = "#5a6474"
                cb_cursor = "arrow"
            else:
                cb_text = "☐" if hold else "☑"
                cb_fg = "#4a6a8a" if hold else "#5eead4"
                cb_cursor = "hand2"
            cb_label = tk.Label(
                queue_list,
                text=cb_text,
                font=("Segoe UI", 17),
                fg=cb_fg,
                bg=row_bg,
                bd=0,
                padx=0,
                pady=0,
                relief="flat",
                highlightthickness=0,
                cursor=cb_cursor,
                anchor="center",
            )
            if status != "done":
                cb_label.bind("<Button-1>", lambda _e, idx=index: toggle_hold_job(idx))
            queue_row_checkbox_labels[item_id] = cb_label
        place_queue_row_checkbox_labels()

    def clear_queue_row_dividers() -> None:
        for div in queue_row_dividers.values():
            div.destroy()
        queue_row_dividers.clear()

    def place_queue_row_dividers() -> None:
        total_width = queue_list.winfo_width() - 4
        if total_width <= 0:
            return
        for item_id, div in queue_row_dividers.items():
            cell_bbox = queue_list.bbox(item_id, "#0")
            if not cell_bbox:
                div.place_forget()
                continue
            _, y, _, height = cell_bbox
            if height <= 0:
                div.place_forget()
                continue
            div.place(x=0, y=y + height - 1, width=total_width, height=1)

    def build_queue_row_dividers() -> None:
        clear_queue_row_dividers()
        for item_id in queue_list.get_children():
            div = tk.Frame(queue_list, bg="#2e4466", bd=0, highlightthickness=0)
            queue_row_dividers[item_id] = div
        place_queue_row_dividers()

    def place_queue_col_dividers() -> None:
        if not queue_col_dividers:
            return
        children = queue_list.get_children()
        first_visible = next(
            (iid for iid in children if queue_list.bbox(iid, "thumb")), None
        )
        if not first_visible:
            for div in queue_col_dividers:
                div.place_forget()
            return
        min_y: int | None = None
        max_y: int | None = None
        for iid in children:
            bb = queue_list.bbox(iid, "thumb")
            if bb:
                if min_y is None or bb[1] < min_y:
                    min_y = bb[1]
                if max_y is None or bb[1] + bb[3] > max_y:
                    max_y = bb[1] + bb[3]
        if min_y is None or max_y is None:
            for div in queue_col_dividers:
                div.place_forget()
            return
        total_h = max_y - min_y
        for div, col in zip(queue_col_dividers, ["run", "thumb", "name", "source", "status", "actions"]):
            bb = queue_list.bbox(first_visible, col)
            if not bb:
                div.place_forget()
                continue
            div.place(x=bb[0], y=0, width=1, height=max_y)
            div.lift()

    def build_queue_col_dividers() -> None:
        nonlocal queue_col_dividers
        for div in queue_col_dividers:
            div.destroy()
        queue_col_dividers = [
            tk.Frame(queue_list, bg="#2e4466", bd=0, highlightthickness=0)
            for _ in range(6)
        ]
        place_queue_col_dividers()

    def clear_queue_row_thumb_labels() -> None:
        for thumb_label in queue_row_thumb_labels.values():
            thumb_label.destroy()
        queue_row_thumb_labels.clear()

    def place_queue_row_thumb_labels() -> None:
        for item_id, thumb_label in queue_row_thumb_labels.items():
            cell_bbox = queue_list.bbox(item_id, "thumb")
            if not cell_bbox:
                thumb_label.place_forget()
                continue

            x, y, width, height = cell_bbox
            if width <= 0 or height <= 0:
                thumb_label.place_forget()
                continue

            thumb_width = 40
            thumb_height = 40
            start_x = x + max(0, (width - thumb_width) // 2)
            start_y = y + max(0, (height - thumb_height) // 2)
            thumb_label.place(x=start_x, y=start_y, width=thumb_width, height=thumb_height)

    def build_queue_row_thumb_labels() -> None:
        clear_queue_row_thumb_labels()

        for item_id in queue_list.get_children():
            thumb_image = queue_thumb_by_item.get(item_id)
            if thumb_image is None:
                continue
            row_bg = row_background_for_item(item_id)
            thumb_label = tk.Label(
                queue_list,
                image=thumb_image,
                bd=0,
                relief="flat",
                highlightthickness=0,
                bg=row_bg,
                cursor="fleur",
            )
            bind_thumb_overlay_events(thumb_label)
            queue_row_thumb_labels[item_id] = thumb_label

        place_queue_row_thumb_labels()

    def clear_queue_row_action_buttons() -> None:
        for delete_button in queue_row_action_buttons.values():
            delete_button.destroy()
        queue_row_action_buttons.clear()

    def place_queue_row_action_buttons() -> None:
        for item_id, delete_button in queue_row_action_buttons.items():
            cell_bbox = queue_list.bbox(item_id, "actions")
            if not cell_bbox:
                delete_button.place_forget()
                continue

            x, y, width, height = cell_bbox
            if width <= 0 or height <= 0:
                delete_button.place_forget()
                continue

            row_bg = row_background_for_item(item_id)
            delete_button.configure(bg=row_bg)

            button_height = max(20, height - 10)
            delete_width = 24
            total_width = delete_width
            start_x = x + max(2, (width - total_width) // 2)
            start_y = y + max(2, (height - button_height) // 2)

            delete_button.place(x=start_x, y=start_y, width=delete_width, height=button_height)

    def build_queue_row_action_buttons() -> None:
        clear_queue_row_action_buttons()

        for item_id in queue_list.get_children():
            try:
                index = int(item_id)
            except (TypeError, ValueError):
                continue

            delete_button = tk.Label(
                queue_list,
                text="✕",
                font=("Segoe UI", 13, "bold"),
                bd=0,
                padx=0,
                pady=0,
                relief="flat",
                highlightthickness=0,
                cursor="hand2",
                fg="#e05252",
                bg="#1c2534",
            )
            delete_button.bind("<Button-1>", lambda _event, idx=index: delete_job_with_confirmation(idx))
            queue_row_action_buttons[item_id] = delete_button

        place_queue_row_action_buttons()

    def sync_all_row_overlays() -> None:
        place_queue_row_drag_handles()
        place_queue_row_checkbox_labels()
        place_queue_row_thumb_labels()
        place_queue_row_action_buttons()
        place_queue_row_dividers()
        place_queue_col_dividers()

    def sync_queue_row_action_buttons(_event: tk.Event | None = None) -> None:
        sync_all_row_overlays()

    def on_queue_yscroll(first: str, last: str) -> None:
        queue_scroll.set(first, last)
        root.after_idle(sync_all_row_overlays)

    def on_queue_double_click(event: tk.Event) -> str:
        clicked_item = queue_list.identify_row(event.y)
        if not clicked_item:
            return "break"
        clicked_col = queue_list.identify_column(event.x)
        if clicked_col in {"#0", "#1", "#6"}:
            return "break"
        try:
            clicked = int(clicked_item)
        except ValueError:
            return "break"
        if clicked < 0 or clicked >= len(job_queue):
            return "break"
        set_queue_selection(clicked)
        open_create_job_dialog(existing_job=job_queue[clicked])
        return "break"

    queue_list.configure(yscrollcommand=on_queue_yscroll)
    queue_list.bind("<<TreeviewSelect>>", sync_queue_row_action_buttons)
    queue_list.bind("<Double-1>", on_queue_double_click)
    queue_list.bind("<Configure>", lambda _event: root.after_idle(sync_all_row_overlays))
    run_button.grid(row=0, column=0, sticky="ew")

    def on_canvas_configure(_event: tk.Event) -> None:
        update_scrollbar_visibility()

    canvas.bind("<Configure>", on_canvas_configure)

    load_job_queue_from_disk()
    rebuild_folder_list(force=True)
    refresh_job_queue_list()
    sync_queue_row_action_buttons()
    update_start_button_state()
    update_scrollbar_visibility()
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train selected datasets with optional step selection.")
    parser.add_argument("names", nargs="*", help="Dataset/model names under training")
    parser.add_argument("--prep-dataset", dest="prep_dataset", action="store_true", help="Run dataset prep step")
    parser.add_argument("--cache-latents", dest="cache_latents", action="store_true", help="Run latent caching step")
    parser.add_argument("--cache-text", dest="cache_text", action="store_true", help="Run text encoder caching step")
    parser.add_argument("--train", dest="train", action="store_true", help="Run train step")
    args = parser.parse_args()

    if args.names:
        settings = load_settings()
        runtime_config = runtime_config_from_settings(settings)
        if runtime_config is None:
            print(f"Missing settings file: {SETTINGS_FILE}")
            print("Run without CLI args once to set Musubi-Tuner location.")
            return 1

        any_step_flag = args.prep_dataset or args.cache_latents or args.cache_text or args.train
        do_prep_dataset = args.prep_dataset if any_step_flag else True
        do_cache_latents = args.cache_latents if any_step_flag else True
        do_cache_text = args.cache_text if any_step_flag else True
        do_train = args.train if any_step_flag else True

        model_names = [name.strip() for name in args.names if name.strip()]
        return train_models(
            runtime_config,
            model_names,
            default_caption_keyword=settings.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
            resolution=DEFAULT_RESOLUTION,
            network_dim=DEFAULT_NETWORK_DIM,
            network_alpha=DEFAULT_NETWORK_ALPHA,
            optimizer_type="prodigy",
            optimizer_args=DEFAULT_PRODIGY_OPTIMIZER_ARGS,
            learning_rate=DEFAULT_LEARNING_RATE,
            train_steps=DEFAULT_TRAIN_STEPS,
            enable_compile_optimizations=(
                settings.get(ENABLE_COMPILE_OPTIMIZATIONS_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_cuda_allow_tf32=(
                settings.get(ENABLE_CUDA_ALLOW_TF32_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_cuda_cudnn_benchmark=(
                settings.get(ENABLE_CUDA_CUDNN_BENCHMARK_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_fp8_dit=(
                settings.get(ENABLE_FP8_DIT_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_gradient_checkpointing_cpu_offload=(
                settings.get(ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY, "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            enable_training_logging=(
                settings.get(TRAIN_ENABLE_LOGGING_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            training_log_backend=get_train_log_backend_setting(settings),
            training_log_tracker_name=settings.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
            stream_training_output=(
                settings.get(TRAIN_STREAM_TO_LOGGER_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            auto_cleanup_states=(
                settings.get(TRAIN_AUTO_CLEANUP_STATES_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            logger=print,
            do_prep_dataset=do_prep_dataset,
            do_cache_latents=do_cache_latents,
            do_cache_text=do_cache_text,
            do_train=do_train,
        )

    return launch_ui()


if __name__ == "__main__":
    raise SystemExit(main())
