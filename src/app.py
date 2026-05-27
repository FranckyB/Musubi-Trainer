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
import uuid
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
    TRAIN_LEARNING_RATE_KEY,
    TRAIN_LOG_BACKEND_KEY,
    TRAIN_LOG_TRACKER_NAME_KEY,
    TRAIN_STREAM_TO_LOGGER_KEY,
    TRAIN_AUTO_START_TENSORBOARD_KEY,
    TRAIN_AUTO_CLEANUP_STATES_KEY,
    TRAIN_NETWORK_ALPHA_KEY,
    TRAIN_NETWORK_DIM_KEY,
    TRAIN_OPTIMIZER_TYPE_KEY,
    TRAIN_ENABLE_LOGGING_KEY,
    TRAIN_RESOLUTION_KEY,
    TRAIN_STEPS_KEY,
    WINDOW_HEIGHT_KEY,
    WINDOW_WIDTH_KEY,
    WINDOW_X_KEY,
    WINDOW_Y_KEY,
    load_settings,
    parse_int_setting,
    load_window_size,
    load_window_position,
    save_settings,
)
from .klein_runtime_config import KleinRuntimeConfig, klein_runtime_config_from_settings, resolve_musubi_python
from .klein_train import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_NETWORK_DIM,
    DEFAULT_RESOLUTION,
    DEFAULT_TRAIN_STEPS,
    JOB_EXIT_CANCELLED,
    JOB_EXIT_SUCCESS,
    run_job,
    train_models,
)


# Model files
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"
DATASET_ORDER_KEY = "dataset_order"
DRAG_START_THRESHOLD_PX = 20
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
DATASET_SETTINGS_FILE_NAME = "settings.json"
DATASET_USE_GLOBAL_TRAIN_KEY = "use_global_train_settings"
TRAIN_DIM_ALPHA_CHOICES = ("16", "32", "64")
JOB_SETTINGS_FILE_NAME = "settings.json"
JOBS_ORDER_FILE_NAME = "_order.json"


def get_positive_int_setting(settings: dict[str, str], key: str, fallback: int, minimum: int = 1) -> int:
    value = parse_int_setting(settings, key)
    if value is None or value < minimum:
        return fallback
    return value


def get_learning_rate_setting(settings: dict[str, str]) -> str:
    value = settings.get(TRAIN_LEARNING_RATE_KEY, "").strip()
    return value if value else DEFAULT_LEARNING_RATE


def get_train_optimizer_setting(settings: dict[str, str]) -> str:
    value = settings.get(TRAIN_OPTIMIZER_TYPE_KEY, "").strip().lower()
    return value if value in {"adamw8bit", "prodigy"} else "prodigy"


def get_train_log_backend_setting(settings: dict[str, str]) -> str:
    _value = settings.get(TRAIN_LOG_BACKEND_KEY, "").strip().lower()
    return "tensorboard"


def is_truthy(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


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
            window.update_idletasks()
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
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    root.title("Musubi Training Launcher")
    root.geometry(f"{ui_config['window_width']}x{ui_config['window_height']}")
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
            height = max(root.winfo_height(), min_height)
        else:
            saved_width, saved_height = saved_size
            width = max(min_width, saved_width)
            height = max(min_height, saved_height)

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
        window_position_applied = True

    def save_main_window_position_now() -> None:
        nonlocal settings_state
        if settings_reset_requested:
            return
        settings_state[WINDOW_X_KEY] = str(root.winfo_x())
        settings_state[WINDOW_Y_KEY] = str(root.winfo_y())
        settings_state[WINDOW_WIDTH_KEY] = str(root.winfo_width())
        settings_state[WINDOW_HEIGHT_KEY] = str(root.winfo_height())
        save_settings(settings_state)

    def schedule_main_window_position_save(_event: tk.Event) -> None:
        if not window_position_applied:
            return
        if root.state() != "normal":
            return
        save_main_window_position_now()

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
    style.configure("PathDisplay.TLabel", background="#1f1f1f", foreground=fg_text)
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
        foreground=[("selected", "#e8f4ff")],
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

    vars_by_name: dict[str, tk.BooleanVar] = {}
    card_widgets: list[tk.Widget] = []
    thumbnail_cache: dict[tuple[str, str, int, int, bool], ImageTk.PhotoImage] = {}
    first_image_cache: dict[str, Path | None] = {}
    checkpoint_cache: dict[str, tuple[Path | None, int]] = {}
    dataset_train_settings_cache: dict[str, dict[str, str]] = {}
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
    runtime_config = klein_runtime_config_from_settings(settings_state)
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
                "[Metrics Viewer] TensorBoard is not installed in configured Musubi Python."
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

    def persist_dataset_order() -> None:
        nonlocal settings_state
        save_dataset_order(settings_state, dataset_order)
        save_settings(settings_state)

    def global_train_settings() -> dict[str, str]:
        return {
            TRAIN_RESOLUTION_KEY: str(
                get_positive_int_setting(settings_state, TRAIN_RESOLUTION_KEY, DEFAULT_RESOLUTION, minimum=64)
            ),
            TRAIN_NETWORK_DIM_KEY: str(get_positive_int_setting(settings_state, TRAIN_NETWORK_DIM_KEY, DEFAULT_NETWORK_DIM)),
            TRAIN_NETWORK_ALPHA_KEY: str(
                get_positive_int_setting(settings_state, TRAIN_NETWORK_ALPHA_KEY, DEFAULT_NETWORK_ALPHA)
            ),
            TRAIN_OPTIMIZER_TYPE_KEY: get_train_optimizer_setting(settings_state),
            TRAIN_LEARNING_RATE_KEY: get_learning_rate_setting(settings_state),
            TRAIN_STEPS_KEY: str(get_positive_int_setting(settings_state, TRAIN_STEPS_KEY, DEFAULT_TRAIN_STEPS)),
        }

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
        dataset_name: str,
        resolution: int,
        default_caption_keyword: str,
    ) -> tuple[Path, Path, int]:
        dataset_dir = dataset_dir_path(dataset_name)
        image_files = dataset_image_files(datasets_root_dir(), dataset_name)
        if not image_files:
            raise RuntimeError(f"Dataset images not found in: {dataset_dir}")
        images_dir = image_files[0].parent

        job_dir = training_job_dir_path(training_name)
        cache_dir = job_dir / "cache"
        output_dir = job_dir / "output"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_captions = 0
        caption_text = default_caption_keyword.strip()
        for image_path in image_files:
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                continue
            caption_path.write_text(caption_text, encoding="utf-8")
            created_captions += 1

        dataset_toml_path = job_dir / "dataset.toml"
        dataset_toml_path.write_text(
            "\n".join(
                [
                    "[general]",
                    f"resolution = [{resolution}, {resolution}]",
                    'caption_extension = ".txt"',
                    "batch_size = 1",
                    "enable_bucket = true",
                    "bucket_no_upscale = false",
                    "",
                    "[[datasets]]",
                    f'image_directory = "{images_dir.resolve().as_posix()}"',
                    f'cache_directory = "{cache_dir.resolve().as_posix()}"',
                    "num_repeats = 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return job_dir, output_dir, created_captions

    def dataset_settings_path(dataset_name: str) -> Path:
        if runtime_config is None:
            return Path(DATASET_SETTINGS_FILE_NAME)
        return dataset_dir_path(dataset_name) / DATASET_SETTINGS_FILE_NAME

    def load_dataset_train_settings_raw(dataset_name: str, refresh: bool = False) -> dict[str, str]:
        if not refresh and dataset_name in dataset_train_settings_cache:
            return dict(dataset_train_settings_cache[dataset_name])

        path = dataset_settings_path(dataset_name)
        if not path.exists() or not path.is_file():
            dataset_train_settings_cache[dataset_name] = {}
            return {}

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            dataset_train_settings_cache[dataset_name] = {}
            return {}

        if not isinstance(loaded, dict):
            dataset_train_settings_cache[dataset_name] = {}
            return {}

        normalized = {str(k): str(v) for k, v in loaded.items()}
        dataset_train_settings_cache[dataset_name] = normalized
        return dict(normalized)

    def save_dataset_train_settings_raw(dataset_name: str, raw_settings: dict[str, str]) -> None:
        path = dataset_settings_path(dataset_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw_settings, indent=2), encoding="utf-8")
        dataset_train_settings_cache[dataset_name] = dict(raw_settings)

    def effective_train_settings_for_dataset(dataset_name: str) -> dict[str, str]:
        global_values = global_train_settings()
        raw = load_dataset_train_settings_raw(dataset_name)
        use_global = is_truthy(raw.get(DATASET_USE_GLOBAL_TRAIN_KEY), default=True)

        if use_global:
            return {
                DATASET_USE_GLOBAL_TRAIN_KEY: "1",
                **global_values,
            }

        return {
            DATASET_USE_GLOBAL_TRAIN_KEY: "0",
            TRAIN_RESOLUTION_KEY: str(
                get_positive_int_setting(raw, TRAIN_RESOLUTION_KEY, int(global_values[TRAIN_RESOLUTION_KEY]), minimum=64)
            ),
            TRAIN_NETWORK_DIM_KEY: str(
                get_positive_int_setting(raw, TRAIN_NETWORK_DIM_KEY, int(global_values[TRAIN_NETWORK_DIM_KEY]))
            ),
            TRAIN_NETWORK_ALPHA_KEY: str(
                get_positive_int_setting(raw, TRAIN_NETWORK_ALPHA_KEY, int(global_values[TRAIN_NETWORK_ALPHA_KEY]))
            ),
            TRAIN_OPTIMIZER_TYPE_KEY: (
                raw.get(TRAIN_OPTIMIZER_TYPE_KEY, "").strip().lower()
                if raw.get(TRAIN_OPTIMIZER_TYPE_KEY, "").strip().lower() in {"adamw8bit", "prodigy"}
                else global_values[TRAIN_OPTIMIZER_TYPE_KEY]
            ),
            TRAIN_LEARNING_RATE_KEY: raw.get(TRAIN_LEARNING_RATE_KEY, "").strip() or global_values[TRAIN_LEARNING_RATE_KEY],
            TRAIN_STEPS_KEY: str(get_positive_int_setting(raw, TRAIN_STEPS_KEY, int(global_values[TRAIN_STEPS_KEY]))),
        }

    def dataset_has_train_override(dataset_name: str) -> bool:
        raw = load_dataset_train_settings_raw(dataset_name)
        return not is_truthy(raw.get(DATASET_USE_GLOBAL_TRAIN_KEY), default=True)

    def open_settings_dialog(required: bool) -> KleinRuntimeConfig | None:
        current_dir = ""
        if runtime_config is not None:
            current_dir = str(runtime_config.musubi_dir)
        current_musubi_python = settings_state.get(MUSUBI_PYTHON_KEY, "").strip()
        current_klein_model_version = settings_state.get(KLEIN_MODEL_VERSION_KEY, "").strip() or "klein-base-9b"
        current_klein_dit = settings_state.get(KLEIN_DIT_KEY, "").strip()
        current_klein_vae = settings_state.get(KLEIN_VAE_KEY, "").strip()
        current_klein_text_encoder = settings_state.get(KLEIN_TEXT_ENCODER_KEY, "").strip()
        current_ltx_model_version = settings_state.get(LTX_MODEL_VERSION_KEY, "").strip()
        current_ltx_dit = settings_state.get(LTX_DIT_KEY, "").strip()
        current_ltx_vae = settings_state.get(LTX_VAE_KEY, "").strip()
        current_ltx_text_encoder = settings_state.get(LTX_TEXT_ENCODER_KEY, "").strip()
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
        current_train_resolution = get_positive_int_setting(
            settings_state,
            TRAIN_RESOLUTION_KEY,
            DEFAULT_RESOLUTION,
            minimum=64,
        )
        current_train_network_dim = get_positive_int_setting(
            settings_state,
            TRAIN_NETWORK_DIM_KEY,
            DEFAULT_NETWORK_DIM,
        )
        current_train_network_alpha = get_positive_int_setting(
            settings_state,
            TRAIN_NETWORK_ALPHA_KEY,
            DEFAULT_NETWORK_ALPHA,
        )
        current_train_optimizer = get_train_optimizer_setting(settings_state)
        current_train_learning_rate = get_learning_rate_setting(settings_state)
        current_train_steps = get_positive_int_setting(
            settings_state,
            TRAIN_STEPS_KEY,
            DEFAULT_TRAIN_STEPS,
        )
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
        current_gc_cpu_offload = settings_state.get(
            ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        result: KleinRuntimeConfig | None = None
        dialog = tk.Toplevel(root)
        dialog.title("Settings")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(True, True)
        dialog.configure(bg=bg_panel)
        set_dark_title_bar(dialog)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=0)

        scroll_host = ttk.Frame(dialog)
        scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        settings_canvas = tk.Canvas(
            scroll_host,
            bg=bg_panel,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar = ttk.Scrollbar(scroll_host, orient="vertical", command=settings_canvas.yview)
        settings_scrollbar.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

        frame = ttk.Frame(settings_canvas, padding=10)
        frame_window_id = settings_canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.columnconfigure(0, weight=1)

        footer = ttk.Frame(dialog, padding=(10, 8, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        def sync_settings_scrollregion(_event: tk.Event | None = None) -> None:
            settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

        def sync_settings_canvas_width(_event: tk.Event) -> None:
            settings_canvas.itemconfigure(frame_window_id, width=_event.width)

        def on_settings_mousewheel(event: tk.Event) -> str:
            delta = int(-event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            settings_canvas.yview_scroll(delta, "units")
            return "break"

        def on_settings_linux_up(_event: tk.Event) -> str:
            settings_canvas.yview_scroll(-1, "units")
            return "break"

        def on_settings_linux_down(_event: tk.Event) -> str:
            settings_canvas.yview_scroll(1, "units")
            return "break"

        frame.bind("<Configure>", sync_settings_scrollregion)
        settings_canvas.bind("<Configure>", sync_settings_canvas_width)
        dialog.bind("<MouseWheel>", on_settings_mousewheel)
        dialog.bind("<Button-4>", on_settings_linux_up)
        dialog.bind("<Button-5>", on_settings_linux_down)

        musubi_section = ttk.LabelFrame(frame, text="Musubi-Tuner", padding=8)
        musubi_section.grid(row=0, column=0, sticky="ew")
        musubi_section.columnconfigure(1, weight=1)

        klein_toggle_var = tk.BooleanVar(value=False)
        klein_toggle = ttk.Checkbutton(frame, text="Show Klein settings", variable=klein_toggle_var)
        klein_toggle.grid(row=1, column=0, sticky="w", pady=(10, 0))

        klein_section = ttk.LabelFrame(frame, text="Klein", padding=8)
        klein_section.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        klein_section.columnconfigure(1, weight=1)

        ltx_toggle_var = tk.BooleanVar(value=False)
        ltx_toggle = ttk.Checkbutton(frame, text="Show LTX settings", variable=ltx_toggle_var)
        ltx_toggle.grid(row=3, column=0, sticky="w", pady=(10, 0))

        ltx_section = ttk.LabelFrame(frame, text="LTX", padding=8)
        ltx_section.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ltx_section.columnconfigure(1, weight=1)

        captions_section = ttk.LabelFrame(frame, text="Captions", padding=8)
        captions_section.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        captions_section.columnconfigure(1, weight=1)

        advanced_section = ttk.LabelFrame(frame, text="Training", padding=8)
        advanced_section.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        advanced_section.columnconfigure(0, weight=0)
        advanced_section.columnconfigure(1, weight=1)
        advanced_section.columnconfigure(2, weight=0)
        advanced_section.columnconfigure(3, weight=1)

        selected_musubi_path = current_dir
        selected_musubi_python = current_musubi_python
        selected_klein_dit = current_klein_dit
        selected_klein_vae = current_klein_vae
        selected_klein_text_encoder = current_klein_text_encoder
        selected_ltx_dit = current_ltx_dit
        selected_ltx_vae = current_ltx_vae
        selected_ltx_text_encoder = current_ltx_text_encoder

        musubi_display_var = tk.StringVar(value=current_dir if current_dir else "(none)")
        musubi_python_display_var = tk.StringVar(value=current_musubi_python if current_musubi_python else "(auto)")
        klein_model_version_var = tk.StringVar(value=current_klein_model_version)
        klein_dit_var = tk.StringVar(value=current_klein_dit if current_klein_dit else "(none)")
        klein_vae_var = tk.StringVar(value=current_klein_vae if current_klein_vae else "(none)")
        klein_text_encoder_var = tk.StringVar(value=current_klein_text_encoder if current_klein_text_encoder else "(none)")
        ltx_model_version_var = tk.StringVar(value=current_ltx_model_version)
        ltx_dit_var = tk.StringVar(value=current_ltx_dit if current_ltx_dit else "(none)")
        ltx_vae_var = tk.StringVar(value=current_ltx_vae if current_ltx_vae else "(none)")
        ltx_text_encoder_var = tk.StringVar(value=current_ltx_text_encoder if current_ltx_text_encoder else "(none)")
        default_caption_keyword_var = tk.StringVar(value=current_default_caption_keyword)
        compile_optimizations_var = tk.BooleanVar(value=current_compile_optimizations)
        cuda_allow_tf32_var = tk.BooleanVar(value=current_cuda_allow_tf32)
        cuda_cudnn_benchmark_var = tk.BooleanVar(value=current_cuda_cudnn_benchmark)
        fp8_dit_var = tk.BooleanVar(value=current_fp8_dit)
        gc_cpu_offload_var = tk.BooleanVar(value=current_gc_cpu_offload)
        train_resolution_var = tk.StringVar(value=str(current_train_resolution))
        train_network_dim_var = tk.StringVar(
            value=str(current_train_network_dim) if str(current_train_network_dim) in TRAIN_DIM_ALPHA_CHOICES else "32"
        )
        train_network_alpha_var = tk.StringVar(
            value=str(current_train_network_alpha) if str(current_train_network_alpha) in TRAIN_DIM_ALPHA_CHOICES else "32"
        )
        train_optimizer_var = tk.StringVar(value=current_train_optimizer)
        train_learning_rate_var = tk.StringVar(value=current_train_learning_rate)
        train_steps_var = tk.StringVar(value=str(current_train_steps))
        enable_training_logging_var = tk.BooleanVar(value=current_enable_training_logging)
        train_log_tracker_name_var = tk.StringVar(value=current_train_log_tracker_name)
        stream_to_logger_var = tk.BooleanVar(value=current_train_stream_to_logger)
        auto_start_tensorboard_var = tk.BooleanVar(value=current_auto_start_tensorboard)
        auto_cleanup_states_var = tk.BooleanVar(value=current_auto_cleanup_states)

        ttk.Label(musubi_section, text="Musubi-Tuner folder:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        musubi_display = ttk.Label(
            musubi_section, textvariable=musubi_display_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        musubi_display.grid(row=0, column=1, sticky="ew")
        ttk.Label(musubi_section, text="Python (venv):").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        musubi_python_display = ttk.Label(
            musubi_section,
            textvariable=musubi_python_display_var,
            anchor="w",
            style="PathDisplay.TLabel",
            padding=(6, 4),
        )
        musubi_python_display.grid(row=1, column=1, sticky="ew", pady=(8, 0))

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

        training_settings_section = ttk.LabelFrame(advanced_section, text="Training settings", padding=8)
        training_settings_section.grid(row=0, column=0, columnspan=4, sticky="ew")
        training_settings_section.columnconfigure(0, weight=0)
        training_settings_section.columnconfigure(1, weight=1)
        training_settings_section.columnconfigure(2, weight=0)
        training_settings_section.columnconfigure(3, weight=1)

        ttk.Label(training_settings_section, text="Optimizer type:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        train_optimizer_combo = ttk.Combobox(
            training_settings_section,
            textvariable=train_optimizer_var,
            values=("adamw8bit", "prodigy"),
            state="readonly",
            width=16,
        )
        train_optimizer_combo.grid(row=0, column=1, sticky="ew")
        ttk.Label(training_settings_section, text="Training steps:").grid(row=0, column=2, sticky="w", padx=(12, 8))
        ttk.Entry(training_settings_section, textvariable=train_steps_var, style="Flat.TEntry").grid(row=0, column=3, sticky="ew")

        ttk.Label(training_settings_section, text="Learning rate:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        train_learning_rate_entry = ttk.Entry(training_settings_section, textvariable=train_learning_rate_var, style="Flat.TEntry")
        train_learning_rate_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(training_settings_section, text="LoRA network dim:").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        ttk.Combobox(
            training_settings_section,
            textvariable=train_network_dim_var,
            values=TRAIN_DIM_ALPHA_CHOICES,
            state="readonly",
            width=10,
        ).grid(row=1, column=3, sticky="ew", pady=(6, 0))

        ttk.Label(training_settings_section, text="Dataset resolution:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(training_settings_section, textvariable=train_resolution_var, style="Flat.TEntry").grid(
            row=2,
            column=1,
            sticky="ew",
            pady=(6, 0),
        )
        ttk.Label(training_settings_section, text="LoRA network alpha:").grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        ttk.Combobox(
            training_settings_section,
            textvariable=train_network_alpha_var,
            values=TRAIN_DIM_ALPHA_CHOICES,
            state="readonly",
            width=10,
        ).grid(row=2, column=3, sticky="ew", pady=(6, 0))

        ttk.Checkbutton(
            training_settings_section,
            text="Enable FP8 (Low VRAM)",
            variable=fp8_dit_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            training_settings_section,
            text="Enable CPU Gradient Checkpointing (Low RAM)",
            variable=gc_cpu_offload_var,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(8, 0))

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

        def sync_optimizer_controls() -> None:
            if train_optimizer_var.get() == "prodigy":
                train_learning_rate_var.set("1")
                train_learning_rate_entry.configure(state="disabled")
            else:
                train_learning_rate_entry.configure(state="normal")

        train_optimizer_var.trace_add("write", lambda *_args: sync_optimizer_controls())
        sync_optimizer_controls()

        ttk.Label(klein_section, text="Model version:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        klein_model_version_entry = ttk.Entry(klein_section, textvariable=klein_model_version_var, style="Flat.TEntry")
        klein_model_version_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(klein_section, text="Model:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        klein_dit_display = ttk.Label(
            klein_section, textvariable=klein_dit_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        klein_dit_display.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(klein_section, text="VAE:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        klein_vae_display = ttk.Label(
            klein_section, textvariable=klein_vae_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        klein_vae_display.grid(row=2, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(klein_section, text="Text Encoder:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        klein_text_encoder_display = ttk.Label(
            klein_section, textvariable=klein_text_encoder_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        klein_text_encoder_display.grid(row=3, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(ltx_section, text="Model version:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ltx_model_version_entry = ttk.Entry(ltx_section, textvariable=ltx_model_version_var, style="Flat.TEntry")
        ltx_model_version_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(ltx_section, text="Model:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ltx_dit_display = ttk.Label(
            ltx_section, textvariable=ltx_dit_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        ltx_dit_display.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(ltx_section, text="VAE:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ltx_vae_display = ttk.Label(
            ltx_section, textvariable=ltx_vae_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        ltx_vae_display.grid(row=2, column=1, sticky="ew", pady=(8, 0))

        ttk.Label(ltx_section, text="Text Encoder:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ltx_text_encoder_display = ttk.Label(
            ltx_section, textvariable=ltx_text_encoder_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        ltx_text_encoder_display.grid(row=3, column=1, sticky="ew", pady=(8, 0))

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
                    musubi_python_display_var.set(selected_musubi_python)
                else:
                    selected_musubi_python = ""
                    musubi_python_display_var.set("(not found - set manually)")
                    messagebox.showwarning(
                        "Python venv not found",
                        "Could not find .venv/venv Python in this Musubi-Tuner folder.\n"
                        "Set Python (venv) manually.",
                        parent=dialog,
                    )

        def browse_musubi_python() -> None:
            nonlocal selected_musubi_python
            initial_dir = selected_musubi_path or str(Path.home())
            if selected_musubi_python:
                initial_dir = str(Path(selected_musubi_python).expanduser().parent)

            filetypes = [("Python executable", "python.exe"), ("All files", "*.*")] if sys.platform == "win32" else [("All files", "*.*")]
            picked = filedialog.askopenfilename(
                parent=dialog,
                title="Select Musubi-Tuner Python executable",
                initialdir=initial_dir,
                filetypes=filetypes,
            )
            if picked:
                selected_musubi_python = picked
                musubi_python_display_var.set(picked)

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

        def browse_klein_dit() -> None:
            nonlocal selected_klein_dit
            picked = browse_file(selected_klein_dit, selected_musubi_path, "Select Klein model file")
            if picked:
                selected_klein_dit = picked
                klein_dit_var.set(picked)

        def browse_klein_vae() -> None:
            nonlocal selected_klein_vae
            picked = browse_file(selected_klein_vae, selected_musubi_path, "Select Klein VAE file")
            if picked:
                selected_klein_vae = picked
                klein_vae_var.set(picked)

        def browse_klein_text_encoder() -> None:
            nonlocal selected_klein_text_encoder
            picked = browse_file(
                selected_klein_text_encoder,
                selected_musubi_path,
                "Select Klein text encoder file",
            )
            if picked:
                selected_klein_text_encoder = picked
                klein_text_encoder_var.set(picked)

        def browse_ltx_dit() -> None:
            nonlocal selected_ltx_dit
            picked = browse_file(selected_ltx_dit, selected_musubi_path, "Select LTX model file")
            if picked:
                selected_ltx_dit = picked
                ltx_dit_var.set(picked)

        def browse_ltx_vae() -> None:
            nonlocal selected_ltx_vae
            picked = browse_file(selected_ltx_vae, selected_musubi_path, "Select LTX VAE file")
            if picked:
                selected_ltx_vae = picked
                ltx_vae_var.set(picked)

        def browse_ltx_text_encoder() -> None:
            nonlocal selected_ltx_text_encoder
            picked = browse_file(
                selected_ltx_text_encoder,
                selected_musubi_path,
                "Select LTX text encoder file",
            )
            if picked:
                selected_ltx_text_encoder = picked
                ltx_text_encoder_var.set(picked)

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
            nonlocal result, settings_state, selected_musubi_python
            if not selected_musubi_path:
                messagebox.showerror("Missing folder", "Musubi-Tuner folder is not set.", parent=dialog)
                return

            musubi_path = Path(selected_musubi_path).expanduser()
            if not musubi_path.exists() or not musubi_path.is_dir():
                messagebox.showerror("Invalid folder", "Choose a valid Musubi-Tuner folder.", parent=dialog)
                return

            musubi_python_path: Path | None = None
            if selected_musubi_python:
                musubi_python_path = Path(selected_musubi_python).expanduser()
                if not musubi_python_path.exists() or not musubi_python_path.is_file():
                    messagebox.showerror("Invalid file", "Choose a valid Python (venv) executable.", parent=dialog)
                    return
            else:
                detected = resolve_musubi_python(musubi_path)
                if detected is not None:
                    musubi_python_path = detected
                    selected_musubi_python = str(detected)
                else:
                    messagebox.showwarning(
                        "Python venv not found",
                        "Could not auto-detect Musubi-Tuner venv (.venv/venv).\n"
                        "Set Python (venv) manually in Settings before running jobs.",
                        parent=dialog,
                    )

            klein_model_version = klein_model_version_var.get().strip()
            if not klein_model_version:
                messagebox.showerror("Missing value", "Klein model version is required.", parent=dialog)
                return

            try:
                train_resolution = int(train_resolution_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "Dataset resolution must be an integer.", parent=dialog)
                return
            if train_resolution < 64:
                messagebox.showerror("Invalid value", "Dataset resolution must be 64 or higher.", parent=dialog)
                return

            try:
                train_network_dim = int(train_network_dim_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "LoRA network dim must be an integer.", parent=dialog)
                return
            if train_network_dim < 1:
                messagebox.showerror("Invalid value", "LoRA network dim must be 1 or higher.", parent=dialog)
                return

            try:
                train_network_alpha = int(train_network_alpha_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "LoRA network alpha must be an integer.", parent=dialog)
                return
            if train_network_alpha < 1:
                messagebox.showerror("Invalid value", "LoRA network alpha must be 1 or higher.", parent=dialog)
                return

            train_optimizer = train_optimizer_var.get().strip().lower()
            if train_optimizer not in {"adamw8bit", "prodigy"}:
                messagebox.showerror("Invalid value", "Optimizer type must be adamw8bit or prodigy.", parent=dialog)
                return

            train_learning_rate = train_learning_rate_var.get().strip()
            if train_optimizer == "prodigy":
                train_learning_rate = "1"
                train_learning_rate_var.set("1")
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

            try:
                train_steps = int(train_steps_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "Train steps must be an integer.", parent=dialog)
                return
            if train_steps < 1:
                messagebox.showerror("Invalid value", "Train steps must be 1 or higher.", parent=dialog)
                return

            normalized_klein_dit = normalize_model_checkpoint_path(selected_klein_dit)
            normalized_klein_vae = normalize_model_checkpoint_path(selected_klein_vae)
            normalized_klein_text_encoder = normalize_model_checkpoint_path(selected_klein_text_encoder)
            normalized_ltx_dit = normalize_model_checkpoint_path(selected_ltx_dit)
            normalized_ltx_vae = normalize_model_checkpoint_path(selected_ltx_vae)
            normalized_ltx_text_encoder = normalize_model_checkpoint_path(selected_ltx_text_encoder)

            for label, raw_path in (
                ("Klein Model", normalized_klein_dit),
                ("Klein VAE", normalized_klein_vae),
                ("Klein Text Encoder", normalized_klein_text_encoder),
            ):
                if not raw_path:
                    continue
                resolved = Path(raw_path).expanduser()
                if not resolved.exists() or not resolved.is_file():
                    messagebox.showerror("Invalid file", f"Choose a valid file for {label}.", parent=dialog)
                    return

            for label, raw_path in (
                ("LTX Model", normalized_ltx_dit),
                ("LTX VAE", normalized_ltx_vae),
                ("LTX Text Encoder", normalized_ltx_text_encoder),
            ):
                if not raw_path:
                    continue
                resolved = Path(raw_path).expanduser()
                if not resolved.exists() or not resolved.is_file():
                    messagebox.showerror("Invalid file", f"Choose a valid file for {label}.", parent=dialog)
                    return

            settings_state[MUSUBI_DIR_KEY] = str(musubi_path)
            settings_state[MUSUBI_PYTHON_KEY] = str(musubi_python_path) if musubi_python_path is not None else ""
            settings_state[KLEIN_MODEL_VERSION_KEY] = klein_model_version
            settings_state[KLEIN_DIT_KEY] = normalized_klein_dit
            settings_state[KLEIN_VAE_KEY] = normalized_klein_vae
            settings_state[KLEIN_TEXT_ENCODER_KEY] = normalized_klein_text_encoder
            settings_state[LTX_MODEL_VERSION_KEY] = ltx_model_version_var.get().strip()
            settings_state[LTX_DIT_KEY] = normalized_ltx_dit
            settings_state[LTX_VAE_KEY] = normalized_ltx_vae
            settings_state[LTX_TEXT_ENCODER_KEY] = normalized_ltx_text_encoder
            settings_state[DEFAULT_CAPTION_KEYWORD_KEY] = default_caption_keyword_var.get().strip()
            settings_state[ENABLE_COMPILE_OPTIMIZATIONS_KEY] = "1" if compile_optimizations_var.get() else "0"
            settings_state[ENABLE_CUDA_ALLOW_TF32_KEY] = "1" if cuda_allow_tf32_var.get() else "0"
            settings_state[ENABLE_CUDA_CUDNN_BENCHMARK_KEY] = "1" if cuda_cudnn_benchmark_var.get() else "0"
            settings_state[ENABLE_FP8_DIT_KEY] = "1" if fp8_dit_var.get() else "0"
            settings_state[ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY] = "1" if gc_cpu_offload_var.get() else "0"
            settings_state[TRAIN_RESOLUTION_KEY] = str(train_resolution)
            settings_state[TRAIN_NETWORK_DIM_KEY] = str(train_network_dim)
            settings_state[TRAIN_NETWORK_ALPHA_KEY] = str(train_network_alpha)
            settings_state[TRAIN_OPTIMIZER_TYPE_KEY] = train_optimizer
            settings_state[TRAIN_LEARNING_RATE_KEY] = train_learning_rate
            settings_state[TRAIN_STEPS_KEY] = str(train_steps)
            settings_state[TRAIN_ENABLE_LOGGING_KEY] = "1" if enable_training_logging_var.get() else "0"
            settings_state[TRAIN_LOG_BACKEND_KEY] = "tensorboard"
            settings_state[TRAIN_LOG_TRACKER_NAME_KEY] = train_log_tracker_name_var.get().strip()
            settings_state[TRAIN_STREAM_TO_LOGGER_KEY] = "1" if stream_to_logger_var.get() else "0"
            settings_state[TRAIN_AUTO_START_TENSORBOARD_KEY] = "1" if auto_start_tensorboard_var.get() else "0"
            settings_state[TRAIN_AUTO_CLEANUP_STATES_KEY] = "1" if auto_cleanup_states_var.get() else "0"
            save_settings(settings_state)
            result = klein_runtime_config_from_settings(settings_state)
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
        ttk.Button(musubi_section, text="Browse File", command=browse_musubi_python).grid(
            row=1, column=2, padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(klein_section, text="Browse File", command=browse_klein_dit).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Button(klein_section, text="Browse File", command=browse_klein_vae).grid(row=2, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Button(klein_section, text="Browse File", command=browse_klein_text_encoder).grid(
            row=3, column=2, padx=(8, 0), pady=(8, 0)
        )

        ttk.Button(ltx_section, text="Browse File", command=browse_ltx_dit).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Button(ltx_section, text="Browse File", command=browse_ltx_vae).grid(row=2, column=2, padx=(8, 0), pady=(8, 0))
        ttk.Button(ltx_section, text="Browse File", command=browse_ltx_text_encoder).grid(
            row=3, column=2, padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(frame, text="LTX path is saved for future support.").grid(
            row=7, column=0, sticky="w", pady=(10, 8)
        )

        button_row = ttk.Frame(footer)
        button_row.grid(row=0, column=0, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        ttk.Button(button_row, text="Reset Settings", command=reset_settings).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Cancel", command=cancel_and_close).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=2)

        def sync_model_sections() -> None:
            if klein_toggle_var.get():
                klein_section.grid()
                klein_toggle.configure(text="Hide Klein settings")
            else:
                klein_section.grid_remove()
                klein_toggle.configure(text="Show Klein settings")

            if ltx_toggle_var.get():
                ltx_section.grid()
                ltx_toggle.configure(text="Hide LTX settings")
            else:
                ltx_section.grid_remove()
                ltx_toggle.configure(text="Show LTX settings")

        klein_toggle_var.trace_add("write", lambda *_args: sync_model_sections())
        ltx_toggle_var.trace_add("write", lambda *_args: sync_model_sections())
        sync_model_sections()

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.update_idletasks()
        sync_settings_scrollregion()

        content_w = max(frame.winfo_reqwidth(), footer.winfo_reqwidth()) + 44
        content_h = frame.winfo_reqheight() + 20
        max_w = max(760, dialog.winfo_screenwidth() - 80)
        max_h = max(480, dialog.winfo_screenheight() - 80)
        win_w = max(780, min(1080, content_w))
        win_w = min(win_w, max_w)
        win_h = min(max(520, content_h), max_h)
        dialog.geometry(f"{win_w}x{win_h}")
        settings_canvas.yview_moveto(0.0)
        center_window(dialog)
        dialog.focus_set()
        root.wait_window(dialog)

        if required and result is None:
            return None

        return result

    if runtime_config is None:
        messagebox.showinfo(
            "First launch setup",
            "Musubi-Tuner location is required before this app can run. Set it in Settings now.",
            parent=root,
        )
        runtime_config = open_settings_dialog(required=True)
        if runtime_config is None:
            root.destroy()
            return 1

    maybe_autostart_tensorboard()

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=0)
    root.rowconfigure(2, weight=5, minsize=360)
    root.rowconfigure(3, weight=0)
    root.rowconfigure(4, weight=0)
    root.rowconfigure(5, weight=0)
    root.rowconfigure(6, weight=2)

    header = ttk.Frame(root, padding=8)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    training_path_var = tk.StringVar(value=f"Training folder: {runtime_config.training_dir}")
    ttk.Label(header, textvariable=training_path_var).grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.grid(row=1, column=0, sticky="ew")
    controls.columnconfigure(1, weight=1)

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

    def open_dataset_config_dialog(dataset_name: str) -> None:
        effective = effective_train_settings_for_dataset(dataset_name)
        global_values = global_train_settings()
        image_count = len(dataset_image_files(datasets_root_dir(), dataset_name))

        dialog = tk.Toplevel(root)
        dialog.title(f"Dataset Configure - {dataset_name}")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=bg_panel)
        set_dark_title_bar(dialog)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        frame = ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        section = ttk.LabelFrame(frame, text=f"Train settings for {dataset_name}", padding=8)
        section.grid(row=0, column=0, sticky="ew")
        section.columnconfigure(0, weight=0)
        section.columnconfigure(1, weight=1)
        section.columnconfigure(2, weight=0)
        section.columnconfigure(3, weight=1)

        use_global_var = tk.BooleanVar(value=is_truthy(effective.get(DATASET_USE_GLOBAL_TRAIN_KEY), default=True))
        train_optimizer_var = tk.StringVar(value=effective[TRAIN_OPTIMIZER_TYPE_KEY])
        train_steps_var = tk.StringVar(value=effective[TRAIN_STEPS_KEY])
        train_learning_rate_var = tk.StringVar(value=effective[TRAIN_LEARNING_RATE_KEY])
        train_network_dim_var = tk.StringVar(
            value=effective[TRAIN_NETWORK_DIM_KEY]
            if effective[TRAIN_NETWORK_DIM_KEY] in TRAIN_DIM_ALPHA_CHOICES
            else "32"
        )
        train_network_alpha_var = tk.StringVar(
            value=effective[TRAIN_NETWORK_ALPHA_KEY]
            if effective[TRAIN_NETWORK_ALPHA_KEY] in TRAIN_DIM_ALPHA_CHOICES
            else "32"
        )
        train_resolution_var = tk.StringVar(value=effective[TRAIN_RESOLUTION_KEY])

        ttk.Checkbutton(
            section,
            text="Use global values",
            variable=use_global_var,
        ).grid(row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(section, text="Optimizer type:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        train_optimizer_combo = ttk.Combobox(
            section,
            textvariable=train_optimizer_var,
            values=("adamw8bit", "prodigy"),
            state="readonly",
            width=14,
        )
        train_optimizer_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(section, text="Training steps:").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(8, 0))
        train_steps_entry = ttk.Entry(section, textvariable=train_steps_var, style="Flat.TEntry")
        train_steps_entry.grid(row=1, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(section, text="Learning rate:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        train_learning_rate_entry = ttk.Entry(section, textvariable=train_learning_rate_var, style="Flat.TEntry")
        train_learning_rate_entry.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(section, text="LoRA network dim:").grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(8, 0))
        train_network_dim_combo = ttk.Combobox(
            section,
            textvariable=train_network_dim_var,
            values=TRAIN_DIM_ALPHA_CHOICES,
            state="readonly",
            width=10,
        )
        train_network_dim_combo.grid(row=2, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(section, text="Dataset resolution:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        train_resolution_entry = ttk.Entry(section, textvariable=train_resolution_var, style="Flat.TEntry")
        train_resolution_entry.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(section, text="LoRA network alpha:").grid(row=3, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        train_network_alpha_combo = ttk.Combobox(
            section,
            textvariable=train_network_alpha_var,
            values=TRAIN_DIM_ALPHA_CHOICES,
            state="readonly",
            width=10,
        )
        train_network_alpha_combo.grid(row=3, column=3, sticky="ew", pady=(6, 0))

        help_label = ttk.Label(
            section,
            text="When Use global values is on, this dataset follows Settings > Advanced values.",
        )
        help_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(section, text=f"Images found: {image_count}").grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))

        def sync_optimizer_controls() -> None:
            if train_optimizer_var.get() == "prodigy":
                train_learning_rate_var.set("1")
                train_learning_rate_entry.configure(state="disabled")
            else:
                train_learning_rate_entry.configure(state="normal")

        def sync_entry_state() -> None:
            if use_global_var.get():
                train_optimizer_combo.configure(state="disabled")
                train_steps_entry.configure(state="disabled")
                train_learning_rate_entry.configure(state="disabled")
                train_network_dim_combo.configure(state="disabled")
                train_network_alpha_combo.configure(state="disabled")
                train_resolution_entry.configure(state="disabled")
                return

            train_optimizer_combo.configure(state="readonly")
            train_steps_entry.configure(state="normal")
            train_network_dim_combo.configure(state="readonly")
            train_network_alpha_combo.configure(state="readonly")
            train_resolution_entry.configure(state="normal")
            sync_optimizer_controls()

        def reset_to_global_defaults() -> None:
            train_optimizer_var.set(global_values[TRAIN_OPTIMIZER_TYPE_KEY])
            train_steps_var.set(global_values[TRAIN_STEPS_KEY])
            train_learning_rate_var.set(global_values[TRAIN_LEARNING_RATE_KEY])
            train_network_dim_var.set(global_values[TRAIN_NETWORK_DIM_KEY])
            train_network_alpha_var.set(global_values[TRAIN_NETWORK_ALPHA_KEY])
            train_resolution_var.set(global_values[TRAIN_RESOLUTION_KEY])
            use_global_var.set(False)
            sync_entry_state()

        def save_and_close() -> None:
            raw_to_save: dict[str, str] = {
                DATASET_USE_GLOBAL_TRAIN_KEY: "1" if use_global_var.get() else "0",
            }

            if not use_global_var.get():
                try:
                    train_steps = int(train_steps_var.get().strip())
                except ValueError:
                    messagebox.showerror("Invalid value", "Training steps must be an integer.", parent=dialog)
                    return
                if train_steps < 1:
                    messagebox.showerror("Invalid value", "Training steps must be 1 or higher.", parent=dialog)
                    return

                train_optimizer = train_optimizer_var.get().strip().lower()
                if train_optimizer not in {"adamw8bit", "prodigy"}:
                    messagebox.showerror("Invalid value", "Optimizer type must be adamw8bit or prodigy.", parent=dialog)
                    return

                train_learning_rate = train_learning_rate_var.get().strip()
                if train_optimizer == "prodigy":
                    train_learning_rate = "1"
                    train_learning_rate_var.set("1")
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

                try:
                    network_dim = int(train_network_dim_var.get().strip())
                except ValueError:
                    messagebox.showerror("Invalid value", "LoRA network dim must be an integer.", parent=dialog)
                    return
                if network_dim < 1:
                    messagebox.showerror("Invalid value", "LoRA network dim must be 1 or higher.", parent=dialog)
                    return

                try:
                    network_alpha = int(train_network_alpha_var.get().strip())
                except ValueError:
                    messagebox.showerror("Invalid value", "LoRA network alpha must be an integer.", parent=dialog)
                    return
                if network_alpha < 1:
                    messagebox.showerror("Invalid value", "LoRA network alpha must be 1 or higher.", parent=dialog)
                    return

                try:
                    resolution = int(train_resolution_var.get().strip())
                except ValueError:
                    messagebox.showerror("Invalid value", "Dataset resolution must be an integer.", parent=dialog)
                    return
                if resolution < 64:
                    messagebox.showerror("Invalid value", "Dataset resolution must be 64 or higher.", parent=dialog)
                    return

                raw_to_save[TRAIN_STEPS_KEY] = str(train_steps)
                raw_to_save[TRAIN_OPTIMIZER_TYPE_KEY] = train_optimizer
                raw_to_save[TRAIN_LEARNING_RATE_KEY] = train_learning_rate
                raw_to_save[TRAIN_NETWORK_DIM_KEY] = str(network_dim)
                raw_to_save[TRAIN_NETWORK_ALPHA_KEY] = str(network_alpha)
                raw_to_save[TRAIN_RESOLUTION_KEY] = str(resolution)

            try:
                save_dataset_train_settings_raw(dataset_name, raw_to_save)
            except OSError as exc:
                messagebox.showerror("Save failed", f"Could not save dataset settings:\n{exc}", parent=dialog)
                return

            checkpoint_cache.pop(dataset_name, None)
            rebuild_folder_list(force=True)
            dialog.destroy()

        button_row = ttk.Frame(frame)
        button_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        button_row.columnconfigure(0, weight=1)

        ttk.Button(button_row, text="Reset", command=reset_to_global_defaults).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=2)

        use_global_var.trace_add("write", lambda *_args: sync_entry_state())
        train_optimizer_var.trace_add("write", lambda *_args: sync_entry_state())
        sync_entry_state()

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        dialog.geometry(f"{max(740, dialog.winfo_reqwidth())}x{dialog.winfo_reqheight()}")
        center_window(dialog)
        dialog.focus_set()
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
        dialog.focus_set()
        selected_list.focus_set()
        root.wait_window(dialog)
        return choice

    def lora_post_hoc_ema_merge(dataset_name: str) -> None:
        output_dir = runtime_config.training_dir / dataset_name / "output"
        lora_post_hoc_ema_merge_for_output(dataset_name, output_dir)

    def lora_post_hoc_ema_merge_for_output(target_name: str, output_dir: Path) -> None:
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
                output_dir,
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
                "Musubi-Tuner Python was not found. Open Settings and set Python (venv).",
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

        dialog.update_idletasks()
        dialog.geometry("820x620")
        center_window(dialog)
        root.wait_window(dialog)

    def show_thumbnail_context_menu(event: tk.Event, dataset_name: str) -> str:
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Open Dataset", command=lambda: open_dataset_in_file_manager(dataset_name))
        menu.add_command(label="Add Images", command=lambda: add_images_to_dataset(dataset_name))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    list_container = ttk.LabelFrame(root, text="Datasets (click thumbnail to select)", padding=8)
    list_container.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 0))
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

    dataset_actions_bar = ttk.Frame(root, padding=(8, 8, 8, 0))
    dataset_actions_bar.grid(row=3, column=0, sticky="ew")
    dataset_actions_bar.columnconfigure(0, weight=1)

    start_bar = ttk.Frame(root, padding=(8, 0, 8, 8))
    start_bar.grid(row=5, column=0, sticky="ew")
    start_bar.columnconfigure(0, weight=1)

    queue_container = ttk.LabelFrame(root, text="Queue", padding=8)
    queue_container.grid(row=4, column=0, sticky="ew", padx=8, pady=(8, 0))
    queue_container.columnconfigure(0, weight=1)
    queue_container.rowconfigure(0, weight=1)

    queue_table_border = tk.Frame(queue_container, bg="#2a4a72", bd=0, highlightthickness=0)
    queue_table_border.grid(row=0, column=0, columnspan=2, sticky="nsew")
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

    log_container = ttk.Frame(root)
    log_container.grid(row=6, column=0, sticky="nsew", padx=8, pady=(0, 8))
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

    def job_settings_file_path(training_name: str) -> Path:
        return training_job_dir_path(training_name) / JOB_SETTINGS_FILE_NAME

    def ensure_jobs_storage() -> None:
        jobs_storage_dir().mkdir(parents=True, exist_ok=True)

    def save_job_order() -> None:
        ensure_jobs_storage()
        order = [job.get("id", "") for job in job_queue if job.get("id", "").strip()]
        jobs_order_file_path().write_text(json.dumps(order, indent=2), encoding="utf-8")

    def save_job_to_disk(job: dict[str, str]) -> None:
        ensure_jobs_storage()
        job_id = job.get("id", "").strip()
        if not job_id:
            return
        training_name = job.get("training_name", "").strip() or job.get("job_name", "").strip()
        if not training_name:
            return
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

    def make_job_id() -> str:
        return uuid.uuid4().hex[:12]

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
        output_dir = Path(job.get("output_dir", "")).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            return 0, False

        output_names = job_output_names(job)
        if not output_names:
            return 0, False

        latest_step = 0
        has_resume_state = False
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
        train_settings = global_train_settings()
        training_name = job_dir.name.strip()
        dataset_name = infer_dataset_name_from_training_dir(job_dir).strip() or training_name
        output_dir = (job_dir / "output").expanduser()
        return {
            "id": make_job_id(),
            "dataset_name": dataset_name,
            "training_name": training_name,
            "training_dir": str(job_dir),
            "job_name": training_name,
            "model": "Klein",
            "output_dir": str(output_dir),
            "resolution": train_settings[TRAIN_RESOLUTION_KEY],
            "network_dim": train_settings[TRAIN_NETWORK_DIM_KEY],
            "network_alpha": train_settings[TRAIN_NETWORK_ALPHA_KEY],
            "optimizer_type": train_settings[TRAIN_OPTIMIZER_TYPE_KEY],
            "learning_rate": train_settings[TRAIN_LEARNING_RATE_KEY],
            "train_steps": train_settings[TRAIN_STEPS_KEY],
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
        has_finished_output = any((output_dir / f"{name}.safetensors").exists() for name in output_names)
        if has_finished_output:
            return "done"

        global_train_steps = get_positive_int_setting(settings_state, TRAIN_STEPS_KEY, DEFAULT_TRAIN_STEPS)
        target_steps = get_positive_int_setting(job, "train_steps", global_train_steps)
        progress_step, has_resume_data = job_resume_progress(job)
        if has_resume_data and progress_step >= target_steps:
            return "done"

        if hold:
            return "paused"

        status = job.get("status", "queued").strip().lower()
        if status == "running":
            return "running" if run_in_progress else "queued"
        if status == "cancelled":
            return "resume"
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

            job_id = normalized.get("id", "").strip() or make_job_id()
            normalized["id"] = job_id
            normalized["training_name"] = training_name
            normalized["training_dir"] = str(child)
            normalized["output_dir"] = str((child / "output").expanduser())
            if not normalized.get("job_name", "").strip():
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
                        else ("RESUME" if status == "resume" else status.upper())
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

        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Open Output Folder", command=lambda: open_job_output_folder(clicked))
        menu.add_command(label="LoRA Post-Hoc EMA Merge", command=lambda: merge_job_output_loras(clicked))
        menu.add_command(label="Duplicate Job", command=lambda: duplicate_job(clicked))
        menu.add_command(label="Edit Job", command=lambda: open_create_job_dialog(existing_job=job_queue[clicked]))
        menu.add_separator()
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
        lora_post_hoc_ema_merge_for_output(job_name, output_dir)

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
        try:
            training_dir_path, output_dir_path, created_captions = ensure_training_job_structure(
                training_name=new_name,
                dataset_name=source_dataset,
                resolution=resolution_value,
                default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
            )
        except Exception as exc:
            messagebox.showerror("Duplicate failed", str(exc), parent=root)
            return

        duplicate = {
            **source,
            "id": make_job_id(),
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
                messagebox.showinfo("No source dataset selected", "Select one source dataset card first.", parent=root)
                return
            if len(selected) > 1:
                messagebox.showinfo("Select one source dataset", "Select exactly one source dataset to create a job.", parent=root)
                return
            dataset_name = selected[0]
        else:
            dataset_name = existing_job.get("dataset_name", "").strip()
            if not dataset_name:
                messagebox.showerror("Invalid job", "Job has no dataset name.", parent=root)
                return

        effective = effective_train_settings_for_dataset(dataset_name)

        dialog = tk.Toplevel(root)
        dialog.title("Edit Job" if existing_job is not None else "Create Job")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(False, False)
        set_dark_title_bar(dialog)

        frame = ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        default_job_name = f"{dataset_name}_Klein"
        if existing_job is not None:
            default_job_name = existing_job.get("job_name", default_job_name)
        job_name_var = tk.StringVar(value=default_job_name)
        model_var = tk.StringVar(value=(existing_job or {}).get("model", "Klein"))
        train_optimizer_var = tk.StringVar(value=(existing_job or {}).get("optimizer_type", get_train_optimizer_setting(effective)))
        train_learning_rate_var = tk.StringVar(
            value=(existing_job or {}).get("learning_rate", effective.get(TRAIN_LEARNING_RATE_KEY, DEFAULT_LEARNING_RATE))
        )
        train_steps_var = tk.StringVar(value=(existing_job or {}).get("train_steps", effective.get(TRAIN_STEPS_KEY, str(DEFAULT_TRAIN_STEPS))))
        train_resolution_var = tk.StringVar(
            value=(existing_job or {}).get("resolution", effective.get(TRAIN_RESOLUTION_KEY, str(DEFAULT_RESOLUTION)))
        )
        train_network_dim_var = tk.StringVar(
            value=(existing_job or {}).get("network_dim", effective.get(TRAIN_NETWORK_DIM_KEY, str(DEFAULT_NETWORK_DIM)))
        )
        train_network_alpha_var = tk.StringVar(
            value=(existing_job or {}).get("network_alpha", effective.get(TRAIN_NETWORK_ALPHA_KEY, str(DEFAULT_NETWORK_ALPHA)))
        )

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
        ttk.Label(frame, text="Source dataset:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(frame, text=dataset_name, style="PathDisplay.TLabel", anchor="w", padding=(6, 4)).grid(
            row=0,
            column=1,
            sticky="ew",
        )

        ttk.Label(frame, text="Model:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Combobox(frame, textvariable=model_var, values=("Klein",), state="readonly", width=14).grid(
            row=1,
            column=1,
            sticky="w",
            pady=(8, 0),
        )

        ttk.Label(frame, text="LoRA name:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(frame, textvariable=job_name_var, style="Flat.TEntry").grid(row=2, column=1, sticky="ew", pady=(8, 0))

        options = ttk.LabelFrame(frame, text="Training settings", padding=8)
        options.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Optimizer type:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        train_optimizer_combo = ttk.Combobox(
            options,
            textvariable=train_optimizer_var,
            values=("adamw8bit", "prodigy"),
            state="readonly",
        )
        train_optimizer_combo.grid(
            row=0,
            column=1,
            sticky="ew",
        )
        ttk.Label(options, text="Training steps:").grid(row=0, column=2, sticky="w", padx=(12, 8))
        ttk.Entry(options, textvariable=train_steps_var, style="Flat.TEntry").grid(row=0, column=3, sticky="ew")

        ttk.Label(options, text="Learning rate:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        train_learning_rate_entry = ttk.Entry(options, textvariable=train_learning_rate_var, style="Flat.TEntry")
        train_learning_rate_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(options, text="LoRA network dim:").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        ttk.Combobox(options, textvariable=train_network_dim_var, values=TRAIN_DIM_ALPHA_CHOICES, state="readonly").grid(
            row=1,
            column=3,
            sticky="ew",
            pady=(6, 0),
        )
        ttk.Label(options, text="Dataset resolution:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(options, textvariable=train_resolution_var, style="Flat.TEntry").grid(row=2, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(options, text="LoRA network alpha:").grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        ttk.Combobox(options, textvariable=train_network_alpha_var, values=TRAIN_DIM_ALPHA_CHOICES, state="readonly").grid(
            row=2,
            column=3,
            sticky="ew",
            pady=(6, 0),
        )

        flags = ttk.LabelFrame(options, text="Advanced flags", padding=8)
        flags.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(flags, text="Enable Torch Compile", variable=compile_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(flags, text="Enable Allow TF32", variable=tf32_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(flags, text="Enable cuDNN Benchmark", variable=cudnn_var).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Checkbutton(flags, text="Enable FP8 (Low VRAM)", variable=fp8_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(flags, text="Enable CPU Gradient Checkpointing (Low RAM)", variable=gc_var).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="w",
            padx=(12, 0),
            pady=(6, 0),
        )

        def sync_optimizer_controls() -> None:
            if train_optimizer_var.get().strip().lower() == "prodigy":
                train_learning_rate_var.set("1")
                train_learning_rate_entry.configure(state="disabled")
            else:
                train_learning_rate_entry.configure(state="normal")

        train_optimizer_var.trace_add("write", lambda *_args: sync_optimizer_controls())
        sync_optimizer_controls()

        def create_job() -> None:
            job_name = job_name_var.get().strip()
            if not job_name:
                messagebox.showerror("Missing value", "LoRA name is required.", parent=dialog)
                return
            if not DATASET_NAME_PATTERN.fullmatch(job_name):
                messagebox.showerror(
                    "Invalid name",
                    "Use only letters, numbers, '_' or '-' for LoRA name.",
                    parent=dialog,
                )
                return

            try:
                _ = int(train_steps_var.get().strip())
                _ = int(train_resolution_var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid value", "Steps and Resolution must be numeric.", parent=dialog)
                return

            train_optimizer = train_optimizer_var.get().strip().lower()
            if train_optimizer not in {"adamw8bit", "prodigy"}:
                messagebox.showerror("Invalid value", "Optimizer type must be adamw8bit or prodigy.", parent=dialog)
                return

            train_learning_rate = train_learning_rate_var.get().strip()
            if train_optimizer == "prodigy":
                train_learning_rate = "1"
                train_learning_rate_var.set("1")
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

            if existing_job is None:
                training_name = job_name
            else:
                training_name = (existing_job or {}).get("training_name", "").strip() or job_name

            try:
                resolution_value = get_positive_int_setting({"resolution": train_resolution_var.get().strip()}, "resolution", DEFAULT_RESOLUTION, minimum=64)
                training_dir_path, output_root, created_captions = ensure_training_job_structure(
                    training_name=training_name,
                    dataset_name=dataset_name,
                    resolution=resolution_value,
                    default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                )
            except Exception as exc:
                messagebox.showerror("Create job failed", str(exc), parent=dialog)
                return

            tracker_name = settings_state.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip() or job_name

            new_job = {
                "id": (existing_job or {}).get("id", "") or make_job_id(),
                "dataset_name": dataset_name,
                "training_name": training_name,
                "training_dir": str(training_dir_path),
                "job_name": job_name,
                "model": model_var.get().strip() or "Klein",
                "output_dir": str(output_root),
                "resolution": train_resolution_var.get().strip(),
                "network_dim": train_network_dim_var.get().strip(),
                "network_alpha": train_network_alpha_var.get().strip(),
                "optimizer_type": train_optimizer,
                "learning_rate": train_learning_rate,
                "train_steps": train_steps_var.get().strip(),
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
                "status": "queued" if existing_job is None else detect_job_status({**(existing_job or {}), "job_name": job_name, "output_dir": str(output_root)}),
            }

            if existing_job is None:
                job_queue.append(new_job)
            else:
                if existing_index is None:
                    job_queue.append(new_job)
                else:
                    job_queue[existing_index] = new_job

            save_job_to_disk(new_job)
            save_job_order()
            refresh_job_queue_list()
            update_start_button_state()
            if existing_job is None:
                log(
                    f"[Queue] Created job: {job_name} (source dataset: {dataset_name}, training: {training_name}, captions added: {created_captions})"
                )
            else:
                log(
                    f"[Queue] Updated job: {job_name} (source dataset: {dataset_name}, training: {training_name}, captions added: {created_captions})"
                )
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Save" if existing_job is not None else "Create Job", command=create_job).grid(row=0, column=1)

        dialog.update_idletasks()
        center_window(dialog)
        root.wait_window(dialog)

    def first_image_path(dataset_name: str) -> Path | None:
        cached = first_image_cache.get(dataset_name)
        if dataset_name in first_image_cache:
            return cached

        image_candidates = dataset_image_files(datasets_root_dir(), dataset_name)
        chosen = image_candidates[0] if image_candidates else None
        first_image_cache[dataset_name] = chosen
        return chosen

    def make_thumbnail(image_path: Path | None, run_state: str, thumb_px: int, has_override: bool = False) -> ImageTk.PhotoImage:
        thumb_size = (thumb_px, thumb_px)
        cache_path = "__none__"
        cache_mtime_ns = 0
        if image_path is not None:
            cache_path = str(image_path)
            try:
                cache_mtime_ns = image_path.stat().st_mtime_ns
            except OSError:
                cache_mtime_ns = 0

        cache_key = (cache_path, run_state, thumb_px, cache_mtime_ns, has_override)
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

        draw = ImageDraw.Draw(image)
        if run_state == "in_progress":
            draw.rectangle((1, 1, thumb_px - 2, thumb_px - 2), outline="#f5b301", width=2)

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
        if has_override:
            badge = _load_badge_pil("settings")
            if badge is not None:
                image.paste(badge, (badge_margin, badge_margin), badge)

        if run_state == "done":
            badge = _load_badge_pil("ok")
            if badge is not None:
                image.paste(badge, (thumb_px - badge_margin - badge_size, badge_margin), badge)
        elif run_state == "in_progress":
            badge = _load_badge_pil("pause")
            if badge is not None:
                image.paste(badge, (thumb_px - badge_margin - badge_size, badge_margin), badge)

        photo = ImageTk.PhotoImage(image)
        thumbnail_cache[cache_key] = photo
        return photo

    def toggle_dataset(name: str) -> None:
        if run_state_by_name.get(name) == "done":
            return
        should_select = not vars_by_name[name].get()
        for dataset_name, var in vars_by_name.items():
            var.set(dataset_name == name and should_select)
            apply_card_style(dataset_name)

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
            dataset_train_settings_cache.clear()

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
            dataset_train_settings_cache.pop(stale_name, None)

        if not names:
            empty_label = ttk.Label(inner, text="No datasets found.")
            empty_label.grid(row=0, column=0, sticky="w")
            card_widgets.append(empty_label)
            update_start_button_state()
            return

        canvas_width = canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = max(1, list_container.winfo_width() - 12)
        if canvas_width <= 1:
            return

        gap = ui_config["card_gap"]
        card_width = ui_config["card_width"]
        thumb_px = ui_config["thumbnail_size"]
        card_height = ui_config["card_height"]
        columns = max(1, (canvas_width - gap) // (card_width + gap))

        for col in range(max(1, len(names))):
            inner.columnconfigure(col, minsize=0, weight=0)
        for col in range(columns):
            inner.columnconfigure(col, minsize=card_width + gap, weight=0)

        for idx, name in enumerate(names):
            effective_train = effective_train_settings_for_dataset(name)
            has_train_override = dataset_has_train_override(name)
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
            thumb = make_thumbnail(image_path, train_state, thumb_px, has_override=has_train_override)
            card_thumb_by_name[name] = thumb

            title_style = "DoneCardTitle.TLabel" if train_state == "done" else "CardTitle.TLabel"
            meta_style = "DoneCardMeta.TLabel" if train_state == "done" else "CardMeta.TLabel"

            image_label = ttk.Label(card, image=thumb, style=title_style, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")

            title_label = ttk.Label(card, text=name, style=title_style, anchor="center")
            title_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            image_count = len(dataset_image_files(datasets_root_dir(), name))
            resolution_text = effective_train.get(TRAIN_RESOLUTION_KEY, str(DEFAULT_RESOLUTION))
            status_text = f"{image_count} IMG / {resolution_text}px"
            status_label = ttk.Label(card, text=status_text, style=meta_style, anchor="center")
            status_label.grid(row=2, column=0, sticky="ew", pady=(2, 8))

            click_targets: list[tk.Widget] = [card, image_label, title_label, status_label]

            for clickable in click_targets:
                clickable.bind("<ButtonPress-1>", lambda _e, n=name: on_card_press(n))
                clickable.bind("<B1-Motion>", lambda _e: on_card_motion())
                clickable.bind("<ButtonRelease-1>", lambda _e, n=name: on_card_release(n))
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

                    if job.get("model", "Klein") != "Klein":
                        log(f"[Queue] Unsupported model '{job.get('model')}' for job {job_name}.")
                        exit_code = 1
                    else:
                        training_name = job.get("training_name", "").strip() or job_name
                        dataset_source_name = job.get("dataset_name", "").strip()
                        resolution_value = get_positive_int_setting(job, "resolution", DEFAULT_RESOLUTION, minimum=64)
                        job_error_details = ""

                        try:
                            _training_dir, output_dir, captions_added = ensure_training_job_structure(
                                training_name=training_name,
                                dataset_name=dataset_source_name,
                                resolution=resolution_value,
                                default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                            )
                            if captions_added > 0:
                                log(f"[Queue] Added {captions_added} missing caption file(s) for source dataset '{dataset_source_name}'.")
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

                            exit_code = run_job(
                                runtime_config,
                                dataset_name=training_name,
                                output_name=job_name,
                                output_dir=output_dir,
                                default_caption_keyword=settings_state.get(DEFAULT_CAPTION_KEYWORD_KEY, ""),
                                resolution=resolution_value,
                                network_dim=get_positive_int_setting(job, "network_dim", DEFAULT_NETWORK_DIM),
                                network_alpha=get_positive_int_setting(job, "network_alpha", DEFAULT_NETWORK_ALPHA),
                                optimizer_type=job.get("optimizer_type", "prodigy"),
                                learning_rate=job.get("learning_rate", DEFAULT_LEARNING_RATE),
                                train_steps=get_positive_int_setting(job, "train_steps", DEFAULT_TRAIN_STEPS),
                                enable_compile_optimizations=flag_to_bool(job.get("enable_compile", "0")),
                                enable_cuda_allow_tf32=flag_to_bool(job.get("enable_tf32", "1")),
                                enable_cuda_cudnn_benchmark=flag_to_bool(job.get("enable_cudnn", "1")),
                                enable_fp8_dit=flag_to_bool(job.get("enable_fp8", "0")),
                                enable_gradient_checkpointing_cpu_offload=flag_to_bool(job.get("enable_gc", "0")),
                                enable_training_logging=is_truthy(settings_state.get(TRAIN_ENABLE_LOGGING_KEY), default=True),
                                training_log_backend=get_train_log_backend_setting(settings_state),
                                training_log_tracker_name=settings_state.get(TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
                                stream_training_output=is_truthy(settings_state.get(TRAIN_STREAM_TO_LOGGER_KEY), default=False),
                                auto_cleanup_states=is_truthy(settings_state.get(TRAIN_AUTO_CLEANUP_STATES_KEY), default=True),
                                logger=log,
                                do_prep_dataset=True,
                                do_cache_latents=True,
                                do_cache_text=True,
                                do_train=True,
                                cancel_requested=(lambda: run_cancel_event is not None and run_cancel_event.is_set()),
                                on_error=capture_job_error,
                            )

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
                                next_status = "resume"
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
    create_dataset_button = ttk.Button(controls, text="Create Dataset", command=create_dataset)
    metrics_viewer_button = ttk.Button(controls, text="TensorBoard", command=open_metrics_viewer_dialog)
    lora_merge_tool_button = ttk.Button(controls, text="LoRA Post-Hoc EMA Merge", command=open_lora_merge_tool_dialog)
    settings_button = ttk.Button(controls, text="Settings", command=apply_settings_from_dialog)
    create_job_large_button = ttk.Button(dataset_actions_bar, text="Create Job", command=open_create_job_dialog)
    run_button = ttk.Button(start_bar, text="START QUEUE", command=run_queue, style="StartDisabled.TButton")

    scan_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    create_dataset_button.grid(row=0, column=2, padx=(0, 8), sticky="e")
    metrics_viewer_button.grid(row=0, column=3, padx=(0, 8), sticky="e")
    lora_merge_tool_button.grid(row=0, column=4, padx=(0, 8), sticky="e")
    settings_button.grid(row=0, column=5, sticky="e")

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
            hold = flag_to_bool(job.get("hold", "0"))
            row_bg = row_background_for_item(item_id)
            cb_text = "☐" if hold else "☑"
            cb_fg = "#4a6a8a" if hold else "#5eead4"
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
                cursor="hand2",
                anchor="center",
            )
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

    def on_canvas_configure(event: tk.Event) -> None:
        request_relayout(event.width)
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
        runtime_config = klein_runtime_config_from_settings(settings)
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
            resolution=get_positive_int_setting(settings, TRAIN_RESOLUTION_KEY, DEFAULT_RESOLUTION, minimum=64),
            network_dim=get_positive_int_setting(settings, TRAIN_NETWORK_DIM_KEY, DEFAULT_NETWORK_DIM),
            network_alpha=get_positive_int_setting(settings, TRAIN_NETWORK_ALPHA_KEY, DEFAULT_NETWORK_ALPHA),
            optimizer_type=get_train_optimizer_setting(settings),
            learning_rate=get_learning_rate_setting(settings),
            train_steps=get_positive_int_setting(settings, TRAIN_STEPS_KEY, DEFAULT_TRAIN_STEPS),
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
