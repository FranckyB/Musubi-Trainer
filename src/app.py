import sys
import argparse
import os
import re
import json
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

from . import app_settings
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
from .train_sdxl import run_job as _run_job_sdxl
from .train_ltx import run_job as _run_job_ltx
from .train_wan import run_job as _run_job_wan
from .train_zimage import run_job as _run_job_zimage
from .train_qwen import run_job as _run_job_qwen
from .lora_merge_utils import (
    compact_merge_selection_token,
    merge_preset_file_token,
    merge_preset_tooltip_text,
    merge_mode_tooltip_text,
    next_merged_output_path,
    post_hoc_ema_mode_args_for_preset,
)
from .launcher_shared import (
    VALID_IMAGE_EXTENSIONS,
    LATENT_SUFFIX,
    DATASET_ORDER_KEY,
    DRAG_START_THRESHOLD_PX,
    DATASET_NAME_PATTERN,
    TRAIN_DIM_ALPHA_CHOICES,
    RESOLUTION_CHOICES,
    OPTIMIZER_TYPE_CHOICES,
    DEFAULT_PRODIGY_OPTIMIZER_ARGS,
    JOB_SETTINGS_FILE_NAME,
    JOBS_ORDER_FILE_NAME,
    JOB_PRESET_FILE_SUFFIX,
    JOB_PROGRESS_FILE_NAME,
    get_positive_int_setting,
    get_non_negative_int_setting,
    get_train_log_backend_setting,
    is_truthy,
    is_valid_folder_name,
    load_dataset_order,
    save_dataset_order,
    latest_checkpoint_for_dataset,
    scan_training_folders,
    dataset_image_files,
    is_step1_ready,
    is_step2_ready,
    is_step3_ready,
    dataset_status,
)

# Model name → run_job function
_KLEIN_MODELS = {"flux2-dev", "klein-base-9b", "klein-9b", "klein-base-4b", "klein-4b"}
_SDXL_MODELS = {"sdxl", "pony", "illustrious"}
_LTX_MODELS = {"ltx-2.3"}
_WAN_MODELS = {"wan2.1-t2v-14b", "wan2.1-i2v-720p-14b", "wan2.1-i2v-480p-14b", "wan2.2-t2v-14b", "wan2.2-i2v-720p-14b", "wan2.2-i2v-480p-14b"}
_ZIMAGE_MODELS = {"zimage-de-turbo"}
_QWEN_MODELS = {"qwen-image", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511", "qwen-image-layered"}

MUSUBI_MAIN_REPO_URL = "https://github.com/kohya-ss/musubi-tuner.git"
MUSUBI_LTX_REPO_URL = "https://github.com/AkaneTendo25/musubi-tuner.git"
MUSUBI_LTX_REPO_BRANCH = "ltx-2-dev"
SD_SCRIPTS_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"


def _run_job_for_model(model_name: str):
    """Return the appropriate run_job function for a given model name."""
    if model_name in _KLEIN_MODELS:
        return _run_job_flux2
    if model_name in _SDXL_MODELS:
        return _run_job_sdxl
    if model_name in _LTX_MODELS:
        return _run_job_ltx
    if model_name in _WAN_MODELS:
        return _run_job_wan
    if model_name in _ZIMAGE_MODELS:
        return _run_job_zimage
    if model_name in _QWEN_MODELS:
        return _run_job_qwen
    return None


class LauncherApplication:
    """Phase-1 class wrapper for incremental migration from nested functions."""

    def run(self) -> int:
        return _launch_ui_impl()


def launch_ui() -> int:
    return LauncherApplication().run()


def _launch_ui_impl() -> int:
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
    color_create_job_enabled = "#2f7de1"
    color_create_job_enabled_active = "#4b93ec"
    color_create_job_disabled = "#3b3b3b"
    workspace_dir = Path(__file__).resolve().parent.parent
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
    root.geometry("780x1000")
    root.resizable(False, True)
    root.minsize(780, 1000)
    root.configure(bg=bg_root)
    ico_path = Path(__file__).resolve().parent / "icons" / "logo.ico"
    if sys.platform == "win32" and ico_path.exists():
        try:
            root.iconbitmap(default=str(ico_path))
        except Exception:
            pass
    set_dark_title_bar(root)

    settings_state = app_settings.load_settings()
    dataset_order: list[str] = load_dataset_order(settings_state)
    window_position_applied = False
    settings_reset_requested = False

    def default_backends_root() -> Path:
        return workspace_dir / "Backends"

    def default_backend_dir(kind: str) -> Path:
        root = default_backends_root()
        if kind == "musubi-main":
            return root / "musubi-main"
        if kind == "musubi-ltx":
            return root / "musubi-ltx"
        if kind == "sd-scripts":
            return root / "sd-scripts"
        return root / kind

    def configured_backends_root() -> Path:
        raw = settings_state.get(app_settings.BACKENDS_ROOT_KEY, "").strip()
        return Path(raw).expanduser() if raw else default_backends_root()

    def configured_backend_dir(setting_key: str, default_kind: str) -> Path:
        raw = settings_state.get(setting_key, "").strip()
        if raw:
            return Path(raw).expanduser()
        return configured_backends_root() / default_kind

    def has_any_backend_setting() -> bool:
        keys = [app_settings.MUSUBI_DIR_KEY, app_settings.MUSUBI_MAIN_DIR_KEY, app_settings.MUSUBI_LTX_DIR_KEY, app_settings.SD_SCRIPTS_DIR_KEY]
        return any(settings_state.get(key, "").strip() for key in keys)

    def has_any_model_setting() -> bool:
        raw_model_paths = settings_state.get(app_settings.MODEL_PATHS_KEY, "").strip()
        if raw_model_paths:
            try:
                payload = json.loads(raw_model_paths)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                for comp_map in payload.values():
                    if not isinstance(comp_map, dict):
                        continue
                    for path_value in comp_map.values():
                        if str(path_value or "").strip():
                            return True

        legacy_keys = [app_settings.KLEIN_DIT_KEY, app_settings.KLEIN_VAE_KEY, app_settings.KLEIN_TEXT_ENCODER_KEY, app_settings.LTX_DIT_KEY, app_settings.LTX_VAE_KEY, app_settings.LTX_TEXT_ENCODER_KEY]
        return any(settings_state.get(key, "").strip() for key in legacy_keys)

    def is_first_launch_unconfigured() -> bool:
        return (not has_any_backend_setting()) and (not has_any_model_setting())

    def apply_initial_main_window_position() -> None:
        nonlocal window_position_applied
        root.update_idletasks()

        min_width = root.winfo_reqwidth()
        min_height = root.winfo_reqheight()

        saved_size = app_settings.load_window_size(settings_state)
        if saved_size is None:
            width = max(root.winfo_width(), min_width)
            height = max(root.winfo_height(), min_height, 1000)
        else:
            saved_width, saved_height = saved_size
            width = max(min_width, saved_width)
            height = max(min_height, 1000, saved_height)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        width = min(width, screen_w)
        height = min(height, screen_h)

        saved_pos = app_settings.load_window_position(settings_state)
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
            saved_sash = app_settings.parse_int_setting(settings_state, app_settings.SASH_POSITION_KEY)
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
        settings_state[app_settings.WINDOW_X_KEY] = str(root.winfo_x())
        settings_state[app_settings.WINDOW_Y_KEY] = str(root.winfo_y())
        settings_state[app_settings.WINDOW_WIDTH_KEY] = str(root.winfo_width())
        settings_state[app_settings.WINDOW_HEIGHT_KEY] = str(root.winfo_height())
        try:
            settings_state[app_settings.SASH_POSITION_KEY] = str(paned.sashpos(0))
        except Exception:
            pass
        app_settings.save_settings(settings_state)

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
        focuscolor="#353535",
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
        focuscolor="#2d2d2d",
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
        bordercolor="#141924",
        lightcolor="#141924",
        darkcolor="#141924",
        relief="flat",
        rowheight=52,
    )
    style.layout(
        "Queue.Treeview",
        [("Treeview.treearea", {"sticky": "nswe"})],
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
        background="#3b3b3b",
        foreground=fg_text,
        padding=(10, 3),
        borderwidth=1,
        bordercolor="#505050",
        lightcolor="#6a6a6a",
        darkcolor="#2d2d2d",
        relief="raised",
        focuscolor="#3b3b3b",
        font=("Segoe UI", 9),
    )
    style.map(
        "QueueAction.TButton",
        background=[("active", "#4a4a4a"), ("disabled", "#2f2f2f")],
        foreground=[("disabled", "#8a8a8a")],
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
        focuscolor=color_start_disabled,
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartDisabled.TButton",
        background=[("active", color_start_disabled), ("disabled", color_start_disabled)],
        foreground=[("disabled", "#c6c6c6")],
    )
    style.configure(
        "StartPlay.TButton",
        background=color_start_disabled,
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=1,
        bordercolor="#2ea95a",
        lightcolor="#63e394",
        darkcolor="#238149",
        focuscolor=color_start_disabled,
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartPlay.TButton",
        background=[("active", "#484848"), ("disabled", color_start_disabled)],
        foreground=[("active", "#ffffff"), ("disabled", "#c6c6c6")],
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
        focuscolor=color_start_enabled,
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
        focuscolor=color_start_in_progress,
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "StartInProgress.TButton",
        background=[("active", color_start_in_progress), ("disabled", color_start_in_progress)],
        foreground=[("active", "#ffffff"), ("disabled", "#ffffff")],
    )
    style.configure(
        "CreateJobLarge.TButton",
        background=color_create_job_enabled,
        foreground="#ffffff",
        padding=(10, 4),
        borderwidth=1,
        bordercolor="#2a64ad",
        lightcolor="#6fa8f4",
        darkcolor="#1f4d84",
        focuscolor=color_create_job_enabled,
        relief="raised",
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "CreateJobLarge.TButton",
        background=[("active", color_create_job_enabled_active), ("disabled", color_create_job_disabled)],
        foreground=[("active", "#ffffff"), ("disabled", "#c6c6c6")],
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
    queue_selection_anchor: int | None = None
    queue_select_toggle_var: tk.StringVar | None = None
    queue_select_toggle_button: ttk.Button | None = None
    queue_hold_toggle_var: tk.StringVar | None = None
    queue_hold_toggle_button: ttk.Button | None = None
    queue_archive_button: ttk.Button | None = None
    queue_restore_button: ttk.Button | None = None
    queue_delete_button: ttk.Button | None = None
    queue_play_mode = False
    queue_worker_thread: threading.Thread | None = None
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
        if settings_state.get(app_settings.TRAIN_AUTO_START_TENSORBOARD_KEY, "0").strip().lower() not in {"1", "true", "yes", "on"}:
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

    def is_valid_sd_scripts_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        has_project_marker = (
            (path / "pyproject.toml").exists()
            or (path / "setup.py").exists()
            or (path / "requirements.txt").exists()
        )
        has_train_entry = any((path / name).exists() for name in (
            "train_network.py",
            "flux_train_network.py",
            "sdxl_train_network.py",
            "anima_train_network.py",
        ))
        return has_project_marker and has_train_entry

    def backend_kind_for_model(model_name: str) -> str:
        if model_name in _SDXL_MODELS:
            return "sd-scripts"
        if model_name in _LTX_MODELS:
            return "musubi-ltx"
        return "musubi-main"

    def backend_repo_rows() -> list[tuple[str, str, str, str, str, str]]:
        return [
            (
                "Musubi Main",
                "musubi-main",
                app_settings.MUSUBI_MAIN_DIR_KEY,
                str(default_backend_dir("musubi-main")),
                MUSUBI_MAIN_REPO_URL,
                "",
            ),
            (
                "Musubi LTX",
                "musubi-ltx",
                app_settings.MUSUBI_LTX_DIR_KEY,
                str(default_backend_dir("musubi-ltx")),
                MUSUBI_LTX_REPO_URL,
                MUSUBI_LTX_REPO_BRANCH,
            ),
            (
                "sd-scripts",
                "sd-scripts",
                app_settings.SD_SCRIPTS_DIR_KEY,
                str(default_backend_dir("sd-scripts")),
                SD_SCRIPTS_REPO_URL,
                "",
            ),
        ]

    def configured_backend_dirs() -> dict[str, Path]:
        root = configured_backends_root()
        main_raw = settings_state.get(app_settings.MUSUBI_MAIN_DIR_KEY, "").strip()
        ltx_raw = settings_state.get(app_settings.MUSUBI_LTX_DIR_KEY, "").strip()
        sd_raw = settings_state.get(app_settings.SD_SCRIPTS_DIR_KEY, "").strip()

        legacy_raw = settings_state.get(app_settings.MUSUBI_DIR_KEY, "").strip()
        legacy_path = Path(legacy_raw).expanduser() if legacy_raw else None

        main_dir = Path(main_raw).expanduser() if main_raw else (legacy_path if legacy_path is not None else root / "musubi-main")
        ltx_dir = Path(ltx_raw).expanduser() if ltx_raw else (root / "musubi-ltx")
        sd_dir = Path(sd_raw).expanduser() if sd_raw else (root / "sd-scripts")
        return {
            "musubi-main": main_dir,
            "musubi-ltx": ltx_dir,
            "sd-scripts": sd_dir,
        }

    def backend_is_valid(kind: str, path: Path) -> bool:
        if kind == "sd-scripts":
            return is_valid_sd_scripts_dir(path)
        return is_valid_musubi_tuner_dir(path)

    def required_backends_ready() -> bool:
        # Keep compatibility with older call sites; startup no longer requires all backends.
        # A backend should only gate model visibility, not app launch.
        return True

    def apply_musubi_dir_setting(musubi_dir: Path) -> bool:
        nonlocal runtime_config, settings_state
        if not is_valid_musubi_tuner_dir(musubi_dir):
            return False
        settings_state[app_settings.MUSUBI_DIR_KEY] = str(musubi_dir)
        settings_state[app_settings.MUSUBI_MAIN_DIR_KEY] = str(musubi_dir)
        settings_state[app_settings.MUSUBI_PYTHON_KEY] = ""
        settings_state.setdefault(app_settings.BACKENDS_ROOT_KEY, str(default_backends_root()))
        app_settings.save_settings(settings_state)
        runtime_config = runtime_config_from_settings(settings_state)
        return runtime_config is not None

    def _run_git_streaming(
        command: list[str],
        logger: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            return 1, str(exc)

        output_lines: list[str] = []
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            output_lines.append(line)
            if logger is not None and line:
                logger(f"[git] {line}")
        process.wait()
        output = "\n".join(output_lines).strip()
        return int(process.returncode or 0), output

    def clone_backend_repo(
        target_dir: Path,
        repo_url: str,
        branch: str = "",
        logger: Callable[[str], None] | None = None,
    ) -> tuple[bool, str]:
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"Could not create parent folder:\n{exc}"

        if target_dir.exists():
            if any(target_dir.iterdir()):
                return False, f"Target folder is not empty:\n{target_dir}"
        else:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return False, f"Could not create target folder:\n{exc}"

        cmd = ["git", "clone"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo_url, str(target_dir)]
        if logger is not None:
            logger(f"[Backends] Running: {' '.join(cmd)}")
        return_code, combined_output = _run_git_streaming(cmd, logger=logger)

        if return_code != 0:
            err_tail = (combined_output or "").strip()
            if len(err_tail) > 1500:
                err_tail = err_tail[-1500:]
            return False, err_tail or "git clone failed."

        return True, ""

    def update_backend_repo(repo_dir: Path, logger: Callable[[str], None] | None = None) -> tuple[bool, str]:
        if not repo_dir.exists() or not repo_dir.is_dir():
            return False, f"Backend folder does not exist:\n{repo_dir}"
        if not (repo_dir / ".git").exists():
            return False, f"Backend folder is not a git checkout:\n{repo_dir}"
        cmd = ["git", "-C", str(repo_dir), "pull", "--ff-only"]
        if logger is not None:
            logger(f"[Backends] Running: {' '.join(cmd)}")
        return_code, combined_output = _run_git_streaming(cmd, logger=logger)

        if return_code != 0:
            err_tail = (combined_output or "").strip()
            if len(err_tail) > 1500:
                err_tail = err_tail[-1500:]
            return False, err_tail or "git pull failed."
        return True, combined_output or "Already up to date."

    def persist_dataset_order() -> None:
        nonlocal settings_state
        save_dataset_order(settings_state, dataset_order)
        app_settings.save_settings(settings_state)

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

    def _set_settings_reset_requested(value: bool) -> None:
        nonlocal settings_reset_requested
        settings_reset_requested = bool(value)

    def open_settings_dialog(required: bool) -> RuntimeConfig | None:
        from .ui.windows import settings_window
        dependencies = {
            "DEFAULT_SAVE_EVERY_N_STEPS": DEFAULT_SAVE_EVERY_N_STEPS,
            "DOWNLOAD_COMPONENT_FRIENDLY_NAMES": DOWNLOAD_COMPONENT_FRIENDLY_NAMES,
            "DOWNLOAD_LOCATIONS": DOWNLOAD_LOCATIONS,
            "DOWNLOAD_LOCATION_MODELS_FOLDER": DOWNLOAD_LOCATION_MODELS_FOLDER,
            "DOWNLOAD_MODELS": DOWNLOAD_MODELS,
            "DOWNLOAD_MODEL_DISPLAY_NAMES": DOWNLOAD_MODEL_DISPLAY_NAMES,
            "DOWNLOAD_MODEL_FAMILIES": DOWNLOAD_MODEL_FAMILIES,
            "JOB_PRESET_FILE_SUFFIX": JOB_PRESET_FILE_SUFFIX,
            "Path": Path,
            "app_settings": app_settings,
            "backend_is_valid": backend_is_valid,
            "backend_repo_rows": backend_repo_rows,
            "bg_panel": bg_panel,
            "center_window": center_window,
            "clone_backend_repo": clone_backend_repo,
            "configured_backend_dirs": configured_backend_dirs,
            "configured_backends_root": configured_backends_root,
            "default_backends_root": default_backends_root,
            "default_models_dir": default_models_dir,
            "download_cli_script_path": Path(__file__).resolve().parent / "download_cli.py",
            "download_workspace_root": download_workspace_root,
            "fg_muted": fg_muted,
            "filedialog": filedialog,
            "find_component": find_component,
            "get_positive_int_setting": get_positive_int_setting,
            "json": json,
            "log": log,
            "messagebox": messagebox,
            "re": re,
            "resolve_musubi_python": resolve_musubi_python,
            "root": root,
            "runtime_config_from_settings": runtime_config_from_settings,
            "set_dark_title_bar": set_dark_title_bar,
            "set_settings_reset_requested": _set_settings_reset_requested,
            "settings_state": settings_state,
            "shutil": shutil,
            "subprocess": subprocess,
            "threading": threading,
            "tk": tk,
            "ttk": ttk,
            "update_backend_repo": update_backend_repo,
        }
        return settings_window.SettingsWindow(**dependencies).open(required)


    if is_first_launch_unconfigured():
        messagebox.showinfo(
            "First launch setup",
            "No model or Kohya-ss tools were found yet.\n\n"
            "Open Settings and set up tools to use:\n"
            "Musubi Main/Musubi LTX/SD-Scripts\n"
            "Also set or download models you want to use.",
            parent=root,
        )
        runtime_config = open_settings_dialog(required=False)
        if runtime_config is None:
            root.destroy()
            return 1
    elif runtime_config is None:
        runtime_config = open_settings_dialog(required=False)
        if runtime_config is None:
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
    ttk.Label(header, text="Datasets:").grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(8, 0, 8, 4))
    controls.grid(row=1, column=0, sticky="ew")
    controls.columnconfigure(4, weight=1)

    def apply_settings_from_dialog(required: bool = False) -> bool:
        nonlocal runtime_config, dataset_order
        updated = open_settings_dialog(required=required)
        if updated is None:
            return False

        runtime_config = updated
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

    def _lora_merge_window():
        from .ui.windows import lora_merge_window

        dependencies = {
            "DND_FILES": DND_FILES,
            "Path": Path,
            "attach_hover_tooltip": attach_hover_tooltip,
            "bg_panel": bg_panel,
            "border_dark": border_dark,
            "center_window": center_window,
            "checkpoint_cache": checkpoint_cache,
            "fg_text": fg_text,
            "filedialog": filedialog,
            "log": log,
            "merge_mode_tooltip_text": merge_mode_tooltip_text,
            "merge_preset_file_token": merge_preset_file_token,
            "merge_preset_tooltip_text": merge_preset_tooltip_text,
            "messagebox": messagebox,
            "next_merged_output_path": next_merged_output_path,
            "os": os,
            "post_hoc_ema_mode_args_for_preset": post_hoc_ema_mode_args_for_preset,
            "rebuild_folder_list": rebuild_folder_list,
            "root": root,
            "runtime_config": runtime_config,
            "set_dark_title_bar": set_dark_title_bar,
            "subprocess": subprocess,
            "tk": tk,
            "tkdnd_available": tkdnd_available,
            "ttk": ttk,
        }
        return lora_merge_window.LoraMergeWindow(**dependencies)

    def ask_lora_merge_options(
        dataset_name: str,
        available_loras: list[Path],
    ) -> tuple[list[str], list[tuple[str, str, list[str], str]]] | None:
        return _lora_merge_window().ask_lora_merge_options(dataset_name, available_loras)

    def lora_post_hoc_ema_merge(dataset_name: str) -> None:
        _lora_merge_window().lora_post_hoc_ema_merge(dataset_name)

    def lora_post_hoc_ema_merge_for_output(target_name: str, output_dir: Path, merge_output_dir: Path | None = None) -> None:
        _lora_merge_window().lora_post_hoc_ema_merge_for_output(target_name, output_dir, merge_output_dir)

    def open_lora_merge_tool_dialog() -> None:
        _lora_merge_window().open_lora_merge_tool_dialog()

    def archive_root_dir() -> Path:
        return datasets_root_dir().parent / "Archives"

    def archive_dataset(dataset_name: str) -> None:
        src = dataset_dir_path(dataset_name)
        if not src.exists() or not src.is_dir():
            messagebox.showerror("Archive failed", f"Dataset folder not found:\n{src}", parent=root)
            return
        dest_parent = archive_root_dir() / "Datasets"
        dest = dest_parent / dataset_name
        if not messagebox.askyesno(
            "Archive dataset",
            f"Move dataset '{dataset_name}' to Archives/Datasets?\n\nThe folder will be moved out of the Datasets directory.",
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
            f"Move {len(names)} dataset(s) to Archives/Datasets?\n\n{label}\n\nThe folders will be moved out of the Datasets directory.",
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

        # Keep the dialog fully visible on smaller displays before centering.
        screen_w = dialog.winfo_screenwidth()
        screen_h = dialog.winfo_screenheight()
        win_w = min(win_w, max(420, screen_w - 40))
        win_h = min(win_h, max(320, screen_h - 80))

        # Center using the final geometry dimensions (not requested size), then show.
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        dialog.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
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

    def archive_selected_jobs() -> None:
        selected_indices = selected_queue_indices()
        if not selected_indices:
            messagebox.showinfo("Archive Jobs", "Select at least one job first.", parent=root)
            return

        selected_names = [job_queue[idx].get("job_name", "unnamed") for idx in selected_indices]
        preview = "\n".join(f"- {name}" for name in selected_names[:12])
        if len(selected_names) > 12:
            preview += f"\n...and {len(selected_names) - 12} more"
        if not messagebox.askyesno(
            "Archive Jobs",
            f"Archive {len(selected_indices)} selected job(s)?\n\n{preview}",
            parent=root,
        ):
            return

        dest_parent = archive_root_dir() / "Jobs"
        archived_names: list[str] = []
        errors: list[str] = []

        for idx in sorted(selected_indices, reverse=True):
            if idx < 0 or idx >= len(job_queue):
                continue
            job = job_queue[idx]
            job_name = job.get("job_name", "unnamed")
            training_name = job.get("training_name", "").strip() or job_name
            training_dir = Path(job.get("training_dir", "")).expanduser()
            if not training_dir.exists() and training_name:
                training_dir = training_job_dir_path(training_name).expanduser()
            dest = dest_parent / training_dir.name

            overwrite = False
            if dest.exists():
                overwrite = messagebox.askyesno(
                    "Archive Jobs",
                    f"Archived job '{training_dir.name}' already exists. Overwrite it?",
                    parent=root,
                )
                if not overwrite:
                    continue

            removed = job_queue.pop(idx)
            remove_job_from_disk(removed)
            archived_names.append(job_name)

            if training_dir.exists() and training_dir.is_dir():
                try:
                    dest_parent.mkdir(parents=True, exist_ok=True)
                    if overwrite and dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(training_dir), str(dest))
                except OSError as exc:
                    errors.append(f"{job_name}: {exc}")

        save_job_order()
        refresh_job_queue_list()
        update_start_button_state()

        if archived_names:
            log(f"[Archive] Archived {len(archived_names)} job(s): {', '.join(archived_names)}")
        if errors:
            messagebox.showerror(
                "Archive Jobs",
                "Some jobs were archived from the queue but their folders could not be moved:\n\n" + "\n".join(errors),
                parent=root,
            )

    def open_restore_jobs_dialog() -> None:
        archive_jobs_dir = archive_root_dir() / "Jobs"
        if not archive_jobs_dir.exists() or not archive_jobs_dir.is_dir():
            messagebox.showinfo("Restore Jobs", "No archived jobs found.", parent=root)
            return

        archived_names = sorted(
            [child.name for child in archive_jobs_dir.iterdir() if child.is_dir()],
            key=str.casefold,
        )
        if not archived_names:
            messagebox.showinfo("Restore Jobs", "No archived jobs found.", parent=root)
            return

        dialog = tk.Toplevel(root)
        dialog.title("Restore Archived Jobs")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=bg_panel)
        dialog.resizable(False, False)

        outer = ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text="Archived Jobs:").grid(row=0, column=0, sticky="w", pady=(0, 6))

        listbox = tk.Listbox(
            outer,
            selectmode="extended",
            height=min(18, max(8, len(archived_names))),
            bg="#0f1724",
            fg=fg_text,
            selectbackground="#1e4a7a",
            selectforeground="#ffffff",
            activestyle="none",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#2a4a72",
            highlightcolor="#2a4a72",
        )
        listbox.grid(row=1, column=0, sticky="nsew")
        for name in archived_names:
            listbox.insert("end", name)

        buttons = ttk.Frame(outer, padding=(0, 8, 0, 0))
        buttons.grid(row=2, column=0, sticky="ew")

        def _select_all() -> None:
            listbox.selection_set(0, "end")

        def _select_none() -> None:
            listbox.selection_clear(0, "end")

        def _restore_selected() -> None:
            selected_positions = listbox.curselection()
            if not selected_positions:
                messagebox.showwarning("Restore Jobs", "Select at least one archived job.", parent=dialog)
                return

            selected_names = [listbox.get(pos) for pos in selected_positions]
            jobs_root = jobs_storage_dir()
            jobs_root.mkdir(parents=True, exist_ok=True)

            restored: list[str] = []
            errors: list[str] = []

            for name in selected_names:
                src = archive_jobs_dir / name
                dest = jobs_root / name
                if not src.exists() or not src.is_dir():
                    errors.append(f"{name}: archived folder missing")
                    continue

                overwrite = False
                if dest.exists():
                    overwrite = messagebox.askyesno(
                        "Restore Jobs",
                        f"Job '{name}' already exists. Overwrite it?",
                        parent=dialog,
                    )
                    if not overwrite:
                        continue

                try:
                    if overwrite and dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(src), str(dest))
                    restored.append(name)
                except OSError as exc:
                    errors.append(f"{name}: {exc}")

            if restored:
                load_job_queue_from_disk()
                refresh_job_queue_list()
                update_start_button_state()
                log(f"[Restore] Restored {len(restored)} job(s): {', '.join(restored)}")

            if errors:
                messagebox.showerror(
                    "Restore Jobs",
                    "Some jobs could not be restored:\n\n" + "\n".join(errors),
                    parent=dialog,
                )

            if restored:
                dialog.destroy()

        ttk.Button(buttons, text="Select All", command=_select_all).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Select None", command=_select_none).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Restore Selected", style="QueueAction.TButton", command=_restore_selected).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).grid(row=0, column=3)

        center_window(dialog)

    def delete_selected_jobs_with_confirmation() -> None:
        selected_indices = selected_queue_indices()
        if not selected_indices:
            messagebox.showinfo("Delete Jobs", "Select at least one job first.", parent=root)
            return

        selected_names = [job_queue[idx].get("job_name", "unnamed") for idx in selected_indices]
        preview = "\n".join(f"- {name}" for name in selected_names[:12])
        if len(selected_names) > 12:
            preview += f"\n...and {len(selected_names) - 12} more"
        if not messagebox.askyesno(
            "Delete Jobs",
            (
                f"Delete {len(selected_indices)} selected job(s)?\n\n"
                "This removes them from the queue and deletes each job folder if it exists.\n\n"
                f"{preview}"
            ),
            parent=root,
        ):
            return

        deleted_names: list[str] = []
        errors: list[str] = []

        for idx in sorted(selected_indices, reverse=True):
            if idx < 0 or idx >= len(job_queue):
                continue
            job = job_queue[idx]
            job_name = job.get("job_name", "unnamed")
            training_name = job.get("training_name", "").strip() or job_name
            training_dir = Path(job.get("training_dir", "")).expanduser()
            if not training_dir.exists() and training_name:
                training_dir = training_job_dir_path(training_name).expanduser()

            removed = job_queue.pop(idx)
            remove_job_from_disk(removed)

            if training_dir.exists() and training_dir.is_dir():
                try:
                    shutil.rmtree(training_dir)
                except OSError as exc:
                    errors.append(f"{job_name}: {exc}")

            deleted_names.append(job_name)

        save_job_order()
        refresh_job_queue_list()
        update_start_button_state()

        if deleted_names:
            log(f"[Queue] Deleted {len(deleted_names)} job(s): {', '.join(deleted_names)}")
        if errors:
            messagebox.showerror(
                "Delete Jobs",
                "Some job folders could not be deleted:\n\n" + "\n".join(errors),
                parent=root,
            )

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

    dataset_table_border = tk.Frame(top_pane, bg="#2a4a72", bd=0, highlightthickness=0)
    dataset_table_border.grid(row=0, column=0, sticky="nsew")
    dataset_table_border.columnconfigure(0, weight=1)
    dataset_table_border.rowconfigure(0, weight=1)

    list_container = ttk.Frame(dataset_table_border, padding=2)
    list_container.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
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

    queue_container = ttk.Frame(bottom_pane, padding=(8, 0, 8, 8))
    queue_container.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
    queue_container.columnconfigure(0, weight=1)
    queue_container.rowconfigure(1, weight=1)

    queue_header = ttk.Frame(queue_container)
    queue_header.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    queue_header.columnconfigure(0, weight=1)
    ttk.Label(queue_header, text="Queue:", style="TLabel").grid(row=0, column=0, sticky="w")

    queue_actions_bar = ttk.Frame(queue_header, padding=(0, 2, 0, 4))
    queue_actions_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
    queue_actions_bar.columnconfigure(6, weight=1)

    queue_table_border = tk.Frame(queue_container, bg="#2a4a72", bd=0, highlightthickness=0)
    queue_table_border.grid(row=1, column=0, columnspan=2, sticky="nsew")
    queue_table_border.columnconfigure(0, weight=1)
    queue_table_border.rowconfigure(0, weight=1)

    queue_list = ttk.Treeview(
        queue_table_border,
        columns=("run", "thumb", "name", "source", "status", "actions"),
        show="tree headings",
        selectmode="extended",
        height=6,
        takefocus=False,
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
    queue_list.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
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
    log_box.configure(state="disabled")
    log_progress_active = False
    log_progress_mark = "log_progress_line_start"

    def _clear_log_box() -> None:
        log_box.configure(state="normal")
        log_box.delete("1.0", "end")
        log_box.configure(state="disabled")

    def _show_log_context_menu(event: "tk.Event[tk.Text]") -> None:
        menu = tk.Menu(
            root, tearoff=0,
            bg=bg_card, fg=fg_text,
            activebackground="#3a3f4b", activeforeground=fg_text,
            bd=0, relief="flat",
        )
        menu.add_command(label="Clear console", command=_clear_log_box)
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
            log_box.configure(state="normal")
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

            log_box.configure(state="disabled")

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

    def ensure_job_training_args_toml(job: dict[str, str], *, silent: bool = True) -> bool:
        model_name = (job.get("model", "klein-base-9b") or "klein-base-9b").strip()
        job_run_fn = _run_job_for_model(model_name)
        if job_run_fn is None:
            return False

        job_runtime_config = runtime_config_for_model(settings_state, model_name) or runtime_config
        if job_runtime_config is None:
            return False

        training_name = job.get("training_name", "").strip() or job.get("job_name", "").strip()
        if not training_name:
            return False

        output_dir = Path(job.get("output_dir", str(training_job_dir_path(training_name) / "output"))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger_fn: Callable[[str], None] = (lambda _message: None) if silent else log
        error_message = ""

        def capture_job_error(message: str) -> None:
            nonlocal error_message
            error_message = message

        run_job_kwargs = {
            "dataset_name": training_name,
            "output_name": job.get("job_name", training_name),
            "output_dir": output_dir,
            "default_caption_keyword": settings_state.get(app_settings.DEFAULT_CAPTION_KEYWORD_KEY, ""),
            "resolution": get_positive_int_setting(job, "resolution", DEFAULT_RESOLUTION, minimum=64),
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
            "enable_training_logging": is_truthy(settings_state.get(app_settings.TRAIN_ENABLE_LOGGING_KEY), default=True),
            "training_log_backend": get_train_log_backend_setting(settings_state),
            "training_log_tracker_name": settings_state.get(app_settings.TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
            "stream_training_output": is_truthy(settings_state.get(app_settings.TRAIN_STREAM_TO_LOGGER_KEY), default=False),
            "auto_cleanup_states": is_truthy(settings_state.get(app_settings.TRAIN_AUTO_CLEANUP_STATES_KEY), default=True),
            "logger": logger_fn,
            "do_prep_dataset": False,
            "do_cache_latents": False,
            "do_cache_text": False,
            "do_train": False,
            "generate_training_args_only": True,
            "cancel_requested": None,
            "on_error": capture_job_error,
        }

        if job_run_fn is _run_job_ltx:
            run_job_kwargs["ltx_mode"] = str(job.get("ltx_mode", "video"))
            run_job_kwargs["ltx_gemma_load_in_4bit"] = flag_to_bool(job.get("ltx_gemma_load_in_4bit", "1"))
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
        elif job_run_fn is _run_job_sdxl:
            run_job_kwargs["lr_scheduler"] = str(job.get("lr_scheduler", "constant") or "constant")
            run_job_kwargs["lr_warmup_steps"] = get_non_negative_int_setting(job, "lr_warmup_steps", 0)
            run_job_kwargs["gradient_accumulation_steps"] = get_positive_int_setting(job, "gradient_accumulation_steps", 1, minimum=1)
            run_job_kwargs["unet_lr"] = str(job.get("sd_unet_lr", "") or "")
            run_job_kwargs["text_encoder_lr"] = str(job.get("sd_text_encoder_lr", "") or "")
            run_job_kwargs["enable_gradient_checkpointing"] = flag_to_bool(job.get("enable_grad_ckpt", "1"))

        exit_code = job_run_fn(job_runtime_config, **run_job_kwargs)
        config_path = output_dir.parent / "training_args.toml"
        if exit_code == JOB_EXIT_SUCCESS and config_path.exists() and config_path.is_file():
            return True

        if (not silent) and error_message:
            log(f"[Queue] training_args generation note for {training_name}: {error_message}")
        return False

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
                    app_settings.TRAIN_SAVE_EVERY_N_STEPS_KEY,
                    DEFAULT_SAVE_EVERY_N_STEPS,
                    minimum=1,
                )
            ),
            "enable_compile": settings_state.get(app_settings.ENABLE_COMPILE_OPTIMIZATIONS_KEY, "0"),
            "enable_tf32": settings_state.get(app_settings.ENABLE_CUDA_ALLOW_TF32_KEY, "1"),
            "enable_cudnn": settings_state.get(app_settings.ENABLE_CUDA_CUDNN_BENCHMARK_KEY, "1"),
            "enable_fp8": settings_state.get(app_settings.ENABLE_FP8_DIT_KEY, "0"),
            "enable_gc": settings_state.get(app_settings.ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY, "0"),
            "enable_logging": bool_to_flag(is_truthy(settings_state.get(app_settings.TRAIN_ENABLE_LOGGING_KEY), default=True)),
            "tracker_name": training_name,
            "stream_output": bool_to_flag(is_truthy(settings_state.get(app_settings.TRAIN_STREAM_TO_LOGGER_KEY), default=False)),
            "auto_cleanup": bool_to_flag(is_truthy(settings_state.get(app_settings.TRAIN_AUTO_CLEANUP_STATES_KEY), default=True)),
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

    def update_dataset_select_toggle_state() -> None:
        if dataset_select_toggle_var is None:
            return
        total_count = len(vars_by_name)
        selected_count = sum(1 for var in vars_by_name.values() if var.get())
        all_selected = total_count > 0 and selected_count == total_count
        dataset_select_toggle_var.set("Select None" if all_selected else "Select All")
        if dataset_select_toggle_button is not None:
            dataset_select_toggle_button.state(["!disabled"] if total_count > 0 else ["disabled"])

    def toggle_dataset_select_all_none() -> None:
        if not vars_by_name:
            update_dataset_select_toggle_state()
            return

        all_selected = all(var.get() for var in vars_by_name.values())
        next_value = not all_selected
        for name, var in vars_by_name.items():
            var.set(next_value)
            apply_card_style(name)
        update_dataset_select_toggle_state()
        update_start_button_state()

    def selected_queue_indices() -> list[int]:
        indices: list[int] = []
        for item_id in queue_list.selection():
            try:
                idx = int(item_id)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(job_queue):
                indices.append(idx)
        return sorted(set(indices))

    def selected_queue_index() -> int | None:
        indices = selected_queue_indices()
        if not indices:
            return None
        return indices[0]

    def set_queue_selection_indices(indices: list[int], anchor_index: int | None = None) -> None:
        nonlocal queue_selection_anchor
        valid = sorted({idx for idx in indices if 0 <= idx < len(job_queue)})
        queue_list.selection_set([str(idx) for idx in valid])

        if valid:
            focus_idx = valid[-1]
            focus_item = str(focus_idx)
            queue_list.focus(focus_item)
            queue_list.see(focus_item)

        if anchor_index is not None and 0 <= anchor_index < len(job_queue):
            queue_selection_anchor = anchor_index
        elif valid:
            queue_selection_anchor = valid[0]
        else:
            queue_selection_anchor = None

    def set_queue_selection(index: int) -> None:
        if index < 0 or index >= len(job_queue):
            return
        set_queue_selection_indices([index], anchor_index=index)

    def update_queue_multi_action_state() -> None:
        if queue_select_toggle_var is None:
            return
        selected_count = len(selected_queue_indices())
        total_count = len(job_queue)
        all_selected = total_count > 0 and selected_count == total_count
        queue_select_toggle_var.set("Select None" if all_selected else "Select All")

        if queue_select_toggle_button is not None:
            queue_select_toggle_button.state(["!disabled"] if total_count > 0 else ["disabled"])
        if queue_hold_toggle_var is not None:
            queue_hold_toggle_var.set("Enable Selected")
            selected = selected_queue_indices()
            toggleable_indices = [
                idx
                for idx in selected
                if 0 <= idx < len(job_queue) and detect_job_status(job_queue[idx]) != "done"
            ]
            if toggleable_indices:
                all_enabled = all(not flag_to_bool(job_queue[idx].get("hold", "0")) for idx in toggleable_indices)
                queue_hold_toggle_var.set("Disable Selected" if all_enabled else "Enable Selected")
        if queue_hold_toggle_button is not None:
            selected = selected_queue_indices()
            toggleable_count = sum(
                1
                for idx in selected
                if 0 <= idx < len(job_queue) and detect_job_status(job_queue[idx]) != "done"
            )
            queue_hold_toggle_button.state(["!disabled"] if toggleable_count > 0 else ["disabled"])
        if queue_archive_button is not None:
            queue_archive_button.state(["!disabled"] if selected_count > 0 else ["disabled"])
        if queue_delete_button is not None:
            queue_delete_button.state(["!disabled"] if selected_count > 0 else ["disabled"])
        if queue_restore_button is not None:
            queue_restore_button.state(["!disabled"])

    def toggle_hold_selected_jobs() -> None:
        selected = selected_queue_indices()
        if not selected:
            return

        toggleable_indices = [
            idx
            for idx in selected
            if 0 <= idx < len(job_queue) and detect_job_status(job_queue[idx]) != "done"
        ]
        if not toggleable_indices:
            return

        all_enabled = all(not flag_to_bool(job_queue[idx].get("hold", "0")) for idx in toggleable_indices)
        target_hold = all_enabled

        changed = False
        for idx in toggleable_indices:
            current = flag_to_bool(job_queue[idx].get("hold", "0"))
            if current == target_hold:
                continue
            job_queue[idx]["hold"] = bool_to_flag(target_hold)
            if target_hold and job_queue[idx].get("status", "") in {"queued", "failed", "running", "resume"}:
                job_queue[idx]["status"] = "paused"
            elif (not target_hold) and job_queue[idx].get("status", "") == "paused":
                job_queue[idx]["status"] = "queued"
            save_job_to_disk(job_queue[idx])
            changed = True

        if not changed:
            update_queue_multi_action_state()
            return

        refresh_job_queue_list()
        set_queue_selection_indices(selected, anchor_index=selected[0])
        update_start_button_state()

    def toggle_queue_select_all_none() -> None:
        if not job_queue:
            set_queue_selection_indices([])
            update_queue_multi_action_state()
            root.after_idle(sync_all_row_overlays)
            return

        selected = selected_queue_indices()
        if len(selected) == len(job_queue):
            set_queue_selection_indices([])
        else:
            set_queue_selection_indices(list(range(len(job_queue))), anchor_index=0)
        update_queue_multi_action_state()
        root.after_idle(sync_all_row_overlays)

    def refresh_job_queue_list() -> None:
        selected_before = selected_queue_indices()
        anchor_before = queue_selection_anchor
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
        set_queue_selection_indices(selected_before, anchor_index=anchor_before)
        update_queue_multi_action_state()

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

        def reset_clicked_job() -> None:
            reset_job_with_confirmation(clicked)

        clicked_status = detect_job_status(job_queue[clicked])
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Open Output Folder", command=lambda: open_job_output_folder(clicked))
        menu.add_command(label="LoRA Post-Hoc EMA Merge", command=lambda: merge_job_output_loras(clicked))
        menu.add_command(label="Duplicate Job", command=lambda: duplicate_job(clicked))
        menu.add_command(label="Edit Job", command=lambda: open_create_job_dialog(existing_job=job_queue[clicked]))
        menu.add_command(label="Clear Job Cache", command=clear_clicked_job_cache)
        menu.add_command(label="Reset Job (Fresh Start)", command=reset_clicked_job)
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
                default_caption_keyword=settings_state.get(app_settings.DEFAULT_CAPTION_KEYWORD_KEY, ""),
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

    def reset_job_with_confirmation(index: int | None = None) -> None:
        idx = selected_queue_index() if index is None else index
        if idx is None or idx < 0 or idx >= len(job_queue):
            return

        job = job_queue[idx]
        job_name = job.get("job_name", "unnamed")
        training_name = job.get("training_name", "").strip() or job_name
        training_dir = Path(job.get("training_dir", "")).expanduser()
        if not training_dir.exists() and training_name:
            training_dir = training_job_dir_path(training_name).expanduser()

        output_dir = Path(job.get("output_dir", "")).expanduser()
        cache_dirs = _job_cache_dirs(job)
        logs_dir = training_dir / "logs"
        progress_path = training_dir / JOB_PROGRESS_FILE_NAME
        training_args_path = training_dir / "training_args.toml"

        if not messagebox.askyesno(
            "Reset job",
            (
                f"Reset job '{job_name}' for a fresh run?\n\n"
                "This keeps the job entry/settings, but clears:\n"
                "- output checkpoints and states\n"
                "- cache latents/text outputs\n"
                "- logs and progress markers\n"
                "- generated training_args.toml"
            ),
            parent=root,
        ):
            return

        removed_files = 0
        removed_dirs = 0

        def clear_directory_contents(target_dir: Path) -> None:
            nonlocal removed_files, removed_dirs
            if not target_dir.exists() or not target_dir.is_dir():
                return
            resolved = target_dir.resolve()
            if resolved == Path(resolved.anchor):
                raise OSError(f"Refusing to clear root directory: {resolved}")
            for child in list(target_dir.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child)
                    removed_dirs += 1
                else:
                    child.unlink(missing_ok=True)
                    removed_files += 1

        try:
            clear_directory_contents(output_dir)
            for cache_dir in cache_dirs:
                clear_directory_contents(cache_dir)
            if logs_dir.exists() and logs_dir.is_dir():
                shutil.rmtree(logs_dir)
                removed_dirs += 1
            if progress_path.exists() and progress_path.is_file():
                progress_path.unlink(missing_ok=True)
                removed_files += 1
            if training_args_path.exists() and training_args_path.is_file():
                training_args_path.unlink(missing_ok=True)
                removed_files += 1
        except OSError as exc:
            messagebox.showerror("Reset job failed", f"Could not reset '{job_name}':\n{exc}", parent=root)
            return

        # Keep hold as-is; detect_job_status will map held jobs to paused.
        job_queue[idx]["status"] = "queued"
        save_job_to_disk(job_queue[idx])
        refresh_job_queue_list()
        set_queue_selection(idx)
        update_start_button_state()

        log(
            f"[Queue] Reset job for fresh run: {job_name} "
            f"({removed_files} file(s), {removed_dirs} folder(s) removed)."
        )
        messagebox.showinfo(
            "Reset job",
            (
                f"Job '{job_name}' has been reset for a fresh run.\n"
                f"Removed {removed_files} file(s) and {removed_dirs} folder(s)."
            ),
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

    def stop_running_job() -> None:
        if not run_in_progress or run_cancel_event is None or run_cancel_event.is_set():
            return
        if not messagebox.askyesno(
            "Stop Job",
            "Stop the currently running job and continue to the next available queued job?",
            parent=root,
        ):
            return
        run_cancel_event.set()
        log("Stop requested for running job. Queue play mode remains active.")
        update_start_button_state()

    def on_queue_press(event: tk.Event) -> str:
        nonlocal queue_drag_index, queue_drag_moved, queue_drag_allowed, queue_selection_anchor
        if len(job_queue) == 0:
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return "break"
        clicked_item = queue_list.identify_row(event.y)
        state_bits = int(event.state or 0)
        shift_pressed = bool(state_bits & 0x0001)
        ctrl_pressed = bool(state_bits & 0x0004)
        if not clicked_item:
            queue_list.selection_set([])
            queue_selection_anchor = None
            update_queue_multi_action_state()
            root.after_idle(sync_all_row_overlays)
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return "break"
        try:
            clicked = int(clicked_item)
        except ValueError:
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return "break"
        if clicked < 0 or clicked >= len(job_queue):
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return "break"

        clicked_col = queue_list.identify_column(event.x)
        if clicked_col == "#1":
            toggle_hold_job(clicked)
            queue_drag_index = None
            queue_drag_moved = False
            queue_drag_allowed = False
            return "break"

        if shift_pressed:
            anchor = queue_selection_anchor
            if anchor is None:
                existing = selected_queue_indices()
                anchor = existing[0] if existing else clicked
            start = min(anchor, clicked)
            end = max(anchor, clicked)
            set_queue_selection_indices(list(range(start, end + 1)), anchor_index=anchor)
        elif ctrl_pressed:
            current = set(selected_queue_indices())
            if clicked in current:
                current.remove(clicked)
            else:
                current.add(clicked)
            set_queue_selection_indices(sorted(current), anchor_index=clicked)
        else:
            set_queue_selection(clicked)

        update_queue_multi_action_state()

        queue_drag_allowed = (not shift_pressed) and (not ctrl_pressed) and clicked_col in {"#0", "#2", "#3", "#4", "#5"}
        queue_drag_index = clicked if queue_drag_allowed else None
        queue_drag_moved = False
        return "break"

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
        from .ui.windows import create_job_window
        dependencies = {
            "DEFAULT_CAPTION_KEYWORD_KEY": app_settings.DEFAULT_CAPTION_KEYWORD_KEY,
            "DEFAULT_LEARNING_RATE": DEFAULT_LEARNING_RATE,
            "DEFAULT_NETWORK_ALPHA": DEFAULT_NETWORK_ALPHA,
            "DEFAULT_NETWORK_DIM": DEFAULT_NETWORK_DIM,
            "DEFAULT_PRODIGY_OPTIMIZER_ARGS": DEFAULT_PRODIGY_OPTIMIZER_ARGS,
            "DEFAULT_RESOLUTION": DEFAULT_RESOLUTION,
            "DEFAULT_SAVE_EVERY_N_STEPS": DEFAULT_SAVE_EVERY_N_STEPS,
            "DEFAULT_TRAIN_STEPS": DEFAULT_TRAIN_STEPS,
            "DOWNLOAD_MODEL_DISPLAY_NAMES": DOWNLOAD_MODEL_DISPLAY_NAMES,
            "DOWNLOAD_MODEL_FAMILIES": DOWNLOAD_MODEL_FAMILIES,
            "ENABLE_COMPILE_OPTIMIZATIONS_KEY": app_settings.ENABLE_COMPILE_OPTIMIZATIONS_KEY,
            "ENABLE_CUDA_ALLOW_TF32_KEY": app_settings.ENABLE_CUDA_ALLOW_TF32_KEY,
            "ENABLE_CUDA_CUDNN_BENCHMARK_KEY": app_settings.ENABLE_CUDA_CUDNN_BENCHMARK_KEY,
            "ENABLE_FP8_DIT_KEY": app_settings.ENABLE_FP8_DIT_KEY,
            "ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY": app_settings.ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY,
            "KLEIN_DIT_KEY": app_settings.KLEIN_DIT_KEY,
            "MODEL_PATHS_KEY": app_settings.MODEL_PATHS_KEY,
            "OPTIMIZER_TYPE_CHOICES": OPTIMIZER_TYPE_CHOICES,
            "PREFERRED_PRESETS_BY_FAMILY_KEY": app_settings.PREFERRED_PRESETS_BY_FAMILY_KEY,
            "RESOLUTION_CHOICES": RESOLUTION_CHOICES,
            "TRAIN_AUTO_CLEANUP_STATES_KEY": app_settings.TRAIN_AUTO_CLEANUP_STATES_KEY,
            "TRAIN_DIM_ALPHA_CHOICES": TRAIN_DIM_ALPHA_CHOICES,
            "TRAIN_ENABLE_LOGGING_KEY": app_settings.TRAIN_ENABLE_LOGGING_KEY,
            "TRAIN_LOG_TRACKER_NAME_KEY": app_settings.TRAIN_LOG_TRACKER_NAME_KEY,
            "TRAIN_SAVE_EVERY_N_STEPS_KEY": app_settings.TRAIN_SAVE_EVERY_N_STEPS_KEY,
            "TRAIN_STREAM_TO_LOGGER_KEY": app_settings.TRAIN_STREAM_TO_LOGGER_KEY,
            "apply_card_style": apply_card_style,
            "attach_hover_tooltip": attach_hover_tooltip,
            "backend_is_valid": backend_is_valid,
            "backend_kind_for_model": backend_kind_for_model,
            "bg_panel": bg_panel,
            "bool_to_flag": bool_to_flag,
            "card_frame_by_name": card_frame_by_name,
            "center_window": center_window,
            "configured_backend_dirs": configured_backend_dirs,
            "dataset_image_files": dataset_image_files,
            "datasets_root_dir": datasets_root_dir,
            "detect_job_element_base_mismatch": detect_job_element_base_mismatch,
            "detect_job_status": detect_job_status,
            "ensure_training_job_structure": ensure_training_job_structure,
            "flag_to_bool": flag_to_bool,
            "get_positive_int_setting": get_positive_int_setting,
            "is_truthy": is_truthy,
            "is_valid_folder_name": is_valid_folder_name,
            "job_preset_file_path": job_preset_file_path,
            "job_queue": job_queue,
            "load_job_presets_from_disk": load_job_presets_from_disk,
            "load_job_queue_from_disk": load_job_queue_from_disk,
            "log": log,
            "refresh_job_queue_list": refresh_job_queue_list,
            "rename_job_elements_to_training_name": rename_job_elements_to_training_name,
            "root": root,
            "save_job_order": save_job_order,
            "save_job_preset_to_disk": save_job_preset_to_disk,
            "save_job_to_disk": save_job_to_disk,
            "ensure_job_training_args_toml": ensure_job_training_args_toml,
            "scan_training_folders": scan_training_folders,
            "selected_dataset_names": selected_dataset_names,
            "set_dark_title_bar": set_dark_title_bar,
            "settings_state": settings_state,
            "training_job_dir_path": training_job_dir_path,
            "unique_job_name": unique_job_name,
            "update_start_button_state": update_start_button_state,
            "vars_by_name": vars_by_name,
            "messagebox": messagebox,
            "simpledialog": simpledialog,
            "tk": tk,
            "ttk": ttk,
            "json": json,
            "Path": Path,
            "shutil": shutil,
        }
        return create_job_window.CreateJobWindow(**dependencies).open(existing_job)

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
        update_dataset_select_toggle_state()

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
            update_dataset_select_toggle_state()
            update_start_button_state()
            return

        gap = 8
        card_width = 172
        thumb_px = 152
        card_height = 212
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
        update_dataset_select_toggle_state()

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
        resize_after_id = root.after(120, rebuild_folder_list)

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

    def is_dataset_panel_scrollable() -> bool:
        first, last = canvas.yview()
        return first > 0.0 or last < 0.999

    def _get_hovered_widget() -> tk.Misc | None:
        try:
            return root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        except KeyError:
            return None

    def on_mousewheel(event: tk.Event) -> str | None:
        hovered = _get_hovered_widget()
        if not is_widget_in_dataset_panel(hovered):
            return None
        if not is_dataset_panel_scrollable():
            return "break"
        delta = int(-event.delta / 120)
        if delta == 0:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")
        return "break"

    def on_mousewheel_linux_up(_event: tk.Event) -> str | None:
        hovered = _get_hovered_widget()
        if not is_widget_in_dataset_panel(hovered):
            return None
        if not is_dataset_panel_scrollable():
            return "break"
        canvas.yview_scroll(-1, "units")
        return "break"

    def on_mousewheel_linux_down(_event: tk.Event) -> str | None:
        hovered = _get_hovered_widget()
        if not is_widget_in_dataset_panel(hovered):
            return None
        if not is_dataset_panel_scrollable():
            return "break"
        canvas.yview_scroll(1, "units")
        return "break"

    root.bind_all("<MouseWheel>", on_mousewheel)
    root.bind_all("<Button-4>", on_mousewheel_linux_up)
    root.bind_all("<Button-5>", on_mousewheel_linux_down)

    def update_queue_border_state() -> None:
        if queue_play_mode:
            border_color = "#2ea95a"
            border_pad = 2
        else:
            border_color = "#2a4a72"
            border_pad = 1

        queue_table_border.configure(bg=border_color)
        queue_list.grid_configure(padx=border_pad, pady=border_pad)
        queue_scroll.grid_configure(pady=border_pad, padx=(0, border_pad))

    def update_start_button_state() -> None:
        if queue_play_mode:
            if run_in_progress and run_cancel_event is not None and run_cancel_event.is_set():
                run_button.configure(text="Stopping", style="StartInProgress.TButton")
            else:
                run_button.configure(text="Pause", style="StartEnabled.TButton")
            run_button.state(["!disabled"])
            update_queue_border_state()
            return

        run_button.configure(text="Start", style="StartPlay.TButton")
        run_button.state(["!disabled"])
        update_queue_border_state()

    def run_queue() -> None:
        nonlocal run_in_progress, run_cancel_event, queue_play_mode, queue_worker_thread

        if queue_play_mode:
            if run_in_progress and run_cancel_event is not None and not run_cancel_event.is_set():
                should_cancel = messagebox.askyesno(
                    "Stop Current Job",
                    "Queue is running. Stop the current job now and switch queue to paused mode?",
                    parent=root,
                )
                if should_cancel:
                    run_cancel_event.set()
                    log("Cancellation requested. Queue will pause after current stop completes.")
            queue_play_mode = False
            update_start_button_state()
            return

        queue_play_mode = True
        update_start_button_state()

        if queue_worker_thread is not None and queue_worker_thread.is_alive():
            return

        model_error_popup_shown = False

        def notify_job_failure_popup(job_name: str, details: str = "") -> None:
            message = f"Job '{job_name}' failed.\n\nSee queue log for full details."

            def show_failure_popup() -> None:
                if not root.winfo_exists():
                    return
                try:
                    root.bell()
                except Exception:
                    pass
                messagebox.showerror("Job Failed", message, parent=root)

            root.after(0, show_failure_popup)

        def log_status(message: str) -> None:
            # Mirror key lifecycle status lines to both the UI log and parent console.
            log(message)
            try:
                print(message, flush=True)
            except Exception:
                pass

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
            nonlocal model_error_popup_shown, run_in_progress, run_cancel_event, queue_play_mode
            try:
                log("")
                log("Queue play mode enabled.")
                failed_jobs: list[str] = []
                while queue_play_mode and root.winfo_exists():
                    runnable_indices = [
                        idx
                        for idx, job in enumerate(job_queue)
                        if (not flag_to_bool(job.get("hold", "0")))
                        and job.get("status", "queued") in {"queued", "failed", "paused", "resume"}
                    ]

                    if not runnable_indices:
                        run_in_progress = False
                        run_cancel_event = None
                        root.after(0, update_start_button_state)
                        time.sleep(0.3)
                        continue

                    if runtime_config is None or runtime_config.dit is None or runtime_config.vae is None or runtime_config.text_encoder is None:
                        missing = []
                        if runtime_config is None or runtime_config.dit is None:
                            missing.append("Model (DiT)")
                        if runtime_config is None or runtime_config.vae is None:
                            missing.append("VAE")
                        if runtime_config is None or runtime_config.text_encoder is None:
                            missing.append("Text Encoder")

                        queue_play_mode = False

                        def show_missing_model_error() -> None:
                            if not root.winfo_exists():
                                return
                            messagebox.showerror(
                                "Model paths not configured",
                                "The following model paths are not set:\n\n"
                                + "\n".join(f"  \u2022 {m}" for m in missing)
                                + "\n\nOpen Settings and configure the paths before starting training.",
                                parent=root,
                            )
                            update_start_button_state()

                        root.after(0, show_missing_model_error)
                        break

                    queue_index = runnable_indices[0]
                    run_cancel_event = threading.Event()
                    run_in_progress = True
                    root.after(0, update_start_button_state)

                    job = job_queue[queue_index]
                    job_name = job.get("job_name", f"job_{queue_index + 1}")
                    dataset_name = job.get("dataset_name", "")
                    job_error_details = ""

                    def mark_running() -> None:
                        if queue_index < len(job_queue):
                            job_queue[queue_index]["status"] = "running"
                            save_job_to_disk(job_queue[queue_index])
                            refresh_job_queue_list()

                    root.after(0, mark_running)

                    model_name = job.get("model", "klein-base-9b") or "klein-base-9b"
                    job_run_fn = _run_job_for_model(model_name)
                    if job_run_fn is None:
                        job_error_details = f"Unsupported model '{model_name}' for job {job_name}."
                        log(f"[Queue] {job_error_details}")
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
                        try:
                            _training_dir, output_dir, captions_added = ensure_training_job_structure(
                                training_name=training_name,
                                datasets=runner_datasets,
                                resolution=resolution_value,
                                batch_size=batch_size_value,
                                default_caption_keyword=settings_state.get(app_settings.DEFAULT_CAPTION_KEYWORD_KEY, ""),
                                model_name=model_name,
                            )
                            if captions_added > 0:
                                ds_label = ", ".join(d["name"] for d in runner_datasets)
                                log(f"[Queue] Added {captions_added} missing caption file(s) for dataset(s): {ds_label}.")
                        except Exception as exc:
                            job_error_details = f"Job setup failed for {job_name}: {exc}"
                            log(f"[Queue] {job_error_details}")
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
                                "default_caption_keyword": settings_state.get(app_settings.DEFAULT_CAPTION_KEYWORD_KEY, ""),
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
                                "enable_training_logging": is_truthy(settings_state.get(app_settings.TRAIN_ENABLE_LOGGING_KEY), default=True),
                                "training_log_backend": get_train_log_backend_setting(settings_state),
                                "training_log_tracker_name": settings_state.get(app_settings.TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
                                "stream_training_output": is_truthy(settings_state.get(app_settings.TRAIN_STREAM_TO_LOGGER_KEY), default=False),
                                "auto_cleanup_states": is_truthy(settings_state.get(app_settings.TRAIN_AUTO_CLEANUP_STATES_KEY), default=True),
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
                                run_job_kwargs["ltx_gemma_load_in_4bit"] = flag_to_bool(job.get("ltx_gemma_load_in_4bit", "1"))
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
                            elif job_run_fn is _run_job_sdxl:
                                run_job_kwargs["lr_scheduler"] = str(job.get("lr_scheduler", "constant") or "constant")
                                run_job_kwargs["lr_warmup_steps"] = get_non_negative_int_setting(job, "lr_warmup_steps", 0)
                                run_job_kwargs["gradient_accumulation_steps"] = get_positive_int_setting(job, "gradient_accumulation_steps", 1, minimum=1)
                                run_job_kwargs["unet_lr"] = str(job.get("sd_unet_lr", "") or "")
                                run_job_kwargs["text_encoder_lr"] = str(job.get("sd_text_encoder_lr", "") or "")
                                run_job_kwargs["enable_gradient_checkpointing"] = flag_to_bool(job.get("enable_grad_ckpt", "1"))

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

                    if exit_code == JOB_EXIT_SUCCESS:
                        log_status(f"[Train] Completed: {job_name}")
                    elif exit_code == JOB_EXIT_CANCELLED or (
                        run_cancel_event is not None and run_cancel_event.is_set()
                    ):
                        log_status(f"[Train] Stopped: {job_name}")
                    else:
                        failure_status = f"[Train] Failed: {job_name}"
                        failure_details = (job_error_details or "").strip()
                        if failure_details:
                            failure_headline = failure_details.splitlines()[0].strip()
                            if failure_headline:
                                failure_status = f"{failure_status} - {failure_headline}"
                        log_status(failure_status)

                    if (
                        exit_code not in {JOB_EXIT_SUCCESS, JOB_EXIT_CANCELLED}
                        and not (run_cancel_event is not None and run_cancel_event.is_set())
                    ):
                        details_text = (job_error_details or "").strip()
                        if details_text:
                            log(f"[Queue] Failure details for {job_name}:")
                            log(details_text)
                            # Mirror full details to the main console as well.
                            log_status(f"[Queue] Failure details for {job_name}:\n{details_text}")
                        else:
                            log_status(
                                f"[Queue] Failure details for {job_name}: (no extended details captured)"
                            )
                        notify_job_failure_popup(job_name, job_error_details)
                        log(
                            "[Queue] Tip: If this started after changing dataset repeats/resolution/model settings, "
                            "reset this job for a fresh run (right-click job -> Reset Job (Fresh Start))."
                        )
                        failed_jobs.append(job_name)

                    if not (run_cancel_event is not None and run_cancel_event.is_set()):
                        refresh_ui_now_from_worker()
                    run_in_progress = False
                    run_cancel_event = None
                    root.after(0, update_start_button_state)

                if failed_jobs:
                    log_status(f"Queue processed with failures: {', '.join(failed_jobs)}")
                if not queue_play_mode:
                    log_status("Queue paused.")
            except Exception as exc:
                log_status(f"Queue failed unexpectedly: {exc}")
                log(traceback.format_exc())

                def show_unexpected_queue_failure_popup() -> None:
                    if not root.winfo_exists():
                        return
                    try:
                        root.bell()
                    except Exception:
                        pass
                    messagebox.showerror(
                        "Queue Failed",
                        f"Queue failed unexpectedly:\n\n{exc}\n\nSee queue log for full traceback.",
                        parent=root,
                    )

                root.after(0, show_unexpected_queue_failure_popup)
            finally:
                def finish_ui() -> None:
                    nonlocal run_in_progress, run_cancel_event, queue_worker_thread
                    if not root.winfo_exists():
                        return
                    rebuild_folder_list(force=True)
                    run_in_progress = False
                    run_cancel_event = None
                    queue_worker_thread = None
                    refresh_job_queue_list()
                    update_start_button_state()

                root.after(0, finish_ui)

        queue_worker_thread = threading.Thread(target=background_train, daemon=True)
        queue_worker_thread.start()

    scan_button = ttk.Button(controls, text="↻", style="QueueAction.TButton", width=3, command=lambda: rebuild_folder_list(force=True))
    dataset_select_toggle_var = tk.StringVar(value="Select All")
    dataset_select_toggle_button = ttk.Button(
        controls,
        textvariable=dataset_select_toggle_var,
        style="QueueAction.TButton",
        command=toggle_dataset_select_all_none,
    )
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
    run_button = ttk.Button(queue_actions_bar, text="▶ Play", command=run_queue, style="StartPlay.TButton")
    queue_reload_button = ttk.Button(
        queue_actions_bar,
        text="↻",
        style="QueueAction.TButton",
        width=3,
        command=lambda: (
            load_job_queue_from_disk(),
            refresh_job_queue_list(),
            sync_queue_row_action_buttons(),
            update_start_button_state(),
        ),
    )
    queue_select_toggle_var = tk.StringVar(value="Select All")
    queue_select_toggle_button = ttk.Button(
        queue_actions_bar,
        textvariable=queue_select_toggle_var,
        style="QueueAction.TButton",
        command=toggle_queue_select_all_none,
    )
    queue_hold_toggle_var = tk.StringVar(value="Enable Selected")
    queue_hold_toggle_button = ttk.Button(
        queue_actions_bar,
        textvariable=queue_hold_toggle_var,
        style="QueueAction.TButton",
        command=toggle_hold_selected_jobs,
    )
    queue_archive_button = ttk.Button(
        queue_actions_bar,
        text="Archive Jobs",
        style="QueueAction.TButton",
        command=archive_selected_jobs,
    )
    queue_restore_button = ttk.Button(
        queue_actions_bar,
        text="Restore Jobs",
        style="QueueAction.TButton",
        command=open_restore_jobs_dialog,
    )
    queue_delete_button = ttk.Button(
        queue_actions_bar,
        text="Delete Jobs",
        style="QueueAction.TButton",
        command=delete_selected_jobs_with_confirmation,
    )

    restore_datasets_button.configure(style="QueueAction.TButton")
    archive_datasets_button.configure(style="QueueAction.TButton")
    create_dataset_button.configure(style="QueueAction.TButton")
    metrics_viewer_button.configure(style="QueueAction.TButton")
    lora_merge_tool_button.configure(style="QueueAction.TButton")

    scan_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    dataset_select_toggle_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    archive_datasets_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    restore_datasets_button.grid(row=0, column=3, padx=(0, 8), sticky="w")
    create_dataset_button.grid(row=0, column=5, padx=(0, 8), sticky="e")
    metrics_viewer_button.grid(row=0, column=6, padx=(0, 8), sticky="e")
    lora_merge_tool_button.grid(row=0, column=7, padx=(0, 8), sticky="e")
    settings_button.grid(row=0, column=8, sticky="e")

    create_job_large_button.configure(style="CreateJobLarge.TButton")
    create_job_large_button.grid(row=0, column=0, sticky="ew")
    queue_reload_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    queue_select_toggle_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    queue_hold_toggle_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    queue_archive_button.grid(row=0, column=3, padx=(0, 8), sticky="w")
    queue_restore_button.grid(row=0, column=4, padx=(0, 8), sticky="w")
    queue_delete_button.grid(row=0, column=5, padx=(0, 8), sticky="w")
    run_button.grid(row=0, column=7, sticky="e")

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
        queue_list.event_generate(sequence, x=x, y=y, state=event.state)
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
        for item_id, action_button in queue_row_action_buttons.items():
            cell_bbox = queue_list.bbox(item_id, "actions")
            if not cell_bbox:
                action_button.place_forget()
                continue

            x, y, width, height = cell_bbox
            if width <= 0 or height <= 0:
                action_button.place_forget()
                continue

            row_bg = row_background_for_item(item_id)
            action_button.configure(bg=row_bg)

            button_height = max(20, height - 10)
            delete_width = 24
            total_width = delete_width
            start_x = x + max(2, (width - total_width) // 2)
            start_y = y + max(2, (height - button_height) // 2)

            action_button.place(x=start_x, y=start_y, width=delete_width, height=button_height)

    def build_queue_row_action_buttons() -> None:
        clear_queue_row_action_buttons()

        for item_id in queue_list.get_children():
            try:
                index = int(item_id)
            except (TypeError, ValueError):
                continue

            status = detect_job_status(job_queue[index])
            is_running_row = status == "running" and run_in_progress
            action_button = tk.Label(
                queue_list,
                text="■" if is_running_row else "✕",
                font=("Segoe UI", 13, "bold"),
                bd=0,
                padx=0,
                pady=0,
                relief="flat",
                highlightthickness=0,
                cursor="hand2",
                fg="#f0a341" if is_running_row else "#e05252",
                bg="#1c2534",
            )
            if is_running_row:
                action_button.bind("<Button-1>", lambda _event: stop_running_job())
            else:
                action_button.bind("<Button-1>", lambda _event, idx=index: delete_job_with_confirmation(idx))
            queue_row_action_buttons[item_id] = action_button

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
        update_queue_multi_action_state()

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
        settings = app_settings.load_settings()
        runtime_config = runtime_config_from_settings(settings)
        if runtime_config is None:
            print(f"Missing settings file: {app_settings.SETTINGS_FILE}")
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
            default_caption_keyword=settings.get(app_settings.DEFAULT_CAPTION_KEYWORD_KEY, ""),
            resolution=DEFAULT_RESOLUTION,
            network_dim=DEFAULT_NETWORK_DIM,
            network_alpha=DEFAULT_NETWORK_ALPHA,
            optimizer_type="prodigy",
            optimizer_args=DEFAULT_PRODIGY_OPTIMIZER_ARGS,
            learning_rate=DEFAULT_LEARNING_RATE,
            train_steps=DEFAULT_TRAIN_STEPS,
            enable_compile_optimizations=(
                settings.get(app_settings.ENABLE_COMPILE_OPTIMIZATIONS_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_cuda_allow_tf32=(
                settings.get(app_settings.ENABLE_CUDA_ALLOW_TF32_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_cuda_cudnn_benchmark=(
                settings.get(app_settings.ENABLE_CUDA_CUDNN_BENCHMARK_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_fp8_dit=(
                settings.get(app_settings.ENABLE_FP8_DIT_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            enable_gradient_checkpointing_cpu_offload=(
                settings.get(app_settings.ENABLE_GRADIENT_CHECKPOINTING_CPU_OFFLOAD_KEY, "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            enable_training_logging=(
                settings.get(app_settings.TRAIN_ENABLE_LOGGING_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
            ),
            training_log_backend=get_train_log_backend_setting(settings),
            training_log_tracker_name=settings.get(app_settings.TRAIN_LOG_TRACKER_NAME_KEY, "").strip(),
            stream_training_output=(
                settings.get(app_settings.TRAIN_STREAM_TO_LOGGER_KEY, "0").strip().lower() in {"1", "true", "yes", "on"}
            ),
            auto_cleanup_states=(
                settings.get(app_settings.TRAIN_AUTO_CLEANUP_STATES_KEY, "1").strip().lower() in {"1", "true", "yes", "on"}
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
