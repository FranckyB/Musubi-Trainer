import subprocess
import sys
import argparse
import re
import configparser
import ctypes
from pathlib import Path
from typing import Callable, Iterable

from .app_settings import (
    KLEIN_DIT_KEY,
    KLEIN_MODEL_VERSION_KEY,
    KLEIN_TEXT_ENCODER_KEY,
    KLEIN_VAE_KEY,
    LTX_DIT_KEY,
    LTX_MODEL_VERSION_KEY,
    LTX_TEXT_ENCODER_KEY,
    LTX_VAE_KEY,
    MUSUBI_DIR_KEY,
    SETTINGS_FILE,
    WINDOW_HEIGHT_KEY,
    WINDOW_WIDTH_KEY,
    WINDOW_X_KEY,
    WINDOW_Y_KEY,
    load_settings,
    load_window_size,
    load_window_position,
    save_settings,
)
from .runtime_config import RuntimeConfig, runtime_config_from_settings


# Training settings
RESOLUTION = 1024
NETWORK_DIM = 32
NETWORK_ALPHA = 32
LR = "1e-4"
STEPS = 3000

# Model files
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"
DATASET_ORDER_KEY = "dataset_order"


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
    images_dir = training_dir / dataset_name / "images"
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])


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


def prep_dataset_minimal(training_dir: Path, dataset_name: str) -> dict[str, int | bool]:
    dataset_dir = training_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_toml = dataset_dir / "dataset.toml"

    had_dataset_toml = dataset_toml.exists()
    if not had_dataset_toml:
        dataset_toml.write_text(
            "\n".join(
                [
                    "[general]",
                    "shuffle_caption = false",
                    'caption_extension = ".txt"',
                    "keep_tokens = 0",
                    "",
                    "[[datasets]]",
                    f"resolution = {RESOLUTION}",
                    "batch_size = 1",
                    "enable_bucket = true",
                    "bucket_no_upscale = false",
                    "",
                    "  [[datasets.subsets]]",
                    '  image_dir = "images"',
                    '  caption_extension = ".txt"',
                    "  num_repeats = 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    created = 0
    for image_path in dataset_image_files(training_dir, dataset_name):
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            continue
        caption_path.write_text(image_path.stem, encoding="utf-8")
        created += 1

    return {"had_dataset_toml": had_dataset_toml, "created": created}


def run_command(args: Iterable[str], cwd: Path) -> None:
    result = subprocess.run(list(args), cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(args)}")


def run_steps_for_model(
    runtime_config: RuntimeConfig,
    model_name: str,
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
    resume_checkpoint: Path | None,
    train_steps_override: int | None,
    logger: Callable[[str], None],
) -> None:
    def require_model_file(path_value: Path | None, label: str) -> Path:
        if path_value is None:
            raise RuntimeError(f"{label} is not configured. Open Settings and select a file for {label}.")
        if not path_value.is_file():
            raise RuntimeError(f"{label} file does not exist: {path_value}")
        return path_value

    dataset_config = runtime_config.training_dir / model_name / "dataset.toml"
    output_dir = runtime_config.training_dir / model_name / "output"
    output_name = f"{model_name}_Klein"

    logger("=" * 58)
    logger(f"MODEL_NAME: {model_name}")
    logger(f"dataset_config: {dataset_config}")
    logger(f"output_dir:     {output_dir}")

    if do_prep_dataset:
        prep_result = prep_dataset_minimal(runtime_config.training_dir, model_name)
        toml_status = "existed" if bool(prep_result["had_dataset_toml"]) else "created"
        logger(f"  prep: dataset.toml {toml_status}, captions created {prep_result['created']}")

    if do_cache_latents:
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
        run_command(
            [
                sys.executable,
                "flux_2_cache_latents.py",
                "--dataset_config",
                str(dataset_config),
                "--vae",
                str(vae_path),
                "--batch_size",
                "16",
                "--model_version",
                runtime_config.model_version,
            ],
            cwd=runtime_config.musubi_dir,
        )

    if do_cache_text:
        text_encoder_path = require_model_file(runtime_config.text_encoder, "Klein Text Encoder")
        run_command(
            [
                sys.executable,
                "flux_2_cache_text_encoder_outputs.py",
                "--dataset_config",
                str(dataset_config),
                "--text_encoder",
                str(text_encoder_path),
                "--batch_size",
                "16",
                "--model_version",
                runtime_config.model_version,
            ],
            cwd=runtime_config.musubi_dir,
        )

    if do_train:
        dit_path = require_model_file(runtime_config.dit, "Klein Model")
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
        text_encoder_path = require_model_file(runtime_config.text_encoder, "Klein Text Encoder")
        train_steps = train_steps_override if train_steps_override is not None else STEPS
        run_command(
            [
                sys.executable,
                "flux_2_train_network.py",
                "--dit", str(dit_path),
                "--vae", str(vae_path),
                "--vae_dtype", "bf16",
                "--text_encoder", str(text_encoder_path),
                "--model_version", runtime_config.model_version,
                "--optimizer_type", "adamw8bit",
                "--timestep_sampling", "flux2_shift",
                "--dataset_config", str(dataset_config),
                "--output_dir", str(output_dir),
                "--output_name", output_name,
                "--network_module", "networks.lora_flux_2",
                "--network_dim", str(NETWORK_DIM),
                "--network_alpha", str(NETWORK_ALPHA),
                "--learning_rate", LR,
                "--max_train_steps", str(train_steps),
                "--mixed_precision", "bf16",
                "--sdpa",
                "--gradient_checkpointing",
                "--persistent_data_loader_workers",
                "--max_data_loader_n_workers", "2",
                "--save_every_n_steps", "250",
                "--seed", "42",
                *( ["--resume", str(resume_checkpoint)] if resume_checkpoint is not None else [] ),
            ],
            cwd=runtime_config.musubi_dir,
        )


def train_models(
    runtime_config: RuntimeConfig,
    model_names: list[str],
    logger: Callable[[str], None],
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
) -> int:
    if not model_names:
        logger("No valid model names entered. Exiting.")
        return 1

    if not (do_prep_dataset or do_cache_latents or do_cache_text or do_train):
        logger("No steps selected. Select at least one step.")
        return 1

    selected_steps: list[str] = []
    if do_prep_dataset:
        selected_steps.append("prep_dataset")
    if do_cache_latents:
        selected_steps.append("cache_latents")
    if do_cache_text:
        selected_steps.append("cache_text")
    if do_train:
        selected_steps.append("train")

    logger(f"Queued models: {', '.join(model_names)}")
    logger(f"Selected steps: {', '.join(selected_steps)}")

    all_steps_selected = do_prep_dataset and do_cache_latents and do_cache_text and do_train

    for index, model_name in enumerate(model_names, start=1):
        logger("")
        logger(f"[{index}/{len(model_names)}] Starting training for: {model_name}")

        effective_do_prep_dataset = do_prep_dataset
        effective_do_cache_latents = do_cache_latents
        effective_do_cache_text = do_cache_text
        effective_do_train = do_train
        resume_checkpoint, resume_step = latest_checkpoint_for_dataset(runtime_config.training_dir, model_name)
        train_steps_override: int | None = None

        if resume_step >= STEPS:
            logger(f"  checkpoint already complete at step {resume_step}: skipping")
            continue

        if resume_checkpoint is not None and resume_step > 0:
            train_steps_override = max(1, STEPS - resume_step)
            logger(f"  resuming from {resume_checkpoint.name} (step {resume_step}), remaining steps {train_steps_override}")

        if all_steps_selected:
            status = dataset_status(runtime_config.training_dir, model_name)
            if status["ready_to_train"]:
                effective_do_prep_dataset = False
                effective_do_cache_latents = False
                effective_do_cache_text = False
                effective_do_train = True
                logger("  dataset is already green (S1/S2/S3 ready): skipping prep/cache and running train only")

        try:
            run_steps_for_model(
                runtime_config,
                model_name,
                do_prep_dataset=effective_do_prep_dataset,
                do_cache_latents=effective_do_cache_latents,
                do_cache_text=effective_do_cache_text,
                do_train=effective_do_train,
                resume_checkpoint=resume_checkpoint if train_steps_override is not None else None,
                train_steps_override=train_steps_override,
                logger=logger,
            )
            logger(f"[{index}/{len(model_names)}] Completed: {model_name}")
        except Exception as exc:
            logger(f"Training failed for '{model_name}': {exc}")
            return 1

    logger("")
    logger("All model runs completed.")
    return 0


def launch_ui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        from PIL import Image, ImageDraw, ImageTk
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
    workspace_dir = Path(__file__).resolve().parent.parent
    ui_config = load_ui_config(Path(__file__).resolve().parent / "app.config")
    default_models_dir = workspace_dir / "Models"

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

    root = tk.Tk()
    root.title("Musubi Training Launcher")
    root.geometry(f"{ui_config['window_width']}x{ui_config['window_height']}")
    root.minsize(ui_config["min_window_width"], ui_config["min_window_height"])
    root.configure(bg=bg_root)
    set_dark_title_bar(root)

    settings_state = load_settings()
    window_position_applied = False

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
        if window_position_applied and root.winfo_exists():
            save_main_window_position_now()
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
        padding=(10, 4),
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
        "StartDisabled.TButton",
        background=color_start_disabled,
        foreground="#c6c6c6",
        padding=(10, 8),
        borderwidth=2,
        bordercolor="#4a4a4a",
        lightcolor="#565656",
        darkcolor="#2f2f2f",
        relief="raised",
        font=("Segoe UI", 10, "bold"),
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
        padding=(10, 8),
        borderwidth=2,
        bordercolor="#2ea95a",
        lightcolor="#63e394",
        darkcolor="#238149",
        relief="raised",
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "StartEnabled.TButton",
        background=[("active", color_start_enabled_active), ("disabled", color_start_disabled)],
        foreground=[("active", "#ffffff"), ("disabled", "#c6c6c6")],
    )

    vars_by_name: dict[str, tk.BooleanVar] = {}
    card_widgets: list[tk.Widget] = []
    thumbnail_cache: dict[tuple[str, bool, int, int], ImageTk.PhotoImage] = {}
    first_image_cache: dict[str, Path | None] = {}
    checkpoint_cache: dict[str, tuple[Path | None, int]] = {}
    run_state_by_name: dict[str, str] = {}
    card_frame_by_name: dict[str, ttk.Frame] = {}
    resize_after_id: str | None = None
    last_canvas_width = 0
    run_in_progress = False
    dataset_order: list[str] = load_dataset_order(settings_state)
    drag_dataset_name: str | None = None
    drag_moved = False
    runtime_config = runtime_config_from_settings(settings_state)

    def persist_dataset_order() -> None:
        nonlocal settings_state
        save_dataset_order(settings_state, dataset_order)
        save_settings(settings_state)

    def open_settings_dialog(required: bool) -> RuntimeConfig | None:
        current_dir = ""
        if runtime_config is not None:
            current_dir = str(runtime_config.musubi_dir)
        current_klein_model_version = settings_state.get(KLEIN_MODEL_VERSION_KEY, "").strip() or "klein-base-9b"
        current_klein_dit = settings_state.get(KLEIN_DIT_KEY, "").strip()
        current_klein_vae = settings_state.get(KLEIN_VAE_KEY, "").strip()
        current_klein_text_encoder = settings_state.get(KLEIN_TEXT_ENCODER_KEY, "").strip()
        current_ltx_model_version = settings_state.get(LTX_MODEL_VERSION_KEY, "").strip()
        current_ltx_dit = settings_state.get(LTX_DIT_KEY, "").strip()
        current_ltx_vae = settings_state.get(LTX_VAE_KEY, "").strip()
        current_ltx_text_encoder = settings_state.get(LTX_TEXT_ENCODER_KEY, "").strip()

        result: RuntimeConfig | None = None
        dialog = tk.Toplevel(root)
        dialog.title("Settings")
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

        musubi_section = ttk.LabelFrame(frame, text="Musubi-Tuner", padding=8)
        musubi_section.grid(row=0, column=0, sticky="ew")
        musubi_section.columnconfigure(1, weight=1)

        klein_section = ttk.LabelFrame(frame, text="Klein", padding=8)
        klein_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        klein_section.columnconfigure(1, weight=1)

        ltx_section = ttk.LabelFrame(frame, text="LTX", padding=8)
        ltx_section.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ltx_section.columnconfigure(1, weight=1)

        selected_musubi_path = current_dir
        selected_klein_dit = current_klein_dit
        selected_klein_vae = current_klein_vae
        selected_klein_text_encoder = current_klein_text_encoder
        selected_ltx_dit = current_ltx_dit
        selected_ltx_vae = current_ltx_vae
        selected_ltx_text_encoder = current_ltx_text_encoder

        musubi_display_var = tk.StringVar(value=current_dir if current_dir else "(none)")
        klein_model_version_var = tk.StringVar(value=current_klein_model_version)
        klein_dit_var = tk.StringVar(value=current_klein_dit if current_klein_dit else "(none)")
        klein_vae_var = tk.StringVar(value=current_klein_vae if current_klein_vae else "(none)")
        klein_text_encoder_var = tk.StringVar(value=current_klein_text_encoder if current_klein_text_encoder else "(none)")
        ltx_model_version_var = tk.StringVar(value=current_ltx_model_version)
        ltx_dit_var = tk.StringVar(value=current_ltx_dit if current_ltx_dit else "(none)")
        ltx_vae_var = tk.StringVar(value=current_ltx_vae if current_ltx_vae else "(none)")
        ltx_text_encoder_var = tk.StringVar(value=current_ltx_text_encoder if current_ltx_text_encoder else "(none)")

        ttk.Label(musubi_section, text="Musubi-Tuner folder:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        musubi_display = ttk.Label(
            musubi_section, textvariable=musubi_display_var, anchor="w", style="PathDisplay.TLabel", padding=(6, 4)
        )
        musubi_display.grid(row=0, column=1, sticky="ew")

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
            nonlocal selected_musubi_path
            picked = filedialog.askdirectory(
                parent=dialog,
                title="Select Musubi-Tuner folder",
                initialdir=selected_musubi_path or str(Path.home()),
            )
            if picked:
                selected_musubi_path = picked
                musubi_display_var.set(picked)

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

        def save_and_close() -> None:
            nonlocal result, settings_state
            if not selected_musubi_path:
                messagebox.showerror("Missing folder", "Musubi-Tuner folder is not set.", parent=dialog)
                return

            musubi_path = Path(selected_musubi_path).expanduser()
            if not musubi_path.exists() or not musubi_path.is_dir():
                messagebox.showerror("Invalid folder", "Choose a valid Musubi-Tuner folder.", parent=dialog)
                return

            klein_model_version = klein_model_version_var.get().strip()
            if not klein_model_version:
                messagebox.showerror("Missing value", "Klein model version is required.", parent=dialog)
                return

            for label, raw_path in (
                ("Klein Model", selected_klein_dit),
                ("Klein VAE", selected_klein_vae),
                ("Klein Text Encoder", selected_klein_text_encoder),
            ):
                if not raw_path:
                    continue
                resolved = Path(raw_path).expanduser()
                if not resolved.exists() or not resolved.is_file():
                    messagebox.showerror("Invalid file", f"Choose a valid file for {label}.", parent=dialog)
                    return

            for label, raw_path in (
                ("LTX Model", selected_ltx_dit),
                ("LTX VAE", selected_ltx_vae),
                ("LTX Text Encoder", selected_ltx_text_encoder),
            ):
                if not raw_path:
                    continue
                resolved = Path(raw_path).expanduser()
                if not resolved.exists() or not resolved.is_file():
                    messagebox.showerror("Invalid file", f"Choose a valid file for {label}.", parent=dialog)
                    return

            settings_state[MUSUBI_DIR_KEY] = str(musubi_path)
            settings_state[KLEIN_MODEL_VERSION_KEY] = klein_model_version
            settings_state[KLEIN_DIT_KEY] = str(Path(selected_klein_dit).expanduser()) if selected_klein_dit else ""
            settings_state[KLEIN_VAE_KEY] = str(Path(selected_klein_vae).expanduser()) if selected_klein_vae else ""
            settings_state[KLEIN_TEXT_ENCODER_KEY] = (
                str(Path(selected_klein_text_encoder).expanduser()) if selected_klein_text_encoder else ""
            )
            settings_state[LTX_MODEL_VERSION_KEY] = ltx_model_version_var.get().strip()
            settings_state[LTX_DIT_KEY] = str(Path(selected_ltx_dit).expanduser()) if selected_ltx_dit else ""
            settings_state[LTX_VAE_KEY] = str(Path(selected_ltx_vae).expanduser()) if selected_ltx_vae else ""
            settings_state[LTX_TEXT_ENCODER_KEY] = (
                str(Path(selected_ltx_text_encoder).expanduser()) if selected_ltx_text_encoder else ""
            )
            save_settings(settings_state)
            result = runtime_config_from_settings(settings_state)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        ttk.Button(musubi_section, text="Browse Folder", command=browse_musubi).grid(row=0, column=2, padx=(8, 0))
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
            row=3, column=0, sticky="w", pady=(10, 8)
        )

        button_row = ttk.Frame(frame)
        button_row.grid(row=4, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=cancel_and_close).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=1)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.update_idletasks()

        content_w = frame.winfo_reqwidth() + 20
        content_h = frame.winfo_reqheight() + 20
        win_w = max(780, min(1080, content_w))
        win_h = max(520, content_h)
        dialog.geometry(f"{win_w}x{win_h}")
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

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=0)
    root.rowconfigure(2, weight=5, minsize=360)
    root.rowconfigure(3, weight=0)
    root.rowconfigure(4, weight=2)

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
        return True

    list_container = ttk.LabelFrame(root, text="Datasets (click thumbnail/title to toggle)", padding=8)
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

    start_bar = ttk.Frame(root, padding=(8, 0, 8, 8))
    start_bar.grid(row=3, column=0, sticky="ew")
    start_bar.columnconfigure(0, weight=1)

    log_box = ScrolledText(root, height=10, wrap="word")
    log_box.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
    log_box.configure(bg="#0e1319", fg=fg_text, insertbackground=fg_text, relief="flat", borderwidth=0)

    def log(message: str) -> None:
        log_box.insert("end", message + "\n")
        log_box.see("end")
        root.update_idletasks()

    def first_image_path(dataset_name: str) -> Path | None:
        cached = first_image_cache.get(dataset_name)
        if dataset_name in first_image_cache:
            return cached

        images_dir = runtime_config.training_dir / dataset_name / "images"
        if not images_dir.exists():
            first_image_cache[dataset_name] = None
            return None

        image_candidates = sorted(
            [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        )
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

        draw = ImageDraw.Draw(image)
        badge_d = max(18, thumb_px // 5)
        pad = 6
        x1 = thumb_px - badge_d - pad
        y1 = pad
        x2 = x1 + badge_d
        y2 = y1 + badge_d

        if run_state == "done":
            draw.ellipse((x1, y1, x2, y2), fill=color_green)
            line_w = max(2, badge_d // 8)
            draw.line(
                (x1 + badge_d * 0.24, y1 + badge_d * 0.56, x1 + badge_d * 0.43, y1 + badge_d * 0.75),
                fill="white",
                width=line_w,
            )
            draw.line(
                (x1 + badge_d * 0.43, y1 + badge_d * 0.75, x1 + badge_d * 0.78, y1 + badge_d * 0.30),
                fill="white",
                width=line_w,
            )
        elif run_state == "in_progress":
            draw.ellipse((x1, y1, x2, y2), fill="#d08a22")
            line_w = max(2, badge_d // 7)
            bar_pad_x = max(3, badge_d // 4)
            bar_top = y1 + max(3, badge_d // 4)
            bar_bottom = y2 - max(3, badge_d // 4)
            draw.line((x1 + bar_pad_x, bar_top, x1 + bar_pad_x, bar_bottom), fill="white", width=line_w)
            draw.line((x2 - bar_pad_x, bar_top, x2 - bar_pad_x, bar_bottom), fill="white", width=line_w)

        photo = ImageTk.PhotoImage(image)
        thumbnail_cache[cache_key] = photo
        return photo

    def toggle_dataset(name: str) -> None:
        if run_state_by_name.get(name) == "done":
            return
        current = vars_by_name[name].get()
        vars_by_name[name].set(not current)

    def apply_card_style(name: str) -> None:
        card = card_frame_by_name.get(name)
        if card is None:
            return
        run_state = run_state_by_name.get(name, "pending")
        if run_state == "done":
            card.configure(style="DoneCard.TFrame")
            return
        selected = vars_by_name.get(name).get() if name in vars_by_name else False
        card.configure(style=("SelectedCard.TFrame" if selected else "Card.TFrame"))

    def on_card_press(name: str) -> None:
        nonlocal drag_dataset_name, drag_moved
        drag_dataset_name = name
        drag_moved = False

    def on_card_motion() -> None:
        nonlocal drag_moved
        if drag_dataset_name is not None:
            drag_moved = True

    def on_card_release(target_name: str) -> str:
        nonlocal drag_dataset_name, drag_moved, dataset_order
        if drag_dataset_name is None:
            return "break"

        source_name = drag_dataset_name
        moved = drag_moved
        drag_dataset_name = None
        drag_moved = False

        if moved:
            if source_name != target_name and source_name in dataset_order and target_name in dataset_order:
                source_idx = dataset_order.index(source_name)
                target_idx = dataset_order.index(target_name)
                dataset_order.insert(target_idx, dataset_order.pop(source_idx))
                persist_dataset_order()
                rebuild_folder_list(force=True)
            return "break"

        toggle_dataset(source_name)
        apply_card_style(source_name)
        update_start_button_state()
        return "break"

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

        scanned_names = scan_training_folders(runtime_config.training_dir)
        existing = [name for name in dataset_order if name in scanned_names]
        new_names = [name for name in scanned_names if name not in existing]
        normalized_order = new_names + existing
        if normalized_order != dataset_order:
            dataset_order = normalized_order
            persist_dataset_order()

        names = dataset_order
        stale_names = [name for name in list(first_image_cache.keys()) if name not in names]
        for stale_name in stale_names:
            first_image_cache.pop(stale_name, None)
            checkpoint_cache.pop(stale_name, None)

        if not names:
            empty_label = ttk.Label(inner, text="No folders found.")
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
            checkpoint_info = checkpoint_cache.get(name)
            if checkpoint_info is None:
                checkpoint_info = latest_checkpoint_for_dataset(runtime_config.training_dir, name)
                checkpoint_cache[name] = checkpoint_info
            checkpoint_path, checkpoint_step = checkpoint_info
            train_state = "done" if checkpoint_step >= STEPS else ("in_progress" if checkpoint_step > 0 else "pending")
            run_state_by_name[name] = train_state

            var = tk.BooleanVar(value=(name in selected_before) and train_state != "done")
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

            title_style = "DoneCardTitle.TLabel" if train_state == "done" else "CardTitle.TLabel"
            meta_style = "DoneCardMeta.TLabel" if train_state == "done" else "CardMeta.TLabel"

            image_label = ttk.Label(card, image=thumb, style=title_style, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")
            title_label = ttk.Label(card, text=name, style=title_style, anchor="center")
            title_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            status_text = ""
            if train_state == "done":
                status_text = "COMPLETED"
            elif train_state == "in_progress" and checkpoint_path is not None:
                status_text = f"RESUME {checkpoint_step}/{STEPS}"
            else:
                status_text = "START"
            status_label = ttk.Label(card, text=status_text, style=meta_style, anchor="center")
            status_label.grid(row=2, column=0, sticky="ew", pady=(2, 8))

            for clickable in (card, image_label, title_label, status_label):
                clickable.bind("<ButtonPress-1>", lambda _e, n=name: on_card_press(n))
                clickable.bind("<B1-Motion>", lambda _e: on_card_motion())
                clickable.bind("<ButtonRelease-1>", lambda _e, n=name: on_card_release(n))

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

    def selected_names() -> list[str]:
        return [name for name, var in vars_by_name.items() if var.get()]

    def update_start_button_state() -> None:
        if run_in_progress:
            run_button.configure(style="StartDisabled.TButton")
            run_button.state(["disabled"])
            return

        has_selection = bool(selected_names())
        if has_selection:
            run_button.configure(style="StartEnabled.TButton")
            run_button.state(["!disabled"])
        else:
            run_button.configure(style="StartDisabled.TButton")
            run_button.state(["disabled"])

    def select_all() -> None:
        for name, var in vars_by_name.items():
            var.set(run_state_by_name.get(name) != "done")
        update_start_button_state()

    def clear_selection() -> None:
        for var in vars_by_name.values():
            var.set(False)
        update_start_button_state()

    def run_selected() -> None:
        nonlocal run_in_progress
        names = selected_names()
        if not names:
            messagebox.showinfo("Nothing selected", "Select at least one folder.")
            return

        run_in_progress = True
        update_start_button_state()
        try:
            log("")
            train_models(
                runtime_config,
                names,
                logger=log,
                do_prep_dataset=True,
                do_cache_latents=True,
                do_cache_text=True,
                do_train=True,
            )
            rebuild_folder_list(force=True)
        finally:
            run_in_progress = False
            update_start_button_state()

    refresh_button = ttk.Button(controls, text="Refresh", command=lambda: rebuild_folder_list(force=True))
    select_all_button = ttk.Button(controls, text="Select All", command=select_all)
    clear_button = ttk.Button(controls, text="Clear", command=clear_selection)
    settings_button = ttk.Button(controls, text="Settings", command=apply_settings_from_dialog)
    run_button = ttk.Button(start_bar, text="START", command=run_selected, style="StartDisabled.TButton")

    refresh_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    select_all_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    clear_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    settings_button.grid(row=0, column=4, sticky="e")
    run_button.grid(row=0, column=0, sticky="ew")

    def on_canvas_configure(event: tk.Event) -> None:
        request_relayout(event.width)
        update_scrollbar_visibility()

    canvas.bind("<Configure>", on_canvas_configure)

    rebuild_folder_list(force=True)
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
        runtime_config = runtime_config_from_settings(load_settings())
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
            logger=print,
            do_prep_dataset=do_prep_dataset,
            do_cache_latents=do_cache_latents,
            do_cache_text=do_cache_text,
            do_train=do_train,
        )

    return launch_ui()


if __name__ == "__main__":
    raise SystemExit(main())
