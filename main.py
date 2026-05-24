import subprocess
import sys
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List


# Training settings
RESOLUTION = 1024
NETWORK_DIM = 32
NETWORK_ALPHA = 32
LR = "1e-4"
STEPS = 3000

# Base directories
MUSUBI_DIR = Path(r"D:\musubi-tuner")
TRAINING_DIR = MUSUBI_DIR / "training"

# Model files
MODEL_VERSION = "klein-base-9b"
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
WINDOW_X_KEY = "window_x"
WINDOW_Y_KEY = "window_y"
MUSUBI_DIR_KEY = "musubi_dir"
KLEIN_MODEL_DIR_KEY = "klein_model_dir"
LTX_MODEL_DIR_KEY = "ltx_model_dir"


@dataclass(frozen=True)
class RuntimeConfig:
    musubi_dir: Path
    training_dir: Path
    model_version: str
    dit: Path
    vae: Path
    text_encoder: Path


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
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


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


def runtime_config_from_settings(settings: dict[str, str]) -> RuntimeConfig | None:
    musubi_raw = settings.get(MUSUBI_DIR_KEY, "").strip()
    if not musubi_raw:
        return None

    musubi_dir = Path(musubi_raw).expanduser()
    klein_model_raw = settings.get(KLEIN_MODEL_DIR_KEY, "").strip()
    if klein_model_raw:
        klein_model_dir = Path(klein_model_raw).expanduser()
    else:
        klein_model_dir = musubi_dir / "models" / "klein"

    return RuntimeConfig(
        musubi_dir=musubi_dir,
        training_dir=musubi_dir / "training",
        model_version=MODEL_VERSION,
        dit=klein_model_dir / "flux-2-klein-base-9b.safetensors",
        vae=klein_model_dir / "ae.safetensors",
        text_encoder=klein_model_dir / "text_encoder" / "model-00001-of-00004.safetensors",
    )


def parse_model_names(raw: str) -> List[str]:
    """Split comma-separated names and trim surrounding whitespace."""
    names = [part.strip() for part in raw.split(",")]
    return [name for name in names if name]


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
    logger: Callable[[str], None],
) -> None:
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
        run_command(
            [
                sys.executable,
                "flux_2_cache_latents.py",
                "--dataset_config",
                str(dataset_config),
                "--vae",
                str(runtime_config.vae),
                "--batch_size",
                "16",
                "--model_version",
                runtime_config.model_version,
            ],
            cwd=runtime_config.musubi_dir,
        )

    if do_cache_text:
        run_command(
            [
                sys.executable,
                "flux_2_cache_text_encoder_outputs.py",
                "--dataset_config",
                str(dataset_config),
                "--text_encoder",
                str(runtime_config.text_encoder),
                "--batch_size",
                "16",
                "--model_version",
                runtime_config.model_version,
            ],
            cwd=runtime_config.musubi_dir,
        )

    if do_train:
        run_command(
            [
                sys.executable,
                "flux_2_train_network.py",
                "--dit", str(runtime_config.dit),
                "--vae", str(runtime_config.vae),
                "--vae_dtype", "bf16",
                "--text_encoder", str(runtime_config.text_encoder),
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
                "--max_train_steps", str(STEPS),
                "--mixed_precision", "bf16",
                "--sdpa",
                "--gradient_checkpointing",
                "--persistent_data_loader_workers",
                "--max_data_loader_n_workers", "2",
                "--save_every_n_steps", "250",
                "--seed", "42",
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
    root.title("Klein Training Launcher")
    root.geometry("660x840")
    root.minsize(620, 800)
    root.configure(bg=bg_root)

    settings_state = load_settings()
    window_position_applied = False
    save_position_after_id: str | None = None

    def apply_initial_main_window_position() -> None:
        nonlocal window_position_applied
        root.update_idletasks()

        width = max(root.winfo_width(), root.winfo_reqwidth())
        height = max(root.winfo_height(), root.winfo_reqheight())
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

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
        save_settings(settings_state)

    def schedule_main_window_position_save(_event: tk.Event) -> None:
        nonlocal save_position_after_id
        if not window_position_applied:
            return
        if root.state() != "normal":
            return
        if save_position_after_id is not None:
            root.after_cancel(save_position_after_id)
        save_position_after_id = root.after(250, save_main_window_position_now)

    root.bind("<Configure>", schedule_main_window_position_save)
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
    style.configure("CardTitle.TLabel", background=bg_card, foreground=fg_text)
    style.configure("CardMeta.TLabel", background=bg_card, foreground=fg_muted)
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

    vars_by_name: dict[str, tk.BooleanVar] = {}
    card_widgets: list[tk.Widget] = []
    thumbnail_refs: list[ImageTk.PhotoImage] = []
    resize_after_id: str | None = None
    dataset_order: list[str] = []
    active_dataset_name: str | None = None
    runtime_config = runtime_config_from_settings(settings_state)

    def open_settings_dialog(required: bool) -> RuntimeConfig | None:
        current_dir = ""
        if runtime_config is not None:
            current_dir = str(runtime_config.musubi_dir)
        current_klein_dir = settings_state.get(KLEIN_MODEL_DIR_KEY, "").strip()
        current_ltx_dir = settings_state.get(LTX_MODEL_DIR_KEY, "").strip()

        result: RuntimeConfig | None = None
        dialog = tk.Toplevel(root)
        dialog.title("Settings")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=bg_panel)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        frame = ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        musubi_section = ttk.LabelFrame(frame, text="Musubi-Tuner", padding=8)
        musubi_section.grid(row=0, column=0, sticky="ew")
        musubi_section.columnconfigure(1, weight=1)

        model_section = ttk.LabelFrame(frame, text="Model Configs", padding=8)
        model_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        model_section.columnconfigure(1, weight=1)

        selected_musubi_path = current_dir
        selected_klein_path = current_klein_dir
        selected_ltx_path = current_ltx_dir

        musubi_display_var = tk.StringVar(value=current_dir if current_dir else "(none)")
        klein_display_var = tk.StringVar(value=current_klein_dir if current_klein_dir else "(none)")
        ltx_display_var = tk.StringVar(value=current_ltx_dir if current_ltx_dir else "(none)")

        ttk.Label(musubi_section, text="Musubi-Tuner folder:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        musubi_display = ttk.Label(
            musubi_section, textvariable=musubi_display_var, anchor="w", relief="sunken", padding=(6, 4)
        )
        musubi_display.grid(row=0, column=1, sticky="ew")

        ttk.Label(model_section, text="Klein model folder:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        klein_display = ttk.Label(
            model_section, textvariable=klein_display_var, anchor="w", relief="sunken", padding=(6, 4)
        )
        klein_display.grid(row=0, column=1, sticky="ew")

        ttk.Label(model_section, text="LTX model folder:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ltx_display = ttk.Label(
            model_section, textvariable=ltx_display_var, anchor="w", relief="sunken", padding=(6, 4)
        )
        ltx_display.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        def browse_musubi() -> None:
            nonlocal selected_musubi_path
            picked = filedialog.askdirectory(initialdir=selected_musubi_path or str(Path.home()))
            if picked:
                selected_musubi_path = picked
                musubi_display_var.set(picked)

        def browse_klein() -> None:
            nonlocal selected_klein_path
            initial_dir = selected_klein_path or selected_musubi_path or str(Path.home())
            picked = filedialog.askdirectory(initialdir=initial_dir)
            if picked:
                selected_klein_path = picked
                klein_display_var.set(picked)

        def browse_ltx() -> None:
            nonlocal selected_ltx_path
            initial_dir = selected_ltx_path or selected_musubi_path or str(Path.home())
            picked = filedialog.askdirectory(initialdir=initial_dir)
            if picked:
                selected_ltx_path = picked
                ltx_display_var.set(picked)

        def save_and_close() -> None:
            nonlocal result, settings_state
            if not selected_musubi_path:
                messagebox.showerror("Missing folder", "Musubi-Tuner folder is not set.", parent=dialog)
                return

            musubi_path = Path(selected_musubi_path).expanduser()
            if not musubi_path.exists() or not musubi_path.is_dir():
                messagebox.showerror("Invalid folder", "Choose a valid Musubi-Tuner folder.", parent=dialog)
                return

            if selected_klein_path:
                klein_path = Path(selected_klein_path).expanduser()
                if not klein_path.exists() or not klein_path.is_dir():
                    messagebox.showerror("Invalid folder", "Choose a valid Klein model folder.", parent=dialog)
                    return

            if selected_ltx_path:
                ltx_path = Path(selected_ltx_path).expanduser()
                if not ltx_path.exists() or not ltx_path.is_dir():
                    messagebox.showerror("Invalid folder", "Choose a valid LTX model folder.", parent=dialog)
                    return

            settings_state[MUSUBI_DIR_KEY] = str(musubi_path)
            settings_state[KLEIN_MODEL_DIR_KEY] = str(Path(selected_klein_path).expanduser()) if selected_klein_path else ""
            settings_state[LTX_MODEL_DIR_KEY] = str(Path(selected_ltx_path).expanduser()) if selected_ltx_path else ""
            save_settings(settings_state)
            result = runtime_config_from_settings(settings_state)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        ttk.Button(musubi_section, text="Browse", command=browse_musubi).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(model_section, text="Browse", command=browse_klein).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(model_section, text="Browse", command=browse_ltx).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(frame, text="LTX path is saved for future support.").grid(
            row=2, column=0, sticky="w", pady=(10, 8)
        )

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=cancel_and_close).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=1)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.update_idletasks()

        content_w = frame.winfo_reqwidth() + 20
        content_h = frame.winfo_reqheight() + 20
        win_w = max(620, min(840, content_w))
        win_h = max(280, content_h)
        dialog.geometry(f"{win_w}x{win_h}")
        center_window(dialog)
        musubi_display.focus_set()
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
    root.rowconfigure(2, weight=0)
    root.rowconfigure(3, weight=5, minsize=360)
    root.rowconfigure(4, weight=2)

    header = ttk.Frame(root, padding=8)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    training_path_var = tk.StringVar(value=f"Training folder: {runtime_config.training_dir}")
    ttk.Label(header, textvariable=training_path_var).grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.grid(row=1, column=0, sticky="ew")

    def apply_settings_from_dialog(required: bool = False) -> bool:
        nonlocal runtime_config, dataset_order, active_dataset_name
        updated = open_settings_dialog(required=required)
        if updated is None:
            return False

        runtime_config = updated
        training_path_var.set(f"Training folder: {runtime_config.training_dir}")
        dataset_order = []
        active_dataset_name = None
        rebuild_folder_list(force=True)
        return True

    menubar = tk.Menu(root)
    settings_menu = tk.Menu(menubar, tearoff=False)
    settings_menu.add_command(label="Settings", command=apply_settings_from_dialog)
    menubar.add_cascade(label="Settings", menu=settings_menu)
    root.config(menu=menubar)

    steps_frame = ttk.LabelFrame(root, text="Steps", padding=8)
    steps_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
    steps_frame.columnconfigure(0, weight=0)
    steps_frame.columnconfigure(1, weight=0)
    steps_frame.columnconfigure(2, weight=0)
    steps_frame.columnconfigure(3, weight=0)

    do_prep_dataset_var = tk.BooleanVar(value=True)
    do_cache_latents_var = tk.BooleanVar(value=True)
    do_cache_text_var = tk.BooleanVar(value=True)
    do_train_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(steps_frame, text="Prep Dataset", variable=do_prep_dataset_var).grid(
        row=0, column=0, sticky="w", padx=(0, 10)
    )
    ttk.Checkbutton(steps_frame, text="Cache Latents", variable=do_cache_latents_var).grid(
        row=0, column=1, sticky="w", padx=(0, 10)
    )
    ttk.Checkbutton(steps_frame, text="Cache Text Encoder", variable=do_cache_text_var).grid(
        row=0, column=2, sticky="w", padx=(0, 10)
    )
    ttk.Checkbutton(steps_frame, text="Train", variable=do_train_var).grid(row=0, column=3, sticky="w")

    list_container = ttk.LabelFrame(root, text="Datasets (click thumbnail/title to toggle)", padding=8)
    list_container.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
    list_container.columnconfigure(0, weight=1)
    list_container.rowconfigure(0, weight=1)

    canvas = tk.Canvas(list_container, highlightthickness=0, bg=bg_panel)
    scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview, style="Dataset.Vertical.TScrollbar")
    inner = ttk.Frame(canvas, style="TFrame")

    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    log_box = ScrolledText(root, height=10, wrap="word")
    log_box.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
    log_box.configure(bg="#0e1319", fg=fg_text, insertbackground=fg_text, relief="flat", borderwidth=0)

    def log(message: str) -> None:
        log_box.insert("end", message + "\n")
        log_box.see("end")
        root.update_idletasks()

    def first_image_path(dataset_name: str) -> Path | None:
        images_dir = runtime_config.training_dir / dataset_name / "images"
        if not images_dir.exists():
            return None

        for suffix in (".png", ".jpg", ".jpeg"):
            matches = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix])
            if matches:
                return matches[0]
        return None

    def make_thumbnail(image_path: Path | None, ready_to_train: bool, thumb_px: int) -> ImageTk.PhotoImage:
        thumb_size = (thumb_px, thumb_px)
        if image_path is None:
            image = Image.new("RGB", thumb_size, color="#3a3a3a")
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail(thumb_size)
                bg = Image.new("RGB", thumb_size, color="#3a3a3a")
                offset_x = (thumb_size[0] - image.width) // 2
                offset_y = (thumb_size[1] - image.height) // 2
                bg.paste(image, (offset_x, offset_y))
                image = bg
            except Exception:
                image = Image.new("RGB", thumb_size, color="#3a3a3a")

        draw = ImageDraw.Draw(image)
        badge_d = max(18, thumb_px // 5)
        pad = 6
        x1 = thumb_px - badge_d - pad
        y1 = pad
        x2 = x1 + badge_d
        y2 = y1 + badge_d

        if ready_to_train:
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
        else:
            draw.ellipse((x1, y1, x2, y2), fill="#525c69")

        photo = ImageTk.PhotoImage(image)
        thumbnail_refs.append(photo)
        return photo

    def toggle_dataset(name: str) -> None:
        current = vars_by_name[name].get()
        vars_by_name[name].set(not current)

    def on_card_click(name: str) -> None:
        nonlocal active_dataset_name
        active_dataset_name = name
        toggle_dataset(name)
        rebuild_folder_list()

    def set_active_dataset(name: str) -> None:
        nonlocal active_dataset_name
        active_dataset_name = name
        rebuild_folder_list()

    def move_active_dataset(direction: int) -> None:
        nonlocal dataset_order
        if not dataset_order:
            return

        if active_dataset_name is None:
            messagebox.showinfo("No active dataset", "Click a dataset card first, then move it.")
            return

        idx = dataset_order.index(active_dataset_name)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(dataset_order):
            return

        dataset_order[idx], dataset_order[new_idx] = dataset_order[new_idx], dataset_order[idx]
        rebuild_folder_list(force=True)

    def rebuild_folder_list(force: bool = False) -> None:
        nonlocal dataset_order, active_dataset_name
        selected_before = {name for name, var in vars_by_name.items() if var.get()}

        for widget in card_widgets:
            widget.destroy()
        card_widgets.clear()
        thumbnail_refs.clear()
        vars_by_name.clear()

        scanned_names = scan_training_folders(runtime_config.training_dir)
        if not dataset_order:
            dataset_order = list(scanned_names)
        else:
            existing = [name for name in dataset_order if name in scanned_names]
            new_names = [name for name in scanned_names if name not in existing]
            dataset_order = existing + new_names

        names = dataset_order

        if not names:
            empty_label = ttk.Label(inner, text="No folders found.")
            empty_label.grid(row=0, column=0, sticky="w")
            card_widgets.append(empty_label)
            return

        if active_dataset_name not in names:
            active_dataset_name = names[0]

        canvas_width = canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 620

        gap = 8
        min_card = 138
        max_card = 192
        columns = max(2, min(4, canvas_width // (min_card + gap)))
        card_width = max(min_card, min(max_card, (canvas_width - gap * (columns + 1)) // columns))
        thumb_px = max(92, min(160, card_width - 20))
        card_height = thumb_px + 84

        for col in range(columns):
            inner.columnconfigure(col, minsize=card_width + gap, weight=0)

        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=name in selected_before)
            vars_by_name[name] = var

            status = dataset_status(runtime_config.training_dir, name)

            card_style = "ActiveCard.TFrame" if name == active_dataset_name else "Card.TFrame"
            card = ttk.Frame(inner, padding=6, style=card_style, width=card_width, height=card_height)
            grid_row = idx // columns
            grid_col = idx % columns
            card.grid(row=grid_row, column=grid_col, sticky="nw", padx=4, pady=4)
            card.grid_propagate(False)

            image_path = first_image_path(name)
            thumb = make_thumbnail(image_path, status["ready_to_train"], thumb_px)

            image_label = ttk.Label(card, image=thumb, style="CardTitle.TLabel")
            image_label.grid(row=0, column=0, sticky="nsew")
            title_label = ttk.Label(card, text=name, style="CardTitle.TLabel")
            title_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
            status_text = (
                f"S1 {'OK' if status['step1'] else '--'}  "
                f"S2 {'OK' if status['step2'] else '--'}  "
                f"S3 {'OK' if status['step3'] else '--'}"
            )
            status_label = ttk.Label(card, text=status_text, style="CardMeta.TLabel")
            status_label.grid(row=2, column=0, sticky="w", pady=(2, 0))
            check = ttk.Checkbutton(
                card,
                text="Selected",
                variable=var,
                style="Card.TCheckbutton",
                command=lambda n=name: set_active_dataset(n),
            )
            check.grid(row=3, column=0, sticky="w")

            for clickable in (card, image_label, title_label, status_label):
                clickable.bind("<Button-1>", lambda _e, n=name: on_card_click(n))

            card_widgets.append(card)

    def request_relayout() -> None:
        nonlocal resize_after_id
        if resize_after_id is not None:
            root.after_cancel(resize_after_id)
        resize_after_id = root.after(120, rebuild_folder_list)

    def on_mousewheel(event: tk.Event) -> str:
        delta = int(-event.delta / 120)
        if delta == 0:
            delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")
        return "break"

    def on_mousewheel_linux_up(_event: tk.Event) -> str:
        canvas.yview_scroll(-1, "units")
        return "break"

    def on_mousewheel_linux_down(_event: tk.Event) -> str:
        canvas.yview_scroll(1, "units")
        return "break"

    def bind_dataset_wheel(_event: tk.Event) -> None:
        root.bind_all("<MouseWheel>", on_mousewheel)
        root.bind_all("<Button-4>", on_mousewheel_linux_up)
        root.bind_all("<Button-5>", on_mousewheel_linux_down)

    def unbind_dataset_wheel(_event: tk.Event) -> None:
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")

    def selected_names() -> list[str]:
        return [name for name, var in vars_by_name.items() if var.get()]

    def select_all() -> None:
        for var in vars_by_name.values():
            var.set(True)

    def clear_selection() -> None:
        for var in vars_by_name.values():
            var.set(False)

    def run_selected() -> None:
        names = selected_names()
        if not names:
            messagebox.showinfo("Nothing selected", "Select at least one folder.")
            return

        if not (
            do_prep_dataset_var.get()
            or do_cache_latents_var.get()
            or do_cache_text_var.get()
            or do_train_var.get()
        ):
            messagebox.showinfo("No steps selected", "Select at least one step.")
            return

        run_button.state(["disabled"])
        try:
            log("")
            train_models(
                runtime_config,
                names,
                logger=log,
                do_prep_dataset=do_prep_dataset_var.get(),
                do_cache_latents=do_cache_latents_var.get(),
                do_cache_text=do_cache_text_var.get(),
                do_train=do_train_var.get(),
            )
            rebuild_folder_list(force=True)
        finally:
            run_button.state(["!disabled"])

    refresh_button = ttk.Button(controls, text="Refresh", command=lambda: rebuild_folder_list(force=True))
    select_all_button = ttk.Button(controls, text="Select All", command=select_all)
    clear_button = ttk.Button(controls, text="Clear", command=clear_selection)
    move_up_button = ttk.Button(controls, text="Move Up", command=lambda: move_active_dataset(-1))
    move_down_button = ttk.Button(controls, text="Move Down", command=lambda: move_active_dataset(1))
    run_button = ttk.Button(controls, text="Run Selected", command=run_selected)

    refresh_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    select_all_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    clear_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    move_up_button.grid(row=0, column=3, padx=(0, 8), sticky="w")
    move_down_button.grid(row=0, column=4, padx=(0, 8), sticky="w")
    run_button.grid(row=0, column=5, padx=(0, 8), sticky="w")

    canvas.bind("<Configure>", lambda _e: request_relayout())
    list_container.bind("<Enter>", bind_dataset_wheel)
    list_container.bind("<Leave>", unbind_dataset_wheel)
    canvas.bind("<Enter>", bind_dataset_wheel)
    canvas.bind("<Leave>", unbind_dataset_wheel)
    inner.bind("<Enter>", bind_dataset_wheel)
    inner.bind("<Leave>", unbind_dataset_wheel)

    rebuild_folder_list(force=True)
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
