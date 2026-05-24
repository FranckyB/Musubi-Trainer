import subprocess
import sys
import argparse
from pathlib import Path
from typing import Callable, Iterable, List

from prep_datasets import process_one_dataset


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
DIT = Path(r"D:\musubi-tuner\models\klein\flux-2-klein-base-9b.safetensors")
VAE = Path(r"D:\musubi-tuner\models\klein\ae.safetensors")
TEXT_ENCODER = Path(r"D:\musubi-tuner\models\klein\text_encoder\model-00001-of-00004.safetensors")
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"


def parse_model_names(raw: str) -> List[str]:
    """Split comma-separated names and trim surrounding whitespace."""
    names = [part.strip() for part in raw.split(",")]
    return [name for name in names if name]


def scan_training_folders(training_dir: Path) -> list[str]:
    if not training_dir.exists():
        return []
    return sorted([path.name for path in training_dir.iterdir() if path.is_dir()])


def dataset_image_files(dataset_name: str) -> list[Path]:
    images_dir = TRAINING_DIR / dataset_name / "images"
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])


def is_step1_ready(dataset_name: str) -> bool:
    dataset_dir = TRAINING_DIR / dataset_name
    dataset_toml_exists = (dataset_dir / "dataset.toml").exists()
    image_files = dataset_image_files(dataset_name)

    if not dataset_toml_exists or not image_files:
        return False

    return all(image_path.with_suffix(".txt").exists() for image_path in image_files)


def is_step2_ready(dataset_name: str) -> bool:
    image_files = dataset_image_files(dataset_name)
    cache_dir = TRAINING_DIR / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        pattern = f"{image_path.stem}_*_{LATENT_SUFFIX}.safetensors"
        if not any(cache_dir.glob(pattern)):
            return False

    return True


def is_step3_ready(dataset_name: str) -> bool:
    image_files = dataset_image_files(dataset_name)
    cache_dir = TRAINING_DIR / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        expected = cache_dir / f"{image_path.stem}_{LATENT_SUFFIX}_te.safetensors"
        if not expected.exists():
            return False

    return True


def dataset_status(dataset_name: str) -> dict[str, bool]:
    step1 = is_step1_ready(dataset_name)
    step2 = is_step2_ready(dataset_name)
    step3 = is_step3_ready(dataset_name)
    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "ready_to_train": step1 and step2 and step3,
    }


def run_command(args: Iterable[str], cwd: Path) -> None:
    result = subprocess.run(list(args), cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(args)}")


def run_steps_for_model(
    model_name: str,
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
    logger: Callable[[str], None],
) -> None:
    dataset_config = TRAINING_DIR / model_name / "dataset.toml"
    output_dir = TRAINING_DIR / model_name / "output"
    output_name = f"{model_name}_Klein"

    logger("=" * 58)
    logger(f"MODEL_NAME: {model_name}")
    logger(f"dataset_config: {dataset_config}")
    logger(f"output_dir:     {output_dir}")

    if do_prep_dataset:
        prep_result = process_one_dataset(TRAINING_DIR, model_name)
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
                str(VAE),
                "--batch_size",
                "16",
                "--model_version",
                MODEL_VERSION,
            ],
            cwd=MUSUBI_DIR,
        )

    if do_cache_text:
        run_command(
            [
                sys.executable,
                "flux_2_cache_text_encoder_outputs.py",
                "--dataset_config",
                str(dataset_config),
                "--text_encoder",
                str(TEXT_ENCODER),
                "--batch_size",
                "16",
                "--model_version",
                MODEL_VERSION,
            ],
            cwd=MUSUBI_DIR,
        )

    if do_train:
        run_command(
            [
                sys.executable,
                "flux_2_train_network.py",
                "--dit", str(DIT),
                "--vae", str(VAE),
                "--vae_dtype", "bf16",
                "--text_encoder", str(TEXT_ENCODER),
                "--model_version", MODEL_VERSION,
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
            cwd=MUSUBI_DIR,
        )


def train_models(
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
            status = dataset_status(model_name)
            if status["ready_to_train"]:
                effective_do_prep_dataset = False
                effective_do_cache_latents = False
                effective_do_cache_text = False
                effective_do_train = True
                logger("  dataset is already green (S1/S2/S3 ready): skipping prep/cache and running train only")

        try:
            run_steps_for_model(
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
        from tkinter import messagebox
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

    root = tk.Tk()
    root.title("Klein Training Launcher")
    root.geometry("660x840")
    root.minsize(620, 800)
    root.configure(bg=bg_root)

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

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=0)
    root.rowconfigure(2, weight=0)
    root.rowconfigure(3, weight=5, minsize=360)
    root.rowconfigure(4, weight=2)

    header = ttk.Frame(root, padding=8)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    ttk.Label(header, text=f"Training folder: {TRAINING_DIR}").grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.grid(row=1, column=0, sticky="ew")

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
        images_dir = TRAINING_DIR / dataset_name / "images"
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

        scanned_names = scan_training_folders(TRAINING_DIR)
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

            status = dataset_status(name)

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
        any_step_flag = args.prep_dataset or args.cache_latents or args.cache_text or args.train
        do_prep_dataset = args.prep_dataset if any_step_flag else True
        do_cache_latents = args.cache_latents if any_step_flag else True
        do_cache_text = args.cache_text if any_step_flag else True
        do_train = args.train if any_step_flag else True

        model_names = [name.strip() for name in args.names if name.strip()]
        return train_models(
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
