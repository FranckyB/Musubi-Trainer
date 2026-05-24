from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_TRAINING_DIR = Path(r"D:\Musubi-Tuner\training")


def build_dataset_toml_content(model_dir: Path) -> str:
    image_dir = (model_dir / "images").resolve().as_posix()
    cache_dir = (model_dir / "cache").resolve().as_posix()
    return f"""[general]
resolution = [1024, 1024]
caption_extension = \".txt\"
batch_size = 1
num_repeats = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = \"{image_dir}\"
cache_directory = \"{cache_dir}\"
"""


def ensure_dataset_toml(model_dir: Path, name: str) -> Path:
    dataset_toml = model_dir / "dataset.toml"
    if dataset_toml.exists():
        print(f"dataset.toml already exists: {dataset_toml}")
        return dataset_toml

    dataset_toml.write_text(build_dataset_toml_content(model_dir), encoding="utf-8")
    print(f"Created dataset.toml: {dataset_toml}")
    return dataset_toml


def dataset_toml_exists(model_dir: Path) -> bool:
    return (model_dir / "dataset.toml").exists()


def create_missing_caption_files(images_dir: Path, default_caption_keyword: str = "") -> tuple[int, int]:
    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return 0, 0

    created = 0
    scanned = 0
    caption_text = default_caption_keyword.strip()

    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        scanned += 1
        caption_path = path.with_suffix(".txt")
        if caption_path.exists():
            continue

        caption_path.write_text(caption_text, encoding="utf-8")
        created += 1

    return scanned, created


def scan_training_folders(training_dir: Path) -> list[str]:
    if not training_dir.exists():
        return []
    return sorted([path.name for path in training_dir.iterdir() if path.is_dir()])


def process_one_dataset(
    training_dir: Path,
    name: str,
    default_caption_keyword: str = "",
) -> dict[str, int | bool | str]:
    model_dir = training_dir / name
    images_dir = model_dir / "images"
    cache_dir = model_dir / "cache"

    model_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    had_dataset_toml = dataset_toml_exists(model_dir)
    ensure_dataset_toml(model_dir, name)
    scanned, created = create_missing_caption_files(images_dir, default_caption_keyword)

    return {
        "name": name,
        "had_dataset_toml": had_dataset_toml,
        "scanned": scanned,
        "created": created,
    }


def process_datasets(
    training_dir: Path,
    names: list[str],
    logger: Callable[[str], None],
    default_caption_keyword: str = "",
) -> None:
    logger(f"Training directory: {training_dir}")
    if not names:
        logger("No folders selected.")
        return

    total_scanned = 0
    total_created = 0

    for index, name in enumerate(names, start=1):
        logger(f"[{index}/{len(names)}] Processing: {name}")
        result = process_one_dataset(training_dir, name, default_caption_keyword)
        status = "existed" if bool(result["had_dataset_toml"]) else "created"
        logger(f"  dataset.toml: {status}")
        logger(f"  scanned images: {result['scanned']}")
        logger(f"  created captions: {result['created']}")

        total_scanned += int(result["scanned"])
        total_created += int(result["created"])

    logger("")
    logger(f"Done. Total scanned images: {total_scanned}")
    logger(f"Done. Total created captions: {total_created}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan training folders and create missing dataset.toml/caption .txt files."
    )
    parser.add_argument("names", nargs="*", help="Dataset/model name(s) under training/[NAME]")
    parser.add_argument(
        "--training-dir",
        default=str(DEFAULT_TRAINING_DIR),
        help="Training root directory to scan (default: D:\\Musubi-Tuner\\training)",
    )
    parser.add_argument(
        "--default-caption-keyword",
        default="",
        help="Default caption keyword for missing .txt files (blank creates empty .txt files)",
    )
    return parser.parse_args()


def launch_ui(training_dir: Path) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
    except ImportError:
        print("Tkinter is not available in this Python environment.")
        print("Run with names in CLI mode instead, for example:")
        print("  python create_dataset_toml_and_captions.py Name1 Name2")
        return 1

    root = tk.Tk()
    root.title("Dataset TOML + Caption Helper")
    root.geometry("780x580")

    vars_by_name: dict[str, tk.BooleanVar] = {}
    folder_checks: list[ttk.Checkbutton] = []

    root.columnconfigure(0, weight=1)
    root.rowconfigure(2, weight=1)

    header = ttk.Frame(root, padding=10)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    ttk.Label(header, text=f"Training folder: {training_dir}").grid(row=0, column=0, sticky="w")

    controls = ttk.Frame(root, padding=(10, 0, 10, 10))
    controls.grid(row=1, column=0, sticky="ew")

    list_container = ttk.LabelFrame(root, text="Folders found", padding=10)
    list_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
    list_container.columnconfigure(0, weight=1)
    list_container.rowconfigure(0, weight=1)

    canvas = tk.Canvas(list_container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)

    inner.bind(
        "<Configure>",
        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    log_box = ScrolledText(root, height=10, wrap="word")
    log_box.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
    root.rowconfigure(3, weight=1)

    def log(message: str) -> None:
        log_box.insert("end", message + "\n")
        log_box.see("end")
        root.update_idletasks()

    def rebuild_folder_list() -> None:
        for check in folder_checks:
            check.destroy()
        folder_checks.clear()
        vars_by_name.clear()

        names = scan_training_folders(training_dir)
        if not names:
            empty_label = ttk.Label(inner, text="No folders found.")
            empty_label.grid(row=0, column=0, sticky="w")
            folder_checks.append(empty_label)  # type: ignore[arg-type]
            return

        for row, name in enumerate(names):
            var = tk.BooleanVar(value=False)
            vars_by_name[name] = var
            check = ttk.Checkbutton(inner, text=name, variable=var)
            check.grid(row=row, column=0, sticky="w", pady=1)
            folder_checks.append(check)

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

        run_button.state(["disabled"])
        try:
            log("")
            process_datasets(training_dir, names, logger=log)
        finally:
            run_button.state(["!disabled"])

    refresh_button = ttk.Button(controls, text="Refresh", command=rebuild_folder_list)
    select_all_button = ttk.Button(controls, text="Select All", command=select_all)
    clear_button = ttk.Button(controls, text="Clear", command=clear_selection)
    run_button = ttk.Button(controls, text="Run Selected", command=run_selected)

    refresh_button.grid(row=0, column=0, padx=(0, 8), sticky="w")
    select_all_button.grid(row=0, column=1, padx=(0, 8), sticky="w")
    clear_button.grid(row=0, column=2, padx=(0, 8), sticky="w")
    run_button.grid(row=0, column=3, padx=(0, 8), sticky="w")

    rebuild_folder_list()
    root.mainloop()
    return 0


def main() -> int:
    args = parse_args()
    training_dir = Path(args.training_dir)
    default_caption_keyword = args.default_caption_keyword

    if args.names:
        names = [name.strip() for name in args.names if name.strip()]
        process_datasets(training_dir, names, logger=print, default_caption_keyword=default_caption_keyword)
        return 0

    return launch_ui(training_dir)


if __name__ == "__main__":
    raise SystemExit(main())
