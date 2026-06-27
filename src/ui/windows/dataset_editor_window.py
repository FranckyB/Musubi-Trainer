from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
import threading
import traceback
import warnings
from typing import Any

from ...launcher_shared import dataset_audio_files


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
}

WHISPER_LANGUAGE_CODES = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "German": "de",
    "French": "fr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Spanish": "es",
    "Italian": "it",
}


class DatasetEditorWindow:
    def __init__(self, **dependencies: object) -> None:
        for name, value in dependencies.items():
            setattr(self, name, value)

    def open(self, dataset_name: str) -> None:
        columns = 4
        tile_size_px = 300
        tile_gap_px = 4
        grid_width_px = (tile_size_px * columns) + (tile_gap_px * (columns + 1))
        # Include dialog frame padding and the vertical scrollbar lane.
        dialog_width_px = grid_width_px + 80
        tile_side_pad_px = tile_gap_px // 2

        dataset_dir = self.dataset_dir_path(dataset_name)
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            self.messagebox.showerror("Edit dataset", f"Dataset folder not found:\n{dataset_dir}", parent=self.root)
            return

        image_paths = self.dataset_image_files(self.datasets_root_dir(), dataset_name)
        audio_paths = dataset_audio_files(self.datasets_root_dir(), dataset_name)
        video_paths = sorted(
            [
                path
                for path in dataset_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
        )

        if (not image_paths) and (audio_paths or video_paths):
            self._open_media_dataset_dialog(
                dataset_name,
                audio_paths,
                video_paths,
                columns,
                tile_size_px,
                tile_gap_px,
                dialog_width_px,
                tile_side_pad_px,
            )
            return

        if not image_paths:
            self.messagebox.showinfo("Edit dataset", "No images found in this dataset.", parent=self.root)
            return

        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Edit Dataset: {dataset_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.bg_panel)
        dialog.resizable(False, True)
        self.set_dark_title_bar(dialog)
        dialog.minsize(dialog_width_px, 760)
        dialog.geometry(f"{dialog_width_px}x920")

        outer = self.ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        self.ttk.Label(
            outer,
            text=f"{dataset_name} ({len(image_paths)} images)",
            style="TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        controls = self.ttk.Frame(outer)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(9, weight=1)
        autotag_all_var = self.tk.BooleanVar(master=dialog, value=False)
        autotag_status_var = self.tk.StringVar(master=dialog, value="Captions auto-save as .txt sidecar files.")
        trigger_word_var = self.tk.StringVar(master=dialog, value="")
        replace_find_var = self.tk.StringVar(master=dialog, value="")
        replace_with_var = self.tk.StringVar(master=dialog, value="")
        caption_mode_choices = ["simple", "detailed", "extra", "mixed", "extra_mixed", "analyze"]
        caption_mode_var = self.tk.StringVar(master=dialog, value="simple")
        autotag_button: Any = None
        caption_button: Any = None
        detailed_button: Any = None
        replace_apply_button: Any = None

        grid_host = self.tk.Frame(
            outer,
            bg=self.bg_panel,
            highlightthickness=1,
            highlightbackground="#4a6ea3",
            bd=0,
        )
        grid_host.grid(row=2, column=0, sticky="nsew")
        grid_host.columnconfigure(0, weight=1)
        grid_host.rowconfigure(0, weight=1)

        editor_canvas = self.tk.Canvas(grid_host, highlightthickness=0, bg=self.bg_panel)
        editor_scroll = self.ttk.Scrollbar(
            grid_host,
            orient="vertical",
            command=editor_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        editor_inner = self.ttk.Frame(editor_canvas)
        editor_inner_id = editor_canvas.create_window((0, 0), window=editor_inner, anchor="nw")
        editor_canvas.configure(yscrollcommand=editor_scroll.set)

        editor_canvas.grid(row=0, column=0, sticky="nsew")
        editor_scroll.grid(row=0, column=1, sticky="ns")

        thumb_refs: list[Any] = []
        caption_path_by_widget: dict[Any, Path] = {}
        pending_save_by_widget: dict[Any, str] = {}
        caption_widget_by_path: dict[Path, Any] = {}
        original_transformers_get_imports: Any = None

        def _hf_token_from_settings() -> str | None:
            settings = getattr(self, "settings_state", {})
            app_settings = getattr(self, "app_settings", None)
            if not isinstance(settings, dict) or app_settings is None:
                return None
            token_key = getattr(app_settings, "HF_TOKEN_KEY", "hf_token")
            raw = settings.get(token_key, "")
            token = str(raw).strip()
            return token or None

        def _model_download_location_from_settings() -> str:
            settings = getattr(self, "settings_state", {})
            app_settings = getattr(self, "app_settings", None)
            default_location = getattr(self, "DOWNLOAD_LOCATION_MODELS_FOLDER", "Models Folder")
            if not isinstance(settings, dict) or app_settings is None:
                return default_location
            key = getattr(app_settings, "MODEL_DOWNLOAD_LOCATION_KEY", "model_download_location")
            location = str(settings.get(key, default_location) or default_location)
            return location

        def _fixed_get_imports(filename: str | Path) -> list[str]:
            if original_transformers_get_imports is None:
                raise RuntimeError("Florence import patch is not initialized correctly.")

            if not str(filename).endswith("modeling_florence2.py"):
                return original_transformers_get_imports(filename)
            imports = original_transformers_get_imports(filename)
            try:
                imports.remove("flash_attn")
            except Exception:
                pass
            return imports

        def _resolve_florence_model_id() -> str:
            return "MiaoshouAI/Florence-2-base-PromptGen-v2.0"

        def _ensure_florence_model_path(model_id: str, token: str | None) -> str:
            from huggingface_hub import snapshot_download

            location = _model_download_location_from_settings()
            hf_location = getattr(self, "DOWNLOAD_LOCATION_HUGGINGFACE", "HuggingFace Cache")
            if location == hf_location:
                return model_id

            ws_root_fn = getattr(self, "download_workspace_root", None)
            if callable(ws_root_fn):
                ws_root = ws_root_fn()
            else:
                ws_root = Path(__file__).resolve().parents[3]
            model_name = model_id.rsplit("/", 1)[-1]
            model_path = ws_root / "Models" / model_name
            if model_path.exists() and model_path.is_dir():
                return str(model_path)

            model_path.mkdir(parents=True, exist_ok=True)
            snapshot_kwargs: dict[str, Any] = {
                "repo_id": model_id,
                "local_dir": str(model_path),
                "local_dir_use_symlinks": False,
            }
            if token:
                snapshot_kwargs["token"] = token
            snapshot_download(**snapshot_kwargs)
            return str(model_path)

        def _caption_prompt_token(mode: str) -> str:
            token_by_mode = {
                "tags": "<GENERATE_TAGS>",
                "simple": "<CAPTION>",
                "detailed": "<DETAILED_CAPTION>",
                "extra": "<MORE_DETAILED_CAPTION>",
                "mixed": "<MIX_CAPTION>",
                "extra_mixed": "<MIX_CAPTION_PLUS>",
                "analyze": "<ANALYZE>",
            }
            return token_by_mode.get(mode, "<GENERATE_TAGS>")

        def _set_autotag_busy(is_busy: bool, status_text: str = "") -> None:
            button_state = "disabled" if is_busy else "normal"
            for action_button in (caption_button, autotag_button, detailed_button, replace_apply_button):
                if action_button is not None:
                    try:
                        action_button.configure(state=button_state)
                    except Exception:
                        pass
            autotag_status_var.set(status_text or "Captions auto-save as .txt sidecar files.")

        def _apply_trigger_word(text: str) -> str:
            trigger = trigger_word_var.get().strip().strip(",")
            if not trigger:
                return text.strip()
            content = text.strip()
            if content:
                return f"{trigger}, {content}"
            return f"{trigger},"

        def _eligible_for_autotag(caption_path: Path, include_non_empty: bool) -> bool:
            if include_non_empty:
                return True
            if not caption_path.exists() or not caption_path.is_file():
                return True
            try:
                return not caption_path.read_text(encoding="utf-8").strip()
            except OSError:
                return True

        def _run_autotag_for_images(include_non_empty: bool, mode: str, specific_image_path: Path | None = None) -> None:
            def _terminal_log(message: str) -> None:
                print(f"[Dataset Editor] {message}", flush=True)

            target_paths: list[Path] = []
            skipped_existing_count = 0
            candidate_paths = [specific_image_path] if specific_image_path is not None else image_paths
            for image_path in candidate_paths:
                if image_path is None:
                    continue
                caption_path = image_path.with_suffix(".txt")
                is_eligible = specific_image_path is not None or _eligible_for_autotag(caption_path, include_non_empty)
                if is_eligible:
                    target_paths.append(image_path)
                else:
                    skipped_existing_count += 1

            _terminal_log(
                f"Autotag request for '{dataset_name}' (mode={mode}, replace_all={include_non_empty}, candidates={len(candidate_paths)}, eligible={len(target_paths)}, skipped_existing={skipped_existing_count})"
            )

            if not target_paths:
                autotag_status_var.set("Nothing to autotag.")
                if specific_image_path is None:
                    self.log(f"[Dataset Editor] Autotag skipped for '{dataset_name}' (mode={mode}): nothing eligible.")
                else:
                    self.log(f"[Dataset Editor] Autotag skipped for '{dataset_name}' image '{specific_image_path.name}' (mode={mode}): nothing eligible.")
                _terminal_log(
                    f"Autotag skipped for '{dataset_name}' (mode={mode}): nothing eligible."
                )
                return

            _set_autotag_busy(True, "Preparing Florence model...")
            if specific_image_path is None:
                self.log(f"[Dataset Editor] Autotag start for '{dataset_name}' (mode={mode}, all={include_non_empty})")
                _terminal_log(
                    f"Autotag start for '{dataset_name}' (mode={mode}, replace_all={include_non_empty}, targets={len(target_paths)})"
                )
            else:
                self.log(f"[Dataset Editor] Autotag start for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})")
                _terminal_log(
                    f"Autotag start for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})"
                )

            def worker() -> None:
                nonlocal original_transformers_get_imports
                model_obj: Any = None
                try:
                    # Keep TensorFlow/absl noise from polluting the console during HF model load.
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
                    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*timm\.models\.layers.*")
                    warnings.filterwarnings("ignore", message=r".*GenerationMixin.*")

                    import torch
                    from PIL import Image as PILImage
                    from transformers import AutoModelForCausalLM, AutoProcessor
                    from transformers import GenerationConfig
                    from transformers.generation.utils import GenerationMixin
                    from transformers.dynamic_module_utils import get_imports as transformers_get_imports
                    from unittest.mock import patch

                    original_transformers_get_imports = transformers_get_imports

                    model_id = _resolve_florence_model_id()
                    token = _hf_token_from_settings()
                    model_ref = _ensure_florence_model_path(model_id, token)
                    _terminal_log(f"Loading Florence model '{model_id}' from '{model_ref}'")

                    if torch.cuda.is_available():
                        device = "cuda"
                        dtype = torch.float16
                    else:
                        device = "cpu"
                        dtype = torch.float32
                    _terminal_log(f"Florence device={device}, dtype={dtype}")

                    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
                        load_kwargs: dict[str, Any] = {
                            "trust_remote_code": True,
                            "dtype": dtype,
                        }
                        # Some Florence remote-code variants trip over sdpa capability checks.
                        # Prefer eager attention first, then fall back if the arg is unsupported.
                        try:
                            model_obj = AutoModelForCausalLM.from_pretrained(
                                model_ref,
                                attn_implementation="eager",
                                **load_kwargs,
                            )
                        except TypeError:
                            model_obj = AutoModelForCausalLM.from_pretrained(
                                model_ref,
                                **load_kwargs,
                            )

                    if not hasattr(model_obj, "_supports_sdpa"):
                        try:
                            setattr(type(model_obj), "_supports_sdpa", False)
                        except Exception:
                            pass
                        try:
                            setattr(model_obj, "_supports_sdpa", False)
                        except Exception:
                            pass

                    language_model = getattr(model_obj, "language_model", None)
                    if language_model is not None and not hasattr(language_model, "generate"):
                        lm_cls = language_model.__class__
                        patched_cls_name = f"{lm_cls.__name__}WithGenerationMixin"
                        patched_cls = type(patched_cls_name, (lm_cls, GenerationMixin), {})
                        language_model.__class__ = patched_cls
                        self.log("[Dataset Editor] Applied Florence GenerationMixin compatibility patch.")

                    if language_model is not None:
                        generation_config = getattr(language_model, "generation_config", None)
                        if generation_config is None:
                            try:
                                language_model.generation_config = GenerationConfig.from_model_config(language_model.config)
                                self.log("[Dataset Editor] Initialized Florence language model generation_config.")
                            except Exception:
                                pass

                    model_obj = model_obj.to(device)
                    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
                    prompt_token = _caption_prompt_token(mode)

                    def _generate_prompt_output(image: Any, prompt: str) -> str:
                        inputs = processor(
                            text=prompt,
                            images=image,
                            return_tensors="pt",
                            do_rescale=False,
                        ).to(dtype).to(device)

                        try:
                            generated_ids = model_obj.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                early_stopping=False,
                                do_sample=False,
                                num_beams=1,
                                use_cache=False,
                            )
                        except AttributeError as generate_exc:
                            if "_supports_sdpa" not in str(generate_exc):
                                raise
                            try:
                                setattr(type(model_obj), "_supports_sdpa", False)
                            except Exception:
                                pass
                            try:
                                setattr(model_obj, "_supports_sdpa", False)
                            except Exception:
                                pass
                            generated_ids = model_obj.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                early_stopping=False,
                                do_sample=False,
                                num_beams=1,
                                use_cache=False,
                            )

                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                        parsed = processor.post_process_generation(
                            generated_text,
                            task=prompt,
                            image_size=(image.width, image.height),
                        )
                        return str(parsed.get(prompt, "")).strip()

                    completed = 0
                    total = len(target_paths)
                    for image_path in target_paths:
                        caption_path = image_path.with_suffix(".txt")
                        with PILImage.open(image_path) as opened_image:
                            image = opened_image.convert("RGB")
                            if mode == "tag_plus_caption_selected":
                                tags = _generate_prompt_output(image, "<GENERATE_TAGS>")
                                caption_mode = caption_mode_var.get().strip() or "simple"
                                caption = _generate_prompt_output(image, _caption_prompt_token(caption_mode))
                                if tags and caption:
                                    tags = f"{tags}, {caption}"
                                else:
                                    tags = tags or caption
                            elif mode == "caption_selected":
                                caption_mode = caption_mode_var.get().strip() or "simple"
                                tags = _generate_prompt_output(image, _caption_prompt_token(caption_mode))
                            else:
                                tags = _generate_prompt_output(image, prompt_token)

                            tags = _apply_trigger_word(tags)

                        preview = tags.replace("\n", " ").strip()
                        if len(preview) > 320:
                            preview = preview[:317] + "..."
                        print(
                            f"[Dataset Editor] [{completed + 1}/{total}] {caption_path.name} ({mode}): {preview}",
                            flush=True,
                        )

                        caption_path.write_text(tags, encoding="utf-8")

                        widget = caption_widget_by_path.get(caption_path)
                        if widget is not None:
                            def _update_widget(target_widget: Any, target_text: str) -> None:
                                target_widget.delete("1.0", "end")
                                target_widget.insert("1.0", target_text)
                                target_widget.edit_modified(False)

                            dialog.after(0, lambda w=widget, t=tags: _update_widget(w, t))

                        completed += 1
                        dialog.after(0, lambda c=completed, t=total: autotag_status_var.set(f"Autotagging... {c}/{t}"))

                    if specific_image_path is None:
                        self.log(f"[Dataset Editor] Autotag complete for '{dataset_name}' (mode={mode}): {completed} image(s)")
                        _terminal_log(
                            f"Autotag complete for '{dataset_name}' (mode={mode}): {completed} image(s)"
                        )
                        dialog.after(0, lambda: _set_autotag_busy(False, f"Autotagged {completed} image(s) [{mode}]."))
                    else:
                        self.log(f"[Dataset Editor] Autotag complete for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})")
                        _terminal_log(
                            f"Autotag complete for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})"
                        )
                        dialog.after(0, lambda: _set_autotag_busy(False, f"Updated '{specific_image_path.name}' [{mode}]."))
                except Exception as exc:
                    error_text = str(exc)
                    stack_text = traceback.format_exc()
                    self.log(f"[Dataset Editor] Autotag failed for '{dataset_name}': {error_text}")
                    _terminal_log(f"Autotag failed for '{dataset_name}': {error_text}")
                    for line in stack_text.strip().splitlines():
                        self.log(f"[Dataset Editor] {line}")
                    print(f"[Dataset Editor] Autotag failed for '{dataset_name}': {error_text}")
                    print(stack_text)
                    dialog.after(0, lambda: _set_autotag_busy(False, "Autotag failed."))
                    dialog.after(0, lambda e=error_text: self.messagebox.showerror("Autotag failed", e, parent=dialog))
                finally:
                    try:
                        if model_obj is not None:
                            del model_obj
                    except Exception:
                        pass

            threading.Thread(target=worker, name="dataset-editor-autotag", daemon=True).start()

        self.ttk.Label(controls, text="Trigger word:", style="TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ttk.Entry(
            controls,
            textvariable=trigger_word_var,
            width=16,
            style="Flat.TEntry",
        ).grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.ttk.Label(controls, text="Caption mode:", style="TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.ttk.Combobox(
            controls,
            textvariable=caption_mode_var,
            values=caption_mode_choices,
            state="readonly",
            width=12,
        ).grid(row=0, column=3, sticky="w", padx=(0, 10))

        caption_button = self.ttk.Button(
            controls,
            text="Auto Caption",
            command=lambda: _run_autotag_for_images(bool(autotag_all_var.get()), "caption_selected"),
        )
        caption_button.grid(row=0, column=4, sticky="w")
        autotag_button = self.ttk.Button(
            controls,
            text="Auto Tag (SDXL)",
            command=lambda: _run_autotag_for_images(bool(autotag_all_var.get()), "tags"),
        )
        autotag_button.grid(row=0, column=5, sticky="w", padx=(8, 0))
        detailed_button = self.ttk.Button(
            controls,
            text="Auto Tag+Caption",
            command=lambda: _run_autotag_for_images(bool(autotag_all_var.get()), "tag_plus_caption_selected"),
        )
        detailed_button.grid(row=0, column=6, sticky="w", padx=(8, 0))
        self.ttk.Checkbutton(
            controls,
            text="Replace All",
            variable=autotag_all_var,
        ).grid(row=0, column=7, sticky="w", padx=(12, 0))

        replace_controls = self.ttk.Frame(controls)
        replace_controls.grid(row=0, column=10, sticky="e", padx=(24, 0))
        self.ttk.Label(replace_controls, text="Replace:", style="TLabel").grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.ttk.Entry(
            replace_controls,
            textvariable=replace_find_var,
            width=12,
            style="Flat.TEntry",
        ).grid(row=0, column=1, sticky="w", padx=(0, 8))
        self.ttk.Label(replace_controls, text="With:", style="TLabel").grid(row=0, column=2, sticky="e", padx=(0, 4))
        self.ttk.Entry(
            replace_controls,
            textvariable=replace_with_var,
            width=12,
            style="Flat.TEntry",
        ).grid(row=0, column=3, sticky="w", padx=(0, 8))

        def _show_image_autotag_menu(event: Any, image_path: Path) -> str:
            def _delete_image_item(target_image_path: Path) -> None:
                confirmed = self.messagebox.askyesno(
                    "Delete image",
                    f"Delete this image and its caption?\n\n{target_image_path.name}",
                    parent=dialog,
                )
                if not confirmed:
                    return

                caption_path = target_image_path.with_suffix(".txt")
                widget = caption_widget_by_path.get(caption_path)
                if widget is not None:
                    flush_caption_save(widget)

                try:
                    if target_image_path.exists():
                        target_image_path.unlink()
                    if caption_path.exists():
                        caption_path.unlink()
                except OSError as exc:
                    self.messagebox.showerror("Delete image", f"Could not delete image:\n{exc}", parent=dialog)
                    return

                dialog.destroy()
                self.root.after(0, lambda: self.open(dataset_name))

            menu = self.tk.Menu(dialog, tearoff=0)
            menu.add_command(label="Auto Caption", command=lambda p=image_path: _run_autotag_for_images(True, "caption_selected", p))
            menu.add_command(label="Auto Tag (SDXL)", command=lambda p=image_path: _run_autotag_for_images(True, "tags", p))
            menu.add_command(label="Auto Tag+Caption", command=lambda p=image_path: _run_autotag_for_images(True, "tag_plus_caption_selected", p))
            menu.add_separator()
            menu.add_command(label="Delete", command=lambda p=image_path: _delete_image_item(p))
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
            return "break"

        def build_caption_thumb(image_path: Path) -> Any:
            thumb_size = (tile_size_px, tile_size_px)
            try:
                image = self.Image.open(image_path).convert("RGB")
                src_w, src_h = image.size
                dst_w, dst_h = thumb_size
                # Fit the full image inside the tile without cropping.
                scale = min(dst_w / src_w, dst_h / src_h)
                resized_w = max(1, int(round(src_w * scale)))
                resized_h = max(1, int(round(src_h * scale)))
                resized = image.resize((resized_w, resized_h), self.Image.Resampling.LANCZOS)
                image = self.Image.new("RGB", thumb_size, color="#000000")
                paste_x = (dst_w - resized_w) // 2
                paste_y = (dst_h - resized_h) // 2
                image.paste(resized, (paste_x, paste_y))
            except Exception:
                image = self.Image.new("RGB", thumb_size, color="#000000")

            image_rgba = image.convert("RGBA")
            border_overlay = self.Image.new("RGBA", thumb_size, (0, 0, 0, 0))
            border_draw = self.ImageDraw.Draw(border_overlay)
            border_draw.rectangle((0, 0, thumb_size[0] - 1, thumb_size[1] - 1), outline=(255, 255, 255, 96), width=1)
            composited = self.Image.alpha_composite(image_rgba, border_overlay).convert("RGB")
            return self.ImageTk.PhotoImage(composited, master=self.root)

        def flush_caption_save(widget: Any) -> None:
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
                self.log(f"[Dataset Editor] Failed to save caption '{caption_path.name}': {exc}")

        def schedule_caption_save(event: Any) -> None:
            widget = event.widget
            if not isinstance(widget, self.tk.Text):
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

        def on_caption_focus_out(event: Any) -> None:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                flush_caption_save(widget)

        def _apply_replace_to_all_captions() -> None:
            find_text = replace_find_var.get()
            replace_text = replace_with_var.get()
            if find_text == "":
                self.messagebox.showwarning(
                    "Replace captions",
                    "Enter text in Replace before applying.",
                    parent=dialog,
                )
                return

            # Commit any pending in-editor edits first so file contents are up to date.
            for text_widget in list(caption_path_by_widget.keys()):
                flush_caption_save(text_widget)

            changed_count = 0
            checked_count = 0
            errors: list[str] = []

            for image_path in image_paths:
                caption_path = image_path.with_suffix(".txt")
                if not caption_path.exists() or not caption_path.is_file():
                    continue

                checked_count += 1
                try:
                    original_text = caption_path.read_text(encoding="utf-8")
                except OSError as exc:
                    errors.append(f"{caption_path.name}: {exc}")
                    continue

                updated_text = original_text.replace(find_text, replace_text)
                if updated_text == original_text:
                    continue

                try:
                    caption_path.write_text(updated_text, encoding="utf-8")
                except OSError as exc:
                    errors.append(f"{caption_path.name}: {exc}")
                    continue

                changed_count += 1
                widget = caption_widget_by_path.get(caption_path)
                if widget is not None:
                    widget.delete("1.0", "end")
                    widget.insert("1.0", updated_text)
                    widget.edit_modified(False)

            autotag_status_var.set(
                f"Replace applied: {changed_count} changed / {checked_count} checked"
            )
            self.log(
                f"[Dataset Editor] Replace captions for '{dataset_name}': "
                f"find='{find_text}' with='{replace_text}', changed={changed_count}, checked={checked_count}, errors={len(errors)}"
            )

            if errors:
                self.messagebox.showerror(
                    "Replace captions",
                    "Some captions could not be updated:\n" + "\n".join(errors[:10]),
                    parent=dialog,
                )

        replace_apply_button = self.ttk.Button(
            replace_controls,
            text="Apply",
            command=_apply_replace_to_all_captions,
        )
        replace_apply_button.grid(row=0, column=4, sticky="w")

        def on_editor_inner_configure(_event: Any) -> None:
            editor_canvas.configure(scrollregion=editor_canvas.bbox("all"))

        def on_editor_canvas_configure(event: Any) -> None:
            editor_canvas.itemconfigure(editor_inner_id, width=event.width)

        def editor_canvas_has_overflow() -> bool:
            bbox = editor_canvas.bbox("all")
            if not bbox:
                return False
            content_height = max(0, int(bbox[3] - bbox[1]))
            viewport_height = max(0, int(editor_canvas.winfo_height()))
            return content_height > (viewport_height + 2)

        def on_editor_mousewheel(event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            delta = int(-event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            editor_canvas.yview_scroll(delta, "units")
            return "break"

        def on_editor_linux_up(_event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            editor_canvas.yview_scroll(-1, "units")
            return "break"

        def on_editor_linux_down(_event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            editor_canvas.yview_scroll(1, "units")
            return "break"

        def on_caption_mousewheel(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                delta = int(-event.delta / 120)
                if delta == 0:
                    delta = -1 if event.delta > 0 else 1
                widget.yview_scroll(delta, "units")
                return "break"
            return on_editor_mousewheel(event)

        def on_caption_linux_up(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                widget.yview_scroll(-1, "units")
                return "break"
            return on_editor_linux_up(event)

        def on_caption_linux_down(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                widget.yview_scroll(1, "units")
                return "break"
            return on_editor_linux_down(event)

        def attach_autohide_scrollbar(text_widget: Any, scrollbar_widget: Any) -> None:
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

            item_frame = self.ttk.Frame(editor_inner, padding=(4, 4, 4, 6), style="TFrame")
            item_frame.grid(row=idx // columns, column=idx % columns, sticky="n", padx=tile_side_pad_px, pady=4)
            item_frame.columnconfigure(0, weight=1)

            photo = build_caption_thumb(image_path)
            thumb_refs.append(photo)
            image_label = self.ttk.Label(item_frame, image=photo, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")
            image_label.bind("<MouseWheel>", on_editor_mousewheel)
            image_label.bind("<Button-4>", on_editor_linux_up)
            image_label.bind("<Button-5>", on_editor_linux_down)
            image_label.bind("<Button-3>", lambda event, p=image_path: _show_image_autotag_menu(event, p))

            name_label = self.ttk.Label(
                item_frame,
                text=image_path.name,
                style="CardMeta.TLabel",
                anchor="center",
                justify="center",
                wraplength=tile_size_px,
            )
            name_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            name_label.bind("<Button-3>", lambda event, p=image_path: _show_image_autotag_menu(event, p))

            caption_shell = self.tk.Frame(
                item_frame,
                width=tile_size_px,
                height=64,
                bg="#111826",
                highlightthickness=1,
                highlightbackground="#2a3a50",
                highlightcolor="#4a6ea3",
                bd=0,
            )
            caption_shell.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            caption_shell.grid_propagate(False)

            caption_widget = self.tk.Text(
                caption_shell,
                width=1,
                height=3,
                wrap="word",
                bg="#111826",
                fg=self.fg_text,
                insertbackground=self.fg_text,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                padx=6,
                pady=5,
            )
            caption_widget.pack(side="left", fill="both", expand=True)

            caption_scroll = self.ttk.Scrollbar(
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
            caption_widget.bind("<MouseWheel>", on_caption_mousewheel)
            caption_widget.bind("<Button-4>", on_caption_linux_up)
            caption_widget.bind("<Button-5>", on_caption_linux_down)
            caption_path_by_widget[caption_widget] = caption_path
            caption_widget_by_path[caption_path] = caption_widget

        def close_editor() -> None:
            for text_widget in list(caption_path_by_widget.keys()):
                flush_caption_save(text_widget)
            dialog.destroy()

        footer = self.tk.Frame(outer, bg=self.bg_panel, bd=0)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        footer_text_color = getattr(self, "fg_muted", self.fg_text)
        self.tk.Label(
            footer,
            textvariable=autotag_status_var,
            bg=self.bg_panel,
            fg=footer_text_color,
            anchor="w",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Button(footer, text="Close", command=close_editor).grid(row=0, column=1, sticky="e")

        editor_inner.bind("<Configure>", on_editor_inner_configure)
        editor_canvas.bind("<MouseWheel>", on_editor_mousewheel)
        editor_canvas.bind("<Button-4>", on_editor_linux_up)
        editor_canvas.bind("<Button-5>", on_editor_linux_down)
        editor_inner.bind("<MouseWheel>", on_editor_mousewheel)
        editor_inner.bind("<Button-4>", on_editor_linux_up)
        editor_inner.bind("<Button-5>", on_editor_linux_down)
        dialog.protocol("WM_DELETE_WINDOW", close_editor)

        self.center_window(dialog)
        dialog.deiconify()
        self.root.wait_window(dialog)

    def _open_media_dataset_dialog(
        self,
        dataset_name: str,
        audio_paths: list[Path],
        video_paths: list[Path],
        columns: int,
        tile_size_px: int,
        tile_gap_px: int,
        dialog_width_px: int,
        tile_side_pad_px: int,
    ) -> None:
        media_items: list[tuple[Path, str]] = [(path, "audio") for path in audio_paths] + [
            (path, "video") for path in video_paths
        ]
        media_items.sort(key=lambda pair: pair[0].name.casefold())

        if not media_items:
            self.messagebox.showinfo("Edit dataset", "No playable media found in this dataset.", parent=self.root)
            return

        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Edit Dataset: {dataset_name}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.bg_panel)
        dialog.resizable(False, True)
        self.set_dark_title_bar(dialog)
        dialog.minsize(dialog_width_px, 760)
        dialog.geometry(f"{dialog_width_px}x920")

        outer = self.ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        audio_count = len(audio_paths)
        video_count = len(video_paths)
        header_parts: list[str] = []
        if audio_count:
            header_parts.append(f"{audio_count} audio clip{'s' if audio_count != 1 else ''}")
        if video_count:
            header_parts.append(f"{video_count} video clip{'s' if video_count != 1 else ''}")
        header_text = ", ".join(header_parts) if header_parts else "0 media"

        self.ttk.Label(
            outer,
            text=f"{dataset_name} ({header_text})",
            style="TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        controls = self.ttk.Frame(outer)
        controls.grid(row=1, column=0, sticky="w", pady=(0, 8))
        transcribe_model_var = self.tk.StringVar(master=dialog, value="Medium")
        transcribe_language_var = self.tk.StringVar(master=dialog, value="Auto-detect")
        replace_existing_var = self.tk.BooleanVar(master=dialog, value=False)
        transcribe_status_var = self.tk.StringVar(master=dialog, value="")
        transcribe_all_button: Any = None
        extract_audio_button: Any = None
        normalize_audio_button: Any = None
        split_defaults: dict[str, float] = {
            "split_min": 4.0,
            "split_max": 8.0,
            "silence_trim": 1.0,
            "discard_under": 1.0,
        }

        self.ttk.Label(controls, text="Whisper:", style="TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.ttk.Combobox(
            controls,
            textvariable=transcribe_model_var,
            values=["Medium", "Large"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=(0, 8))

        self.ttk.Label(controls, text="Language:", style="TLabel").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.ttk.Combobox(
            controls,
            textvariable=transcribe_language_var,
            values=["Auto-detect", *WHISPER_LANGUAGE_CODES.keys()],
            state="readonly",
            width=14,
        ).grid(row=0, column=3, sticky="w", padx=(0, 8))

        self.ttk.Checkbutton(
            controls,
            text="Replace Existing",
            variable=replace_existing_var,
        ).grid(row=0, column=4, sticky="w", padx=(0, 8))

        transcribe_all_button = self.ttk.Button(
            controls,
            text="Transcribe All Audio",
        )
        transcribe_all_button.grid(row=0, column=5, sticky="w")

        extract_audio_button = self.ttk.Button(
            controls,
            text="Extract Audio From Videos",
        )
        extract_audio_button.grid(row=0, column=6, sticky="w", padx=(8, 0))

        normalize_audio_button = self.ttk.Button(
            controls,
            text="Normalize Gain (All Audio)",
        )
        normalize_audio_button.grid(row=0, column=7, sticky="w", padx=(8, 0))

        self.ttk.Label(controls, textvariable=transcribe_status_var, style="TLabel").grid(
            row=0, column=8, sticky="w", padx=(10, 0)
        )

        grid_host = self.tk.Frame(
            outer,
            bg=self.bg_panel,
            highlightthickness=1,
            highlightbackground="#4a6ea3",
            bd=0,
        )
        grid_host.grid(row=2, column=0, sticky="nsew")
        grid_host.columnconfigure(0, weight=1)
        grid_host.rowconfigure(0, weight=1)

        editor_canvas = self.tk.Canvas(grid_host, highlightthickness=0, bg=self.bg_panel)
        editor_scroll = self.ttk.Scrollbar(
            grid_host,
            orient="vertical",
            command=editor_canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )
        editor_inner = self.ttk.Frame(editor_canvas)
        editor_inner_id = editor_canvas.create_window((0, 0), window=editor_inner, anchor="nw")
        editor_canvas.configure(yscrollcommand=editor_scroll.set)

        editor_canvas.grid(row=0, column=0, sticky="nsew")
        editor_scroll.grid(row=0, column=1, sticky="ns")

        thumb_refs: list[Any] = []
        active_players: dict[Path, subprocess.Popen[Any]] = {}
        ffplay_path = shutil.which("ffplay")
        ffprobe_path = shutil.which("ffprobe")
        caption_path_by_widget: dict[Any, Path] = {}
        pending_save_by_widget: dict[Any, str] = {}
        caption_widget_by_path: dict[Path, Any] = {}
        audio_duration_cache: dict[Path, float | None] = {}
        enhancement_notice_state = {"remove_music_download_logged": False}

        def _set_transcribe_busy(is_busy: bool, status_text: str = "") -> None:
            button_state = "disabled" if is_busy else "normal"
            for action_button in (transcribe_all_button, extract_audio_button, normalize_audio_button):
                if action_button is not None:
                    try:
                        action_button.configure(state=button_state)
                    except Exception:
                        pass
            transcribe_status_var.set(status_text)

        def _resolve_ffmpeg_executable() -> str:
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError(
                    "ffmpeg was not found in PATH. Install ffmpeg and restart Musubi-Trainer."
                )

            ffmpeg_executable = str(Path(ffmpeg_path).resolve())
            if not Path(ffmpeg_executable).exists():
                raise RuntimeError(
                    f"Resolved ffmpeg path does not exist: {ffmpeg_executable}"
                )
            return ffmpeg_executable

        def _workspace_root_for_downloads() -> Path:
            ws_root_fn = getattr(self, "download_workspace_root", None)
            if callable(ws_root_fn):
                return ws_root_fn()
            return Path(__file__).resolve().parents[3]

        def _ask_auto_split_params() -> tuple[float, float, float, float] | None:
            prompt = self.tk.Toplevel(dialog)
            prompt.withdraw()
            prompt.title("Auto-Split Parameters")
            prompt.transient(dialog)
            prompt.grab_set()
            prompt.configure(bg=self.bg_panel)
            prompt.resizable(False, False)
            self.set_dark_title_bar(prompt)

            root_frame = self.ttk.Frame(prompt, padding=10)
            root_frame.grid(row=0, column=0, sticky="nsew")
            root_frame.columnconfigure(1, weight=1)

            split_min_var = self.tk.StringVar(master=prompt, value=f"{split_defaults['split_min']:.1f}")
            split_max_var = self.tk.StringVar(master=prompt, value=f"{split_defaults['split_max']:.1f}")
            silence_trim_var = self.tk.StringVar(master=prompt, value=f"{split_defaults['silence_trim']:.1f}")
            discard_under_var = self.tk.StringVar(master=prompt, value=f"{split_defaults['discard_under']:.1f}")

            rows = [
                ("Min clip duration (s):", split_min_var),
                ("Max clip duration (s):", split_max_var),
                ("Silence trim (s):", silence_trim_var),
                ("Discard under (s):", discard_under_var),
            ]
            for idx, (label_text, var) in enumerate(rows):
                self.ttk.Label(root_frame, text=label_text, style="TLabel").grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
                self.ttk.Entry(root_frame, textvariable=var, width=10, style="Flat.TEntry").grid(row=idx, column=1, sticky="w", pady=(0, 6))

            result: dict[str, tuple[float, float, float, float] | None] = {"value": None}

            def _submit() -> None:
                try:
                    split_min = float(split_min_var.get().strip())
                    split_max = float(split_max_var.get().strip())
                    silence_trim = float(silence_trim_var.get().strip())
                    discard_under = float(discard_under_var.get().strip())
                except ValueError:
                    self.messagebox.showerror("Auto-Split Audio", "All values must be numeric.", parent=prompt)
                    return

                if split_min <= 0:
                    self.messagebox.showerror("Auto-Split Audio", "Min clip duration must be greater than 0.", parent=prompt)
                    return
                if split_max < split_min:
                    self.messagebox.showerror("Auto-Split Audio", "Max clip duration must be greater than or equal to min clip duration.", parent=prompt)
                    return
                if silence_trim < 0 or discard_under < 0:
                    self.messagebox.showerror("Auto-Split Audio", "Silence trim and discard-under must be non-negative.", parent=prompt)
                    return

                split_defaults["split_min"] = split_min
                split_defaults["split_max"] = split_max
                split_defaults["silence_trim"] = silence_trim
                split_defaults["discard_under"] = discard_under
                result["value"] = (split_min, split_max, silence_trim, discard_under)
                prompt.destroy()

            def _cancel() -> None:
                result["value"] = None
                prompt.destroy()

            buttons = self.ttk.Frame(root_frame)
            buttons.grid(row=len(rows), column=0, columnspan=2, sticky="e", pady=(6, 0))
            self.ttk.Button(buttons, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 6))
            self.ttk.Button(buttons, text="Split", command=_submit).grid(row=0, column=1)

            prompt.protocol("WM_DELETE_WINDOW", _cancel)
            self.center_window(prompt)
            prompt.deiconify()
            dialog.wait_window(prompt)
            return result["value"]

        def _normalize_media_path(path_value: Path) -> Path:
            dataset_dir = self.dataset_dir_path(dataset_name)
            candidate_path = Path(path_value)
            if not candidate_path.is_absolute():
                candidate = dataset_dir / candidate_path.name
                if candidate.exists():
                    candidate_path = candidate
            return candidate_path.resolve()

        def _audio_duration_seconds(audio_path: Path) -> float | None:
            normalized = _normalize_media_path(audio_path)
            cached = audio_duration_cache.get(normalized)
            if cached is not None or normalized in audio_duration_cache:
                return cached

            duration_s: float | None = None
            if ffprobe_path is not None:
                try:
                    probe = subprocess.run(
                        [
                            ffprobe_path,
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            str(normalized),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if probe.returncode == 0:
                        raw = (probe.stdout or "").strip()
                        if raw:
                            parsed = float(raw)
                            duration_s = parsed if parsed > 0 else None
                except Exception:
                    duration_s = None

            if duration_s is None and normalized.suffix.lower() == ".wav":
                try:
                    with wave.open(str(normalized), "rb") as wav_file:
                        frame_rate = float(wav_file.getframerate() or 0)
                        frame_count = float(wav_file.getnframes() or 0)
                        if frame_rate > 0:
                            duration_s = frame_count / frame_rate
                except Exception:
                    duration_s = None

            audio_duration_cache[normalized] = duration_s
            return duration_s

        def _is_long_audio_clip(audio_path: Path, threshold_s: float = 15.0) -> bool:
            duration_s = _audio_duration_seconds(audio_path)
            return duration_s is not None and duration_s > threshold_s

        def _delete_media_item(media_path: Path, *, ask_confirmation: bool = True) -> None:
            normalized_media_path = _normalize_media_path(media_path)
            if ask_confirmation:
                confirmed = self.messagebox.askyesno(
                    "Delete clip",
                    f"Delete this clip and its caption?\n\n{normalized_media_path.name}",
                    parent=dialog,
                )
                if not confirmed:
                    return

            stop_media(normalized_media_path)
            caption_path = normalized_media_path.with_suffix(".txt")
            widget = caption_widget_by_path.get(caption_path)
            if widget is not None:
                flush_caption_save(widget)

            try:
                if normalized_media_path.exists():
                    normalized_media_path.unlink()
                if caption_path.exists():
                    caption_path.unlink()
            except OSError as exc:
                self.messagebox.showerror("Delete clip", f"Could not delete clip:\n{exc}", parent=dialog)
                return

            dialog.destroy()
            self.root.after(0, lambda: self.open(dataset_name))

        def _run_whisper_transcription(audio_file_paths: list[Path], replace_existing: bool) -> tuple[int, int, int]:
            if not audio_file_paths:
                return (0, 0, 0)

            def _terminal_log(message: str) -> None:
                print(f"[Dataset Editor] {message}", flush=True)

            model_choice = (transcribe_model_var.get().strip() or "Medium").lower()
            language_choice = transcribe_language_var.get().strip()
            options: dict[str, str] = {}
            lang_code = WHISPER_LANGUAGE_CODES.get(language_choice)
            if lang_code:
                options["language"] = lang_code

            download_root = _workspace_root_for_downloads() / "Models" / "whisper"
            download_root.mkdir(parents=True, exist_ok=True)

            self.log(
                f"[Dataset Editor] Whisper transcribe start for '{dataset_name}' ({len(audio_file_paths)} clip(s), model={model_choice}, replace={replace_existing})"
            )
            _terminal_log(
                f"Whisper transcribe start for '{dataset_name}' ({len(audio_file_paths)} clip(s), model={model_choice}, replace={replace_existing})"
            )

            def _caption_has_content(caption_path: Path) -> bool:
                if not caption_path.exists() or not caption_path.is_file():
                    return False
                try:
                    return bool(caption_path.read_text(encoding="utf-8").strip())
                except OSError:
                    return False

            transcribed = 0
            skipped = 0
            errors = 0
            dataset_dir = self.dataset_dir_path(dataset_name)

            eligible_audio_paths: list[Path] = []
            for audio_path in audio_file_paths:
                normalized_audio_path = Path(audio_path)
                if not normalized_audio_path.is_absolute():
                    candidate = dataset_dir / normalized_audio_path.name
                    if candidate.exists():
                        normalized_audio_path = candidate
                normalized_audio_path = normalized_audio_path.resolve()

                if not normalized_audio_path.exists():
                    errors += 1
                    self.log(
                        f"[Dataset Editor] Whisper failed for '{normalized_audio_path.name}': file not found ({normalized_audio_path})"
                    )
                    _terminal_log(
                        f"Whisper failed for '{normalized_audio_path.name}': file not found ({normalized_audio_path})"
                    )
                    try:
                        dialog.after(
                            0,
                            lambda c=transcribed, s=skipped, e=errors, t=len(audio_file_paths): transcribe_status_var.set(
                                f"Transcribing... {c}/{t} (skipped {s}, errors {e})"
                            ),
                        )
                    except Exception:
                        pass
                    continue

                caption_path = normalized_audio_path.with_suffix(".txt")
                if (not replace_existing) and _caption_has_content(caption_path):
                    skipped += 1
                    _terminal_log(
                        f"Skipped '{normalized_audio_path.name}' (caption is non-empty; Replace Existing is off)"
                    )
                    try:
                        dialog.after(
                            0,
                            lambda c=transcribed, s=skipped, e=errors, t=len(audio_file_paths): transcribe_status_var.set(
                                f"Transcribing... {c}/{t} (skipped {s}, errors {e})"
                            ),
                        )
                    except Exception:
                        pass
                    continue

                eligible_audio_paths.append(normalized_audio_path)

            if not eligible_audio_paths:
                self.log(
                    f"[Dataset Editor] Whisper transcribe complete for '{dataset_name}': transcribed={transcribed}, skipped={skipped}, errors={errors}"
                )
                _terminal_log(
                    f"No eligible clips to transcribe for '{dataset_name}' (all skipped/missing)."
                )
                _terminal_log(
                    f"Whisper transcribe complete for '{dataset_name}': transcribed={transcribed}, skipped={skipped}, errors={errors}"
                )
                return (transcribed, skipped, errors)

            ffmpeg_executable = _resolve_ffmpeg_executable()

            try:
                ffmpeg_probe = subprocess.run(
                    [ffmpeg_executable, "-version"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ffmpeg_probe.returncode != 0:
                    probe_err = (ffmpeg_probe.stderr or ffmpeg_probe.stdout or "").strip()
                    raise RuntimeError(probe_err or "ffmpeg -version failed")
            except Exception as exc:
                raise RuntimeError(
                    f"Whisper ffmpeg probe failed at '{ffmpeg_executable}': {exc}"
                ) from exc

            import whisper  # type: ignore[import-not-found]
            import numpy as np

            import whisper.audio as whisper_audio  # type: ignore[import-not-found]

            sample_rate = int(getattr(whisper_audio, "SAMPLE_RATE", 16000))

            def _load_audio_with_explicit_ffmpeg(file: str, sr: int = sample_rate) -> Any:
                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-nostdin",
                    "-threads",
                    "0",
                    "-i",
                    str(file),
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sr),
                    "-",
                ]
                try:
                    completed = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                except subprocess.CalledProcessError as ffmpeg_error:
                    stderr_text = (ffmpeg_error.stderr or b"").decode(errors="ignore").strip()
                    raise RuntimeError(
                        f"Failed to load audio via ffmpeg for '{file}': {stderr_text or ffmpeg_error}"
                    ) from ffmpeg_error

                return np.frombuffer(completed.stdout, np.int16).flatten().astype(np.float32) / 32768.0

            whisper_audio.load_audio = _load_audio_with_explicit_ffmpeg

            model = whisper.load_model(model_choice, download_root=str(download_root))

            for normalized_audio_path in eligible_audio_paths:
                caption_path = normalized_audio_path.with_suffix(".txt")

                try:
                    result = model.transcribe(str(normalized_audio_path), **options)
                    text = str(result.get("text", "")).strip()
                    caption_path.write_text(text, encoding="utf-8")
                    transcribed += 1
                    _terminal_log(
                        f"Transcribed '{normalized_audio_path.name}' -> '{caption_path.name}' ({len(text)} chars)"
                    )

                    widget = caption_widget_by_path.get(caption_path)
                    if widget is not None:
                        def _update_widget(target_widget: Any, target_text: str) -> None:
                            target_widget.delete("1.0", "end")
                            target_widget.insert("1.0", target_text)
                            target_widget.edit_modified(False)

                        try:
                            dialog.after(0, lambda w=widget, t=text: _update_widget(w, t))
                        except Exception:
                            pass
                except Exception as exc:
                    errors += 1
                    self.log(
                        f"[Dataset Editor] Whisper failed for '{normalized_audio_path.name}' ({normalized_audio_path}): {exc!r}"
                    )
                    _terminal_log(
                        f"Whisper failed for '{normalized_audio_path.name}' ({normalized_audio_path}): {exc!r}"
                    )

                try:
                    dialog.after(
                        0,
                        lambda c=transcribed, s=skipped, e=errors, t=len(audio_file_paths): transcribe_status_var.set(
                            f"Transcribing... {c}/{t} (skipped {s}, errors {e})"
                        ),
                    )
                except Exception:
                    pass

            self.log(
                f"[Dataset Editor] Whisper transcribe complete for '{dataset_name}': transcribed={transcribed}, skipped={skipped}, errors={errors}"
            )
            _terminal_log(
                f"Whisper transcribe complete for '{dataset_name}': transcribed={transcribed}, skipped={skipped}, errors={errors}"
            )
            return (transcribed, skipped, errors)

        def _start_whisper_transcription(audio_file_paths: list[Path], replace_existing: bool) -> None:
            if not audio_file_paths:
                transcribe_status_var.set("No audio clips found.")
                return

            _set_transcribe_busy(True, "Loading Whisper model...")

            def worker() -> None:
                try:
                    transcribed, skipped, errors = _run_whisper_transcription(audio_file_paths, replace_existing)
                    dialog.after(
                        0,
                        lambda: _set_transcribe_busy(
                            False,
                            f"Done: {transcribed} transcribed, {skipped} skipped, {errors} errors",
                        ),
                    )
                except ImportError:
                    dialog.after(
                        0,
                        lambda: _set_transcribe_busy(False, "Whisper package is not installed."),
                    )
                    dialog.after(
                        0,
                        lambda: self.messagebox.showerror(
                            "Transcribe Audio",
                            "Whisper is not installed in this environment.\nInstall package 'openai-whisper' to enable transcription.",
                            parent=dialog,
                        ),
                    )
                except Exception as exc:
                    error_text = str(exc)
                    self.log(f"[Dataset Editor] Whisper transcription failed for '{dataset_name}': {error_text}")
                    dialog.after(0, lambda: _set_transcribe_busy(False, "Transcription failed."))
                    dialog.after(0, lambda: self.messagebox.showerror("Transcribe Audio", error_text, parent=dialog))

            threading.Thread(target=worker, name="dataset-editor-whisper", daemon=True).start()

        def _split_into_phrase_segments(
            full_text: str,
            word_rows: list[dict[str, Any]],
            min_duration: float,
            max_duration: float,
            silence_trim: float,
            discard_under: float,
        ) -> list[tuple[float, float, str]]:
            if not word_rows or not full_text.strip():
                return []

            class _WordTs:
                def __init__(self, text: str, start_time: float, end_time: float) -> None:
                    self.text = text
                    self.start_time = start_time
                    self.end_time = end_time

            all_words: list[_WordTs] = []
            for row in word_rows:
                text = str(row.get("word", "")).strip()
                if not text:
                    continue
                start = float(row.get("start", 0.0) or 0.0)
                end = float(row.get("end", start) or start)
                if end < start:
                    end = start
                all_words.append(_WordTs(text, start, end))

            if not all_words:
                return []

            text_tokens = full_text.split()
            sample = all_words[: min(30, len(all_words))]
            has_punct_in_words = any(re.search(r"[.!?,]", w.text) for w in sample)

            if has_punct_in_words or len(text_tokens) != len(all_words):
                def get_word_text(i: int) -> str:
                    return all_words[i].text.strip()
            else:
                def get_word_text(i: int) -> str:
                    return text_tokens[i]

            sentence_ranges: list[tuple[int, int, str]] = []
            sent_start = 0
            for i in range(len(all_words)):
                is_last = i == len(all_words) - 1
                word_text = get_word_text(i)
                ends_sentence = bool(re.search(r"[.!?][\"')]}]*$", word_text))
                if ends_sentence or is_last:
                    sentence_text = " ".join(get_word_text(j) for j in range(sent_start, i + 1)).strip()
                    sentence_ranges.append((sent_start, i, sentence_text))
                    sent_start = i + 1

            if not sentence_ranges:
                return []

            silence_cuts: set[int] = set()
            if silence_trim > 0:
                for i in range(len(all_words) - 1):
                    gap = all_words[i + 1].start_time - all_words[i].end_time
                    if gap > silence_trim:
                        silence_cuts.add(i)

            segments: list[tuple[float, float, str]] = []
            group_texts: list[str] = []
            group_word_start = sentence_ranges[0][0]

            for si, (s_start, s_end, s_text) in enumerate(sentence_ranges):
                is_last_sentence = si == len(sentence_ranges) - 1

                cuts_in_sentence = sorted(c for c in silence_cuts if s_start <= c < s_end)
                if cuts_in_sentence:
                    if group_texts:
                        grp_start = all_words[group_word_start].start_time
                        grp_end = all_words[s_start - 1].end_time if s_start > 0 else grp_start
                        combined = " ".join(group_texts).strip()
                        if combined:
                            segments.append((grp_start, grp_end, combined))
                        group_texts = []

                    boundaries = [s_start] + [c + 1 for c in cuts_in_sentence] + [s_end + 1]
                    for bi in range(len(boundaries) - 1):
                        chunk_start = boundaries[bi]
                        chunk_end = boundaries[bi + 1] - 1
                        if chunk_end < chunk_start:
                            continue
                        chunk_text = " ".join(get_word_text(wi) for wi in range(chunk_start, chunk_end + 1)).strip()
                        if not chunk_text:
                            continue
                        segments.append((all_words[chunk_start].start_time, all_words[chunk_end].end_time, chunk_text))

                    if not is_last_sentence:
                        group_word_start = sentence_ranges[si + 1][0]
                    continue

                if group_texts and s_start > 0 and (s_start - 1) in silence_cuts:
                    grp_start = all_words[group_word_start].start_time
                    grp_end = all_words[s_start - 1].end_time
                    combined = " ".join(group_texts).strip()
                    if combined:
                        segments.append((grp_start, grp_end, combined))
                    group_texts = []
                    group_word_start = s_start

                group_texts.append(s_text)
                group_end_idx = s_end
                grp_start_time = all_words[group_word_start].start_time
                grp_end_time = all_words[min(group_end_idx, len(all_words) - 1)].end_time
                grp_duration = grp_end_time - grp_start_time

                next_crosses_silence = False
                if not is_last_sentence:
                    next_s_start = sentence_ranges[si + 1][0]
                    for wi in range(group_end_idx, next_s_start):
                        if wi in silence_cuts:
                            next_crosses_silence = True
                            break

                if grp_duration >= min_duration or is_last_sentence or next_crosses_silence:
                    combined = " ".join(group_texts).strip()
                    if combined:
                        segments.append((grp_start_time, grp_end_time, combined))
                    group_texts = []
                    if not is_last_sentence:
                        group_word_start = sentence_ranges[si + 1][0]

            if max_duration > 0:
                final_segments: list[tuple[float, float, str]] = []
                for seg_start, seg_end, seg_text in segments:
                    seg_duration = seg_end - seg_start
                    if seg_duration <= max_duration:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    seg_word_indices = [
                        i
                        for i, w in enumerate(all_words)
                        if w.start_time >= seg_start - 0.01 and w.end_time <= seg_end + 0.01
                    ]
                    if not seg_word_indices:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    comma_indices = [wi for wi in seg_word_indices if get_word_text(wi).endswith(",")]
                    if not comma_indices:
                        final_segments.append((seg_start, seg_end, seg_text))
                        continue

                    sub_start_idx = seg_word_indices[0]
                    for ci, comma_wi in enumerate(comma_indices):
                        is_last_comma = ci == len(comma_indices) - 1
                        sub_start_time = all_words[sub_start_idx].start_time
                        sub_end_time = all_words[comma_wi].end_time
                        sub_duration = sub_end_time - sub_start_time
                        should_cut = sub_duration >= min_duration and (sub_duration >= max_duration or is_last_comma)
                        if should_cut:
                            sub_text = " ".join(get_word_text(j) for j in range(sub_start_idx, comma_wi + 1)).strip()
                            if sub_text:
                                final_segments.append((sub_start_time, sub_end_time, sub_text))
                            sub_start_idx = comma_wi + 1

                    last_word_idx = seg_word_indices[-1]
                    if sub_start_idx <= last_word_idx:
                        sub_text = " ".join(get_word_text(j) for j in range(sub_start_idx, last_word_idx + 1)).strip()
                        if sub_text:
                            final_segments.append((all_words[sub_start_idx].start_time, all_words[last_word_idx].end_time, sub_text))

                # Hard safety pass: if punctuation-based splitting cannot reduce duration
                # (e.g. long span without commas), enforce max_duration by word timings.
                strict_segments: list[tuple[float, float, str]] = []
                for seg_start, seg_end, seg_text in final_segments:
                    seg_duration = seg_end - seg_start
                    if seg_duration <= max_duration:
                        strict_segments.append((seg_start, seg_end, seg_text))
                        continue

                    seg_word_indices = [
                        i
                        for i, w in enumerate(all_words)
                        if w.start_time >= seg_start - 0.01 and w.end_time <= seg_end + 0.01
                    ]
                    if not seg_word_indices:
                        strict_segments.append((seg_start, seg_end, seg_text))
                        continue

                    chunk_start_idx = seg_word_indices[0]
                    last_word_idx = seg_word_indices[-1]
                    while chunk_start_idx <= last_word_idx:
                        chunk_end_idx = chunk_start_idx
                        while chunk_end_idx < last_word_idx:
                            next_idx = chunk_end_idx + 1
                            next_duration = all_words[next_idx].end_time - all_words[chunk_start_idx].start_time
                            if next_duration > max_duration:
                                break
                            chunk_end_idx = next_idx

                        chunk_text = " ".join(get_word_text(j) for j in range(chunk_start_idx, chunk_end_idx + 1)).strip()
                        if chunk_text:
                            strict_segments.append(
                                (
                                    all_words[chunk_start_idx].start_time,
                                    all_words[chunk_end_idx].end_time,
                                    chunk_text,
                                )
                            )

                        if chunk_end_idx >= last_word_idx:
                            break
                        chunk_start_idx = chunk_end_idx + 1

                segments = strict_segments

            if discard_under > 0:
                segments = [s for s in segments if (s[1] - s[0]) >= discard_under]

            return segments

        def _run_auto_split_audio(
            audio_file_paths: list[Path],
            split_min: float,
            split_max: float,
            silence_trim: float,
            discard_under: float,
        ) -> tuple[int, int, int, list[Path]]:
            if not audio_file_paths:
                return (0, 0, 0, [])

            def _terminal_log(message: str) -> None:
                print(f"[Dataset Editor] {message}", flush=True)

            if split_min <= 0:
                raise RuntimeError("Min clip duration must be greater than 0.")
            if split_max < split_min:
                raise RuntimeError("Max clip duration must be greater than or equal to min clip duration.")
            if silence_trim < 0 or discard_under < 0:
                raise RuntimeError("Silence trim and discard-under must be non-negative.")

            ffmpeg_executable = _resolve_ffmpeg_executable()
            model_choice = (transcribe_model_var.get().strip() or "Medium").lower()
            language_choice = transcribe_language_var.get().strip()
            whisper_options: dict[str, Any] = {"word_timestamps": True}
            lang_code = WHISPER_LANGUAGE_CODES.get(language_choice)
            if lang_code:
                whisper_options["language"] = lang_code

            download_root = _workspace_root_for_downloads() / "Models" / "whisper"
            download_root.mkdir(parents=True, exist_ok=True)

            try:
                ffmpeg_probe = subprocess.run(
                    [ffmpeg_executable, "-version"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ffmpeg_probe.returncode != 0:
                    probe_err = (ffmpeg_probe.stderr or ffmpeg_probe.stdout or "").strip()
                    raise RuntimeError(probe_err or "ffmpeg -version failed")
            except Exception as exc:
                raise RuntimeError(
                    f"Whisper ffmpeg probe failed at '{ffmpeg_executable}': {exc}"
                ) from exc

            import whisper  # type: ignore[import-not-found]
            import numpy as np
            import whisper.audio as whisper_audio  # type: ignore[import-not-found]

            sample_rate = int(getattr(whisper_audio, "SAMPLE_RATE", 16000))

            def _load_audio_with_explicit_ffmpeg(file: str, sr: int = sample_rate) -> Any:
                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-nostdin",
                    "-threads",
                    "0",
                    "-i",
                    str(file),
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sr),
                    "-",
                ]
                try:
                    completed = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                except subprocess.CalledProcessError as ffmpeg_error:
                    stderr_text = (ffmpeg_error.stderr or b"").decode(errors="ignore").strip()
                    raise RuntimeError(
                        f"Failed to load audio via ffmpeg for '{file}': {stderr_text or ffmpeg_error}"
                    ) from ffmpeg_error

                return np.frombuffer(completed.stdout, np.int16).flatten().astype(np.float32) / 32768.0

            whisper_audio.load_audio = _load_audio_with_explicit_ffmpeg

            model = whisper.load_model(model_choice, download_root=str(download_root))

            dataset_dir = self.dataset_dir_path(dataset_name)
            split_files = 0
            skipped_files = 0
            errors = 0
            split_sources: list[Path] = []

            _terminal_log(
                (
                    f"Auto-split start for '{dataset_name}' ({len(audio_file_paths)} clip(s), model={model_choice}, "
                    f"split_min={split_min:.2f}, split_max={split_max:.2f}, silence_trim={silence_trim:.2f}, discard_under={discard_under:.2f})"
                )
            )

            for source_path in audio_file_paths:
                normalized_audio_path = Path(source_path)
                if not normalized_audio_path.is_absolute():
                    candidate = dataset_dir / normalized_audio_path.name
                    if candidate.exists():
                        normalized_audio_path = candidate
                normalized_audio_path = normalized_audio_path.resolve()

                if not normalized_audio_path.exists():
                    errors += 1
                    _terminal_log(f"Auto-split failed for '{normalized_audio_path.name}': file not found ({normalized_audio_path})")
                    continue

                try:
                    result = model.transcribe(str(normalized_audio_path), **whisper_options)
                    segments = result.get("segments", []) if isinstance(result, dict) else []
                    if not segments:
                        skipped_files += 1
                        _terminal_log(f"Skipped '{normalized_audio_path.name}': no speech segments found")
                        continue

                    full_text = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
                    if not full_text:
                        skipped_files += 1
                        _terminal_log(f"Skipped '{normalized_audio_path.name}': empty transcript")
                        continue

                    all_words: list[dict[str, Any]] = []
                    for seg in segments:
                        if not isinstance(seg, dict):
                            continue
                        for w in seg.get("words", []) or []:
                            if isinstance(w, dict):
                                all_words.append(w)

                    if not all_words:
                        skipped_files += 1
                        _terminal_log(f"Skipped '{normalized_audio_path.name}': no word timestamps from ASR")
                        continue

                    base_name = normalized_audio_path.stem
                    chunk_index = 1
                    created_for_file = 0

                    phrase_segments = _split_into_phrase_segments(
                        full_text,
                        all_words,
                        min_duration=split_min,
                        max_duration=split_max,
                        silence_trim=silence_trim,
                        discard_under=discard_under,
                    )

                    if not phrase_segments:
                        skipped_files += 1
                        _terminal_log(f"Skipped '{normalized_audio_path.name}': no valid phrase segments")
                        continue

                    _terminal_log(
                        (
                            f"Phrase split '{normalized_audio_path.name}': {len(phrase_segments)} segment(s), "
                            f"min={split_min:.2f}s, max={split_max:.2f}s"
                        )
                    )

                    for start, end, chunk_text in phrase_segments:
                        if (end - start) < 0.35:
                            continue
                        chunk_text = re.sub(r"\s+", " ", chunk_text.strip())
                        if not chunk_text:
                            continue

                        chunk_stem = f"{base_name}_chunk_{chunk_index:03d}"
                        chunk_audio = normalized_audio_path.with_name(f"{chunk_stem}.wav")
                        while chunk_audio.exists():
                            chunk_index += 1
                            chunk_stem = f"{base_name}_chunk_{chunk_index:03d}"
                            chunk_audio = normalized_audio_path.with_name(f"{chunk_stem}.wav")

                        ffmpeg_cmd = [
                            ffmpeg_executable,
                            "-y",
                            "-nostdin",
                            "-loglevel",
                            "error",
                            "-ss",
                            f"{start:.3f}",
                            "-to",
                            f"{end:.3f}",
                            "-i",
                            str(normalized_audio_path),
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            str(chunk_audio),
                        ]
                        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                        if ffmpeg_result.returncode != 0:
                            stderr_text = (ffmpeg_result.stderr or "").strip()
                            raise RuntimeError(stderr_text or "ffmpeg split command failed")

                        chunk_caption = chunk_audio.with_suffix(".txt")
                        chunk_caption.write_text(chunk_text, encoding="utf-8")
                        created_for_file += 1
                        split_files += 1
                        _terminal_log(
                            f"Split '{normalized_audio_path.name}' -> '{chunk_audio.name}' ({end - start:.2f}s, {len(chunk_text)} chars)"
                        )

                        chunk_index += 1

                    if created_for_file == 0:
                        skipped_files += 1
                        _terminal_log(f"Skipped '{normalized_audio_path.name}': no chunks created")
                    else:
                        split_sources.append(normalized_audio_path)

                except Exception as exc:
                    errors += 1
                    _terminal_log(f"Auto-split failed for '{normalized_audio_path.name}': {exc!r}")

            _terminal_log(
                f"Auto-split complete for '{dataset_name}': chunks_created={split_files}, skipped={skipped_files}, errors={errors}"
            )
            return (split_files, skipped_files, errors, split_sources)

        def _run_extract_audio_from_videos(video_file_paths: list[Path]) -> tuple[int, int]:
            if not video_file_paths:
                return (0, 0)

            ffmpeg_executable = _resolve_ffmpeg_executable()
            dataset_dir = self.dataset_dir_path(dataset_name)
            extracted = 0
            errors = 0

            self.log(
                f"[Dataset Editor] Audio extraction start for '{dataset_name}' ({len(video_file_paths)} video clip(s))"
            )

            for source_path in video_file_paths:
                normalized_video_path = Path(source_path)
                if not normalized_video_path.is_absolute():
                    candidate = dataset_dir / normalized_video_path.name
                    if candidate.exists():
                        normalized_video_path = candidate
                normalized_video_path = normalized_video_path.resolve()

                if not normalized_video_path.exists():
                    errors += 1
                    self.log(
                        f"[Dataset Editor] Audio extraction failed for '{normalized_video_path.name}': file not found"
                    )
                    continue

                out_path = normalized_video_path.with_name(f"{normalized_video_path.stem}_audio.wav")
                suffix_index = 1
                while out_path.exists():
                    out_path = normalized_video_path.with_name(
                        f"{normalized_video_path.stem}_audio_{suffix_index:03d}.wav"
                    )
                    suffix_index += 1

                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-y",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(normalized_video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-acodec",
                    "pcm_s16le",
                    str(out_path),
                ]
                ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                if ffmpeg_result.returncode != 0:
                    errors += 1
                    stderr_text = (ffmpeg_result.stderr or "").strip()
                    self.log(
                        f"[Dataset Editor] Audio extraction failed for '{normalized_video_path.name}': {stderr_text or 'ffmpeg failed'}"
                    )
                    continue

                source_caption = normalized_video_path.with_suffix(".txt")
                target_caption = out_path.with_suffix(".txt")
                if source_caption.exists() and source_caption.is_file() and not target_caption.exists():
                    try:
                        target_caption.write_text(source_caption.read_text(encoding="utf-8"), encoding="utf-8")
                    except OSError:
                        pass

                extracted += 1
                self.log(
                    f"[Dataset Editor] Extracted audio: '{normalized_video_path.name}' -> '{out_path.name}'"
                )

            self.log(
                f"[Dataset Editor] Audio extraction complete for '{dataset_name}': extracted={extracted}, errors={errors}"
            )
            return (extracted, errors)

        def _start_extract_audio_from_videos(video_file_paths: list[Path]) -> None:
            if not video_file_paths:
                transcribe_status_var.set("No video clips found.")
                return

            _set_transcribe_busy(True, "Extracting audio from videos...")

            def worker() -> None:
                try:
                    extracted, errors = _run_extract_audio_from_videos(video_file_paths)

                    def _finish_extract() -> None:
                        _set_transcribe_busy(False, f"Extracted {extracted} clip(s), {errors} errors")
                        if extracted > 0:
                            dialog.destroy()
                            self.root.after(0, lambda: self.open(dataset_name))

                    dialog.after(0, _finish_extract)
                except Exception as exc:
                    error_text = str(exc)
                    self.log(f"[Dataset Editor] Audio extraction failed for '{dataset_name}': {error_text}")
                    dialog.after(0, lambda: _set_transcribe_busy(False, "Audio extraction failed."))
                    dialog.after(0, lambda: self.messagebox.showerror("Extract Audio", error_text, parent=dialog))

            threading.Thread(target=worker, name="dataset-editor-extract-audio", daemon=True).start()

        def _run_normalize_audio_gain(audio_file_paths: list[Path]) -> tuple[int, int]:
            if not audio_file_paths:
                return (0, 0)

            ffmpeg_executable = _resolve_ffmpeg_executable()
            dataset_dir = self.dataset_dir_path(dataset_name)
            normalized = 0
            errors = 0

            self.log(
                f"[Dataset Editor] Gain normalization start for '{dataset_name}' ({len(audio_file_paths)} clip(s))"
            )

            for source_path in audio_file_paths:
                normalized_audio_path = Path(source_path)
                if not normalized_audio_path.is_absolute():
                    candidate = dataset_dir / normalized_audio_path.name
                    if candidate.exists():
                        normalized_audio_path = candidate
                normalized_audio_path = normalized_audio_path.resolve()

                if not normalized_audio_path.exists():
                    errors += 1
                    self.log(
                        f"[Dataset Editor] Gain normalization failed for '{normalized_audio_path.name}': file not found"
                    )
                    continue

                temp_out = normalized_audio_path.with_name(f"{normalized_audio_path.stem}__norm_tmp.wav")
                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-y",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(normalized_audio_path),
                    "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-acodec",
                    "pcm_s16le",
                    str(temp_out),
                ]
                ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                if ffmpeg_result.returncode != 0:
                    errors += 1
                    stderr_text = (ffmpeg_result.stderr or "").strip()
                    self.log(
                        f"[Dataset Editor] Gain normalization failed for '{normalized_audio_path.name}': {stderr_text or 'ffmpeg failed'}"
                    )
                    continue

                final_out = normalized_audio_path.with_name(f"{normalized_audio_path.stem}_norm.wav")
                suffix_index = 1
                while final_out.exists() and final_out != normalized_audio_path:
                    final_out = normalized_audio_path.with_name(
                        f"{normalized_audio_path.stem}_norm_{suffix_index:03d}.wav"
                    )
                    suffix_index += 1

                try:
                    shutil.move(str(temp_out), str(final_out))
                    source_caption = normalized_audio_path.with_suffix(".txt")
                    target_caption = final_out.with_suffix(".txt")
                    if source_caption.exists() and source_caption.is_file() and not target_caption.exists():
                        target_caption.write_text(source_caption.read_text(encoding="utf-8"), encoding="utf-8")
                    normalized += 1
                    self.log(
                        f"[Dataset Editor] Normalized gain: '{normalized_audio_path.name}' -> '{final_out.name}'"
                    )
                except OSError as exc:
                    errors += 1
                    self.log(
                        f"[Dataset Editor] Gain normalization move failed for '{normalized_audio_path.name}': {exc}"
                    )

            self.log(
                f"[Dataset Editor] Gain normalization complete for '{dataset_name}': normalized={normalized}, errors={errors}"
            )
            return (normalized, errors)

        def _start_normalize_audio_gain(audio_file_paths: list[Path]) -> None:
            if not audio_file_paths:
                transcribe_status_var.set("No audio clips found.")
                return

            _set_transcribe_busy(True, "Normalizing audio gain...")

            def worker() -> None:
                try:
                    normalized_count, errors = _run_normalize_audio_gain(audio_file_paths)

                    def _finish_normalize() -> None:
                        _set_transcribe_busy(False, f"Normalized {normalized_count} clip(s), {errors} errors")
                        if normalized_count > 0:
                            dialog.destroy()
                            self.root.after(0, lambda: self.open(dataset_name))

                    dialog.after(0, _finish_normalize)
                except Exception as exc:
                    error_text = str(exc)
                    self.log(f"[Dataset Editor] Gain normalization failed for '{dataset_name}': {error_text}")
                    dialog.after(0, lambda: _set_transcribe_busy(False, "Gain normalization failed."))
                    dialog.after(0, lambda: self.messagebox.showerror("Normalize Gain", error_text, parent=dialog))

            threading.Thread(target=worker, name="dataset-editor-normalize-gain", daemon=True).start()

        def _run_audio_enhancement_preset(audio_file_paths: list[Path], preset_key: str) -> tuple[int, int]:
            if not audio_file_paths:
                return (0, 0)

            scripts_dir = Path(sys.executable).resolve().parent
            preset_map: dict[str, dict[str, str]] = {
                "deepfilternet": {
                    "label": "Denoise",
                    "suffix": "dfn",
                    "exe": str(scripts_dir / "deepFilter.exe"),
                },
                "melband_roformer": {
                    "label": "Remove Music",
                    "suffix": "mbr",
                    "exe": str(scripts_dir / "audio-separator.exe"),
                    "model": "melband_roformer_inst_v1.ckpt",
                },
            }
            preset = preset_map.get(preset_key)
            if preset is None:
                raise RuntimeError(f"Unknown enhancement preset: {preset_key}")

            if preset_key == "melband_roformer" and not enhancement_notice_state["remove_music_download_logged"]:
                message = (
                    "[Dataset Editor] Remove Music first run: downloading model if missing in cache. "
                    "This may take a few minutes."
                )
                self.log(message)
                print(message, flush=True)
                enhancement_notice_state["remove_music_download_logged"] = True

            dataset_dir = self.dataset_dir_path(dataset_name)
            processed = 0
            errors = 0

            self.log(
                f"[Dataset Editor] {preset['label']} start for '{dataset_name}' ({len(audio_file_paths)} clip(s))"
            )

            for source_path in audio_file_paths:
                normalized_audio_path = Path(source_path)
                if not normalized_audio_path.is_absolute():
                    candidate = dataset_dir / normalized_audio_path.name
                    if candidate.exists():
                        normalized_audio_path = candidate
                normalized_audio_path = normalized_audio_path.resolve()

                if not normalized_audio_path.exists():
                    errors += 1
                    self.log(
                        f"[Dataset Editor] {preset['label']} failed for '{normalized_audio_path.name}': file not found"
                    )
                    continue

                output_dir = normalized_audio_path.parent
                cli_executable = preset.get("exe", "")
                if not cli_executable or (not Path(cli_executable).exists()):
                    errors += 1
                    self.log(
                        f"[Dataset Editor] {preset['label']} failed: CLI not found at '{cli_executable}'"
                    )
                    continue

                run_started_at = time.time()
                before_outputs = {
                    path.resolve()
                    for path in output_dir.glob("*.wav")
                    if path.is_file()
                }

                if preset_key == "deepfilternet":
                    # DeepFilterNet currently breaks with torchaudio>=2.11 due removed torchaudio.backend API.
                    cli_cmd = [
                        cli_executable,
                        str(normalized_audio_path),
                        "-o",
                        str(output_dir),
                    ]
                else:
                    cli_cmd = [
                        cli_executable,
                        str(normalized_audio_path),
                        "-m",
                        str(preset.get("model", "melband_roformer_inst_v1.ckpt")),
                        "--output_dir",
                        str(output_dir),
                        "--output_format",
                        "WAV",
                        "--single_stem",
                        "Vocals",
                    ]

                cli_result = subprocess.run(cli_cmd, capture_output=True, text=True, check=False)
                if cli_result.returncode != 0:
                    errors += 1
                    stderr_text = (cli_result.stderr or "").strip()
                    stdout_text = (cli_result.stdout or "").strip()
                    if preset_key == "deepfilternet" and "torchaudio.backend" in (stderr_text + "\n" + stdout_text):
                        self.log(
                            "[Dataset Editor] DeepFilterNet failed: installed package is incompatible with current torchaudio API "
                            "(missing 'torchaudio.backend')."
                        )
                    else:
                        self.log(
                            f"[Dataset Editor] {preset['label']} failed for '{normalized_audio_path.name}': "
                            f"{stderr_text or stdout_text or 'tool failed'}"
                        )
                    continue

                after_outputs = {
                    path.resolve()
                    for path in output_dir.glob("*.wav")
                    if path.is_file()
                }
                changed_outputs: list[Path] = []
                for output_path in after_outputs:
                    try:
                        if output_path not in before_outputs or output_path.stat().st_mtime >= (run_started_at - 0.5):
                            changed_outputs.append(output_path)
                    except OSError:
                        continue

                if preset_key == "melband_roformer":
                    vocals_outputs = [path for path in changed_outputs if "vocals" in path.name.lower()]
                    if vocals_outputs:
                        changed_outputs = vocals_outputs

                if not changed_outputs:
                    errors += 1
                    self.log(
                        f"[Dataset Editor] {preset['label']} failed for '{normalized_audio_path.name}': no output file was created or updated"
                    )
                    continue

                changed_outputs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                produced_output = changed_outputs[0]
                final_out = normalized_audio_path.with_name(f"{normalized_audio_path.stem}_{preset['suffix']}.wav")
                suffix_index = 1
                while final_out.exists() and final_out != produced_output:
                    final_out = normalized_audio_path.with_name(
                        f"{normalized_audio_path.stem}_{preset['suffix']}_{suffix_index:03d}.wav"
                    )
                    suffix_index += 1

                try:
                    if produced_output.resolve() != final_out.resolve():
                        if final_out.exists():
                            final_out.unlink()
                        shutil.move(str(produced_output), str(final_out))
                    source_caption = normalized_audio_path.with_suffix(".txt")
                    target_caption = final_out.with_suffix(".txt")
                    if source_caption.exists() and source_caption.is_file() and not target_caption.exists():
                        target_caption.write_text(source_caption.read_text(encoding="utf-8"), encoding="utf-8")
                    processed += 1
                    self.log(
                        f"[Dataset Editor] {preset['label']}: '{normalized_audio_path.name}' -> '{final_out.name}'"
                    )
                except OSError as exc:
                    errors += 1
                    self.log(
                        f"[Dataset Editor] {preset['label']} move failed for '{normalized_audio_path.name}': {exc}"
                    )

            self.log(
                f"[Dataset Editor] {preset['label']} complete for '{dataset_name}': processed={processed}, errors={errors}"
            )
            return (processed, errors)

        def _start_audio_enhancement_preset(audio_file_paths: list[Path], preset_key: str) -> None:
            if not audio_file_paths:
                transcribe_status_var.set("No audio clips found.")
                return

            label_map = {
                "deepfilternet": "Denoise",
                "melband_roformer": "Remove Music",
            }
            label = label_map.get(preset_key, "Enhancement")
            _set_transcribe_busy(True, f"Applying {label}...")

            def worker() -> None:
                try:
                    processed_count, errors = _run_audio_enhancement_preset(audio_file_paths, preset_key)

                    def _finish_enhance() -> None:
                        _set_transcribe_busy(False, f"{label}: {processed_count} clip(s), {errors} errors")
                        if processed_count == 0 and errors > 0:
                            if preset_key == "deepfilternet":
                                self.messagebox.showerror(
                                    label,
                                    (
                                        "Denoise failed to run in this environment.\n\n"
                                        "Current blocker: DeepFilterNet 0.5.6 expects older torchaudio APIs "
                                        "(missing module 'torchaudio.backend').\n\n"
                                        "Use Remove Music for now, or move DeepFilterNet into a separate "
                                        "audio-tools environment pinned to older torch/torchaudio."
                                    ),
                                    parent=dialog,
                                )
                            else:
                                self.messagebox.showerror(
                                    label,
                                    f"{label} did not produce any output files. Check the terminal log for details.",
                                    parent=dialog,
                                )
                        if processed_count > 0:
                            dialog.destroy()
                            self.root.after(0, lambda: self.open(dataset_name))

                    dialog.after(0, _finish_enhance)
                except Exception as exc:
                    error_text = str(exc)
                    self.log(f"[Dataset Editor] {label} failed for '{dataset_name}': {error_text}")
                    dialog.after(0, lambda: _set_transcribe_busy(False, f"{label} failed."))
                    dialog.after(0, lambda: self.messagebox.showerror(label, error_text, parent=dialog))

            threading.Thread(target=worker, name="dataset-editor-audio-enhance", daemon=True).start()

        def _start_auto_split_audio(
            audio_file_paths: list[Path],
            split_params: tuple[float, float, float, float] | None = None,
        ) -> None:
            if not audio_file_paths:
                transcribe_status_var.set("No audio clips found.")
                return

            selected_params = split_params or _ask_auto_split_params()
            if selected_params is None:
                return

            split_min, split_max, silence_trim, discard_under = selected_params

            _set_transcribe_busy(True, "Auto-splitting audio...")

            def worker() -> None:
                try:
                    created, skipped_count, errors, split_sources = _run_auto_split_audio(
                        audio_file_paths,
                        split_min,
                        split_max,
                        silence_trim,
                        discard_under,
                    )

                    def _finish_split() -> None:
                        _set_transcribe_busy(
                            False,
                            f"Split done: {created} chunks, {skipped_count} skipped, {errors} errors",
                        )
                        if len(audio_file_paths) == 1 and created > 0 and split_sources:
                            source_clip = split_sources[0]
                            remove_original = self.messagebox.askyesno(
                                "Remove original clip?",
                                (
                                    f"Split complete for '{source_clip.name}'.\n\n"
                                    "Remove the original clip now?\n"
                                    "(Generated chunks and captions are already saved.)"
                                ),
                                parent=dialog,
                            )
                            if remove_original:
                                _delete_media_item(source_clip, ask_confirmation=False)

                    dialog.after(0, _finish_split)
                except ImportError:
                    dialog.after(0, lambda: _set_transcribe_busy(False, "Whisper package is not installed."))
                    dialog.after(
                        0,
                        lambda: self.messagebox.showerror(
                            "Auto-Split Audio",
                            "Whisper is not installed in this environment.\nInstall package 'openai-whisper' to enable auto-split.",
                            parent=dialog,
                        ),
                    )
                except Exception as exc:
                    error_text = str(exc)
                    self.log(f"[Dataset Editor] Auto-split failed for '{dataset_name}': {error_text}")
                    dialog.after(0, lambda: _set_transcribe_busy(False, "Auto-split failed."))
                    dialog.after(0, lambda: self.messagebox.showerror("Auto-Split Audio", error_text, parent=dialog))

            threading.Thread(target=worker, name="dataset-editor-auto-split", daemon=True).start()

        def on_editor_inner_configure(_event: Any) -> None:
            editor_canvas.configure(scrollregion=editor_canvas.bbox("all"))

        def on_editor_canvas_configure(event: Any) -> None:
            editor_canvas.itemconfigure(editor_inner_id, width=event.width)

        def editor_canvas_has_overflow() -> bool:
            bbox = editor_canvas.bbox("all")
            if not bbox:
                return False
            content_height = max(0, int(bbox[3] - bbox[1]))
            viewport_height = max(0, int(editor_canvas.winfo_height()))
            return content_height > (viewport_height + 2)

        def on_editor_mousewheel(event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            delta = int(-event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            editor_canvas.yview_scroll(delta, "units")
            return "break"

        def on_editor_linux_up(_event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            editor_canvas.yview_scroll(-1, "units")
            return "break"

        def on_editor_linux_down(_event: Any) -> str:
            if not editor_canvas_has_overflow():
                return "break"
            editor_canvas.yview_scroll(1, "units")
            return "break"

        def on_caption_mousewheel(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                delta = int(-event.delta / 120)
                if delta == 0:
                    delta = -1 if event.delta > 0 else 1
                widget.yview_scroll(delta, "units")
                return "break"
            return on_editor_mousewheel(event)

        def on_caption_linux_up(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                widget.yview_scroll(-1, "units")
                return "break"
            return on_editor_linux_up(event)

        def on_caption_linux_down(event: Any) -> str:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                widget.yview_scroll(1, "units")
                return "break"
            return on_editor_linux_down(event)

        def attach_autohide_scrollbar(text_widget: Any, scrollbar_widget: Any) -> None:
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

        def flush_caption_save(widget: Any) -> None:
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
                self.log(f"[Dataset Editor] Failed to save caption '{caption_path.name}': {exc}")

        def schedule_caption_save(event: Any) -> None:
            widget = event.widget
            if not isinstance(widget, self.tk.Text):
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

        def on_caption_focus_out(event: Any) -> None:
            widget = event.widget
            if isinstance(widget, self.tk.Text):
                flush_caption_save(widget)

        def open_media_external(path: Path) -> None:
            try:
                if os.name == "nt":
                    os.startfile(str(path))
                elif os.uname().sysname == "Darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            except Exception as exc:
                self.messagebox.showerror("Play media", f"Could not open media file:\n{exc}", parent=dialog)

        def stop_media(path: Path) -> None:
            process = active_players.pop(path, None)
            if process is None:
                return
            try:
                if process.poll() is None:
                    process.terminate()
            except Exception:
                pass

        def stop_other_media(except_path: Path | None = None) -> None:
            for media_path, process in list(active_players.items()):
                if except_path is not None and media_path == except_path:
                    continue
                try:
                    if process.poll() is None:
                        process.terminate()
                except Exception:
                    pass
                finally:
                    active_players.pop(media_path, None)

        def play_media(path: Path, media_kind: str) -> None:
            existing = active_players.get(path)
            if existing is not None and existing.poll() is None:
                if media_kind == "audio":
                    stop_media(path)
                return

            stop_other_media()

            if ffplay_path is not None:
                cmd = [ffplay_path, "-autoexit", "-loglevel", "quiet"]
                if media_kind == "audio":
                    cmd.append("-nodisp")
                cmd.append(str(path))
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    active_players[path] = process
                    return
                except Exception:
                    pass

            open_media_external(path)

        def _format_seconds_label(total_seconds: float) -> str:
            value = max(0.0, float(total_seconds))
            minutes = int(value // 60)
            seconds = value - (minutes * 60)
            return f"{minutes:02d}:{seconds:05.2f}"

        def _preview_audio_segment(audio_path: Path, start_s: float, end_s: float) -> bool:
            normalized_audio_path = _normalize_media_path(audio_path)
            if ffplay_path is None:
                self.messagebox.showerror(
                    "Trim Audio",
                    "ffplay was not found in PATH. Install ffmpeg (including ffplay) to preview segments.",
                    parent=dialog,
                )
                return False

            segment_len = max(0.1, end_s - start_s)
            stop_other_media()
            cmd = [
                ffplay_path,
                "-autoexit",
                "-loglevel",
                "quiet",
                "-nodisp",
                "-ss",
                f"{start_s:.3f}",
                "-t",
                f"{segment_len:.3f}",
                str(normalized_audio_path),
            ]
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                active_players[normalized_audio_path] = process
                return True
            except Exception as exc:
                self.messagebox.showerror("Trim Audio", f"Could not preview segment:\n{exc}", parent=dialog)
                return False

        def _open_trim_audio_dialog(audio_path: Path) -> None:
            normalized_audio_path = _normalize_media_path(audio_path)
            if not normalized_audio_path.exists():
                self.messagebox.showerror(
                    "Trim Audio",
                    f"Audio file not found:\n{normalized_audio_path}",
                    parent=dialog,
                )
                return

            duration_s = _audio_duration_seconds(normalized_audio_path)
            if duration_s is None or duration_s <= 0:
                self.messagebox.showerror(
                    "Trim Audio",
                    "Could not determine clip duration. Ensure ffprobe is available and the file is valid.",
                    parent=dialog,
                )
                return

            ffmpeg_executable = _resolve_ffmpeg_executable()
            session_temp_root = _workspace_root_for_downloads() / "Temp" / "trim_sessions"
            session_temp_root.mkdir(parents=True, exist_ok=True)
            session_temp_dir = Path(tempfile.mkdtemp(prefix="musubi_trim_", dir=str(session_temp_root)))
            working_audio_path = session_temp_dir / f"{normalized_audio_path.stem}__edit_tmp.wav"
            temp_caption_path = working_audio_path.with_suffix(".txt")

            make_temp_cmd = [
                ffmpeg_executable,
                "-y",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(normalized_audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-acodec",
                "pcm_s16le",
                str(working_audio_path),
            ]
            make_temp_result = subprocess.run(make_temp_cmd, capture_output=True, text=True, check=False)
            if make_temp_result.returncode != 0 or (not working_audio_path.exists()):
                stderr_text = (make_temp_result.stderr or "").strip()
                try:
                    shutil.rmtree(session_temp_dir, ignore_errors=True)
                except Exception:
                    pass
                self.messagebox.showerror(
                    "Trim Audio",
                    stderr_text or "Could not create a temporary working clip.",
                    parent=dialog,
                )
                return

            duration_s = _audio_duration_seconds(working_audio_path)
            if duration_s is None or duration_s <= 0:
                self.messagebox.showerror(
                    "Trim Audio",
                    "Could not determine temporary clip duration.",
                    parent=dialog,
                )
                try:
                    shutil.rmtree(session_temp_dir, ignore_errors=True)
                except Exception:
                    pass
                return

            trim_prompt = self.tk.Toplevel(dialog)
            trim_prompt.withdraw()
            trim_prompt.title(f"Trim Audio: {normalized_audio_path.name}")
            trim_prompt.transient(dialog)
            trim_prompt.grab_set()
            trim_prompt.configure(bg=self.bg_panel)
            trim_prompt.resizable(False, False)
            self.set_dark_title_bar(trim_prompt)

            root_frame = self.ttk.Frame(trim_prompt, padding=10)
            root_frame.grid(row=0, column=0, sticky="nsew")
            root_frame.columnconfigure(0, weight=1)

            start_var = self.tk.DoubleVar(master=trim_prompt, value=0.0)
            end_var = self.tk.DoubleVar(master=trim_prompt, value=float(duration_s))
            start_input_var = self.tk.StringVar(master=trim_prompt, value=_format_seconds_label(0.0))
            end_input_var = self.tk.StringVar(master=trim_prompt, value=_format_seconds_label(duration_s))
            total_label_var = self.tk.StringVar(master=trim_prompt, value=_format_seconds_label(duration_s))
            playhead_label_var = self.tk.StringVar(master=trim_prompt, value=_format_seconds_label(0.0))
            duration_label_var = self.tk.StringVar(master=trim_prompt, value=f"Duration: {_format_seconds_label(duration_s)}")
            max_end_s = float(duration_s)

            waveform_bins_count = 320
            waveform_values: list[float] = []
            drag_state: dict[str, Any] = {"mode": None, "resume_after_seek": False}
            playhead_var = self.tk.DoubleVar(master=trim_prompt, value=0.0)
            playback_started_at = 0.0
            playback_start_offset_s = 0.0
            playback_end_offset_s = float(duration_s)
            play_pause_label_var = self.tk.StringVar(master=trim_prompt, value="Play")
            playback_poll_after_id: str | None = None

            def _extract_waveform_bins(target_path: Path, bins: int) -> list[float]:
                ffmpeg_executable = _resolve_ffmpeg_executable()
                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(target_path),
                    "-f",
                    "s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "8000",
                    "-",
                ]
                completed = subprocess.run(ffmpeg_cmd, capture_output=True, check=False)
                if completed.returncode != 0:
                    return []
                pcm_bytes = completed.stdout or b""
                if len(pcm_bytes) < 2:
                    return []

                sample_count = len(pcm_bytes) // 2
                if sample_count <= 0:
                    return []

                samples_view = memoryview(pcm_bytes[: sample_count * 2]).cast("h")
                chunk_size = max(1, sample_count // bins)
                values: list[float] = []
                index = 0
                while index < sample_count and len(values) < bins:
                    chunk = samples_view[index : min(sample_count, index + chunk_size)]
                    peak = 0
                    for sample in chunk:
                        amp = sample if sample >= 0 else -sample
                        if amp > peak:
                            peak = amp
                    values.append(min(1.0, float(peak) / 32768.0))
                    index += chunk_size

                if not values:
                    return []

                while len(values) < bins:
                    values.append(values[-1])
                return values[:bins]

            self.ttk.Label(root_frame, textvariable=duration_label_var, style="TLabel").grid(
                row=0,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(0, 8),
            )

            waveform_canvas_width = 900
            waveform_canvas_height = 140
            waveform_shell = self.tk.Frame(
                root_frame,
                bg="#0f1623",
                highlightthickness=1,
                highlightbackground="#2a3a50",
                bd=0,
            )
            waveform_shell.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
            waveform_canvas = self.tk.Canvas(
                waveform_shell,
                width=waveform_canvas_width,
                height=waveform_canvas_height,
                bg="#0f1623",
                highlightthickness=0,
                bd=0,
            )
            waveform_canvas.pack(fill="x", expand=True)

            values_trim_row = self.ttk.Frame(root_frame)
            values_trim_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            values_trim_row.columnconfigure(8, weight=1)
            self.ttk.Label(values_trim_row, text="In:", style="TLabel").grid(row=0, column=0, sticky="w")
            in_entry = self.ttk.Entry(values_trim_row, textvariable=start_input_var, width=10)
            in_entry.grid(row=0, column=1, sticky="w", padx=(6, 14))
            self.ttk.Label(values_trim_row, text="Out:", style="TLabel").grid(row=0, column=2, sticky="w")
            out_entry = self.ttk.Entry(values_trim_row, textvariable=end_input_var, width=10)
            out_entry.grid(row=0, column=3, sticky="w", padx=(6, 14))
            self.ttk.Label(values_trim_row, text="Total:", style="TLabel").grid(row=0, column=4, sticky="w")
            self.ttk.Label(values_trim_row, textvariable=total_label_var, style="TLabel", width=10).grid(
                row=0,
                column=5,
                sticky="w",
                padx=(6, 0),
            )
            self.ttk.Label(values_trim_row, text="Playhead:", style="TLabel").grid(row=0, column=6, sticky="w", padx=(14, 0))
            self.ttk.Label(values_trim_row, textvariable=playhead_label_var, style="TLabel", width=10).grid(
                row=0,
                column=7,
                sticky="w",
                padx=(6, 0),
            )

            def _parse_time_input(value: str) -> float | None:
                raw = value.strip()
                if not raw:
                    return None
                try:
                    if ":" not in raw:
                        parsed = float(raw)
                        return parsed if parsed >= 0.0 else None

                    parts = [segment.strip() for segment in raw.split(":")]
                    if len(parts) not in (2, 3):
                        return None
                    if any((not segment) for segment in parts):
                        return None

                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        if minutes < 0 or seconds < 0:
                            return None
                        return (minutes * 60.0) + seconds

                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    if hours < 0 or minutes < 0 or seconds < 0:
                        return None
                    return (hours * 3600.0) + (minutes * 60.0) + seconds
                except ValueError:
                    return None

            def _draw_waveform_overlay(start_s: float, end_s: float, playhead_s: float) -> None:
                waveform_canvas.delete("all")
                width = waveform_canvas_width
                height = waveform_canvas_height
                mid_y = height // 2
                handle_w = 12.0
                handle_h = 10.0

                if not waveform_values:
                    waveform_canvas.create_text(
                        width // 2,
                        mid_y,
                        text="Waveform preview unavailable",
                        fill="#8aa6c8",
                        font=("Segoe UI", 9),
                    )
                else:
                    bar_count = len(waveform_values)
                    bar_w = max(1.0, float(width) / float(bar_count))
                    for idx, amp in enumerate(waveform_values):
                        x0 = idx * bar_w
                        x1 = x0 + max(1.0, bar_w - 0.6)
                        half_h = max(1.0, amp * (height * 0.45))
                        waveform_canvas.create_rectangle(
                            x0,
                            mid_y - half_h,
                            x1,
                            mid_y + half_h,
                            fill="#4ea8de",
                            outline="",
                        )

                start_x = (start_s / max_end_s) * width if max_end_s > 0 else 0.0
                end_x = (end_s / max_end_s) * width if max_end_s > 0 else float(width)
                playhead_x = (playhead_s / max_end_s) * width if max_end_s > 0 else 0.0
                start_x = max(0.0, min(float(width), start_x))
                end_x = max(0.0, min(float(width), end_x))
                playhead_x = max(0.0, min(float(width), playhead_x))

                if start_x > 0:
                    waveform_canvas.create_rectangle(0, 0, start_x, height, fill="#0b1018", outline="", stipple="gray50")
                if end_x < width:
                    waveform_canvas.create_rectangle(end_x, 0, width, height, fill="#0b1018", outline="", stipple="gray50")

                waveform_canvas.create_line(start_x, 0, start_x, height, fill="#ffd166", width=2)
                waveform_canvas.create_line(end_x, 0, end_x, height, fill="#ef476f", width=2)
                waveform_canvas.create_line(playhead_x, 0, playhead_x, height, fill="#f5f9ff", width=1)

                waveform_canvas.create_rectangle(
                    start_x - (handle_w / 2.0),
                    0,
                    start_x + (handle_w / 2.0),
                    handle_h,
                    fill="#1e2430",
                    outline="#ffd166",
                    width=2,
                )
                waveform_canvas.create_rectangle(
                    end_x - (handle_w / 2.0),
                    0,
                    end_x + (handle_w / 2.0),
                    handle_h,
                    fill="#1e2430",
                    outline="#ef476f",
                    width=2,
                )

            def _update_playhead_visual() -> None:
                start_s = float(start_var.get())
                end_s = float(end_var.get())
                playhead_s = max(0.0, min(float(playhead_var.get()), max_end_s))
                playhead_var.set(playhead_s)
                _draw_waveform_overlay(start_s, end_s, playhead_s)

            def _set_playhead_seconds(value: float) -> None:
                playhead_var.set(max(0.0, min(float(value), max_end_s)))
                playhead_label_var.set(_format_seconds_label(float(playhead_var.get())))
                _update_playhead_visual()

            def _clamp_points(changed: str) -> tuple[float, float]:
                start_s = max(0.0, min(float(start_var.get()), max_end_s))
                end_s = max(0.0, min(float(end_var.get()), max_end_s))
                minimum_gap = 0.1

                if changed == "start" and start_s > (end_s - minimum_gap):
                    end_s = min(max_end_s, start_s + minimum_gap)
                if changed == "end" and end_s < (start_s + minimum_gap):
                    start_s = max(0.0, end_s - minimum_gap)
                if end_s < (start_s + minimum_gap):
                    end_s = min(max_end_s, start_s + minimum_gap)

                start_var.set(start_s)
                end_var.set(end_s)
                current_head = max(start_s, min(float(playhead_var.get()), end_s))
                playhead_var.set(current_head)
                playhead_label_var.set(_format_seconds_label(current_head))
                start_input_var.set(_format_seconds_label(start_s))
                end_input_var.set(_format_seconds_label(end_s))
                total_label_var.set(_format_seconds_label(end_s - start_s))
                _draw_waveform_overlay(start_s, end_s, current_head)
                return (start_s, end_s)

            def _set_point_from_canvas_x(changed: str, canvas_x: float) -> None:
                width = float(waveform_canvas_width)
                clamped_x = max(0.0, min(width, float(canvas_x)))
                new_time = (clamped_x / width) * max_end_s if width > 0 else 0.0
                if changed == "start":
                    start_var.set(new_time)
                    _clamp_points("start")
                else:
                    end_var.set(new_time)
                    _clamp_points("end")

            def _is_trim_preview_playing() -> bool:
                process = active_players.get(working_audio_path)
                return process is not None and process.poll() is None

            def _refresh_play_pause_label() -> None:
                play_pause_label_var.set("⏸" if _is_trim_preview_playing() else "▶")

            def _start_playback_from(play_from_s: float) -> None:
                nonlocal playback_started_at, playback_start_offset_s, playback_end_offset_s
                clip_end_s = max_end_s
                play_from = max(0.0, min(float(play_from_s), clip_end_s))
                if play_from >= clip_end_s:
                    play_from = max(0.0, clip_end_s - 0.05)
                ok = _preview_audio_segment(working_audio_path, play_from, clip_end_s)
                if ok:
                    playback_start_offset_s = play_from
                    playback_end_offset_s = clip_end_s
                    playback_started_at = time.monotonic()
                    _set_playhead_seconds(play_from)
                _refresh_play_pause_label()

            def _resume_after_seek_if_needed() -> None:
                if bool(drag_state.get("resume_after_seek", False)):
                    _start_playback_from(float(playhead_var.get()))
                else:
                    _refresh_play_pause_label()

            def _on_waveform_press(event: Any) -> None:
                start_s = float(start_var.get())
                end_s = float(end_var.get())
                width = float(waveform_canvas_width)
                handle_w = 12.0
                handle_h = 10.0
                start_x = (start_s / max_end_s) * width if max_end_s > 0 else 0.0
                end_x = (end_s / max_end_s) * width if max_end_s > 0 else width
                click_x = float(event.x)
                click_y = float(event.y)

                drag_state["mode"] = None
                drag_state["resume_after_seek"] = False

                if click_y <= handle_h and abs(click_x - start_x) <= (handle_w / 2.0 + 2.0):
                    drag_state["mode"] = "start"
                    _set_point_from_canvas_x("start", click_x)
                    return

                if click_y <= handle_h and abs(click_x - end_x) <= (handle_w / 2.0 + 2.0):
                    drag_state["mode"] = "end"
                    _set_point_from_canvas_x("end", click_x)
                    return

                drag_state["mode"] = "playhead"
                resume_after_seek = _is_trim_preview_playing()
                drag_state["resume_after_seek"] = resume_after_seek
                if resume_after_seek:
                    stop_media(working_audio_path)

                clamped_x = max(0.0, min(width, click_x))
                new_time = (clamped_x / width) * max_end_s if width > 0 else 0.0
                _set_playhead_seconds(new_time)

            def _on_waveform_drag(event: Any) -> None:
                active_mode = drag_state.get("mode")
                if active_mode in {"start", "end"}:
                    _set_point_from_canvas_x(str(active_mode), float(event.x))
                    return

                if active_mode == "playhead":
                    width = float(waveform_canvas_width)
                    clamped_x = max(0.0, min(width, float(event.x)))
                    new_time = (clamped_x / width) * max_end_s if width > 0 else 0.0
                    _set_playhead_seconds(new_time)

            def _on_waveform_release(_event: Any) -> None:
                if drag_state.get("mode") == "playhead":
                    _resume_after_seek_if_needed()
                drag_state["mode"] = None
                drag_state["resume_after_seek"] = False

            def _apply_time_entry(changed: str) -> None:
                raw_value = start_input_var.get() if changed == "start" else end_input_var.get()
                parsed = _parse_time_input(raw_value)
                if parsed is None:
                    self.messagebox.showerror(
                        "Trim Audio",
                        "Invalid time value. Use seconds (e.g. 12.5), mm:ss, or hh:mm:ss.",
                        parent=trim_prompt,
                    )
                    _clamp_points(changed)
                    return
                if changed == "start":
                    start_var.set(parsed)
                else:
                    end_var.set(parsed)
                _clamp_points(changed)

            waveform_canvas.bind("<Button-1>", _on_waveform_press)
            waveform_canvas.bind("<B1-Motion>", _on_waveform_drag)
            waveform_canvas.bind("<ButtonRelease-1>", _on_waveform_release)
            in_entry.bind("<Return>", lambda _event: _apply_time_entry("start"))
            in_entry.bind("<KP_Enter>", lambda _event: _apply_time_entry("start"))
            out_entry.bind("<Return>", lambda _event: _apply_time_entry("end"))
            out_entry.bind("<KP_Enter>", lambda _event: _apply_time_entry("end"))
            _clamp_points("start")

            def _poll_playback_state() -> None:
                nonlocal playback_poll_after_id
                if not trim_prompt.winfo_exists():
                    playback_poll_after_id = None
                    return
                if _is_trim_preview_playing():
                    elapsed_s = max(0.0, time.monotonic() - playback_started_at)
                    current_s = min(playback_end_offset_s, playback_start_offset_s + elapsed_s)
                    _set_playhead_seconds(current_s)
                _refresh_play_pause_label()
                playback_poll_after_id = trim_prompt.after(180, _poll_playback_state)

            def _toggle_play_pause() -> None:
                if _is_trim_preview_playing():
                    stop_media(working_audio_path)
                    _refresh_play_pause_label()
                    return
                current_head = float(playhead_var.get())
                if current_head < 0.0 or current_head >= max_end_s:
                    current_head = 0.0
                    _set_playhead_seconds(current_head)
                _start_playback_from(current_head)

            def _refresh_working_clip_state() -> None:
                nonlocal waveform_values, max_end_s, playback_end_offset_s
                updated_duration = _audio_duration_seconds(working_audio_path)
                if updated_duration is None or updated_duration <= 0:
                    raise RuntimeError("Could not determine edited clip duration.")
                max_end_s = float(updated_duration)
                duration_label_var.set(f"Duration: {_format_seconds_label(max_end_s)}")
                playback_end_offset_s = max_end_s
                start_var.set(0.0)
                end_var.set(max_end_s)
                playhead_var.set(0.0)
                try:
                    waveform_values = _extract_waveform_bins(working_audio_path, waveform_bins_count)
                except Exception:
                    waveform_values = []
                _clamp_points("start")
                _refresh_play_pause_label()

            def _apply_trim_to_working() -> None:
                start_s, end_s = _clamp_points("start")
                segment_len = end_s - start_s
                if segment_len <= 0.0:
                    self.messagebox.showerror("Trim Audio", "Trim length must be greater than 0.", parent=trim_prompt)
                    return

                stop_media(working_audio_path)
                temp_out = working_audio_path.with_name(f"{working_audio_path.stem}__trim_apply.wav")
                ffmpeg_cmd = [
                    ffmpeg_executable,
                    "-y",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_s:.3f}",
                    "-to",
                    f"{end_s:.3f}",
                    "-i",
                    str(working_audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-acodec",
                    "pcm_s16le",
                    str(temp_out),
                ]
                ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                if ffmpeg_result.returncode != 0:
                    stderr_text = (ffmpeg_result.stderr or "").strip()
                    self.messagebox.showerror(
                        "Trim Audio",
                        stderr_text or "ffmpeg trim command failed.",
                        parent=trim_prompt,
                    )
                    return

                try:
                    if working_audio_path.exists():
                        working_audio_path.unlink()
                    shutil.move(str(temp_out), str(working_audio_path))
                    _refresh_working_clip_state()
                except OSError as exc:
                    self.messagebox.showerror("Trim Audio", f"Could not finalize trimmed clip:\n{exc}", parent=trim_prompt)
                    return

            def _adopt_generated_variant(suffix: str) -> bool:
                variants: list[Path] = []
                stem_prefix = f"{working_audio_path.stem}_{suffix}"
                for ext in ("wav", "flac"):
                    for path in working_audio_path.parent.iterdir():
                        if not path.is_file():
                            continue
                        if path.suffix.lower() != f".{ext}":
                            continue
                        if path.name.startswith(stem_prefix):
                            variants.append(path)

                # Fallback for tools that encode model/tag names rather than our suffix.
                if not variants and suffix == "mbr":
                    for ext in ("wav", "flac"):
                        for path in working_audio_path.parent.iterdir():
                            if not path.is_file():
                                continue
                            if path.suffix.lower() != f".{ext}":
                                continue
                            name_lower = path.name.lower()
                            if path.name.startswith(working_audio_path.stem) and "melband" in name_lower and "roformer" in name_lower:
                                variants.append(path)
                if not variants:
                    self.log(
                        f"[Dataset Editor] Could not find generated variant for suffix '{suffix}' in '{working_audio_path.parent}'"
                    )
                    return False
                variants.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                chosen = variants[0]

                last_error: OSError | None = None
                for _attempt in range(8):
                    try:
                        stop_media(working_audio_path)
                        if working_audio_path.exists():
                            working_audio_path.unlink()
                        shutil.move(str(chosen), str(working_audio_path))
                        for stale in variants[1:]:
                            try:
                                stale.unlink()
                            except OSError:
                                pass
                        variant_caption = chosen.with_suffix(".txt")
                        if variant_caption.exists():
                            try:
                                variant_caption.unlink()
                            except OSError:
                                pass
                        return True
                    except OSError as exc:
                        last_error = exc
                        time.sleep(0.08)

                if last_error is not None:
                    self.log(
                        f"[Dataset Editor] Failed to adopt generated variant '{chosen.name}': {last_error}"
                    )
                return False

            def _normalize_current_clip() -> None:
                processed_count, errors = _run_normalize_audio_gain([working_audio_path])
                if processed_count <= 0:
                    self.messagebox.showerror("Normalize", f"Normalize failed ({errors} error(s)).", parent=trim_prompt)
                    return
                if not _adopt_generated_variant("norm"):
                    self.messagebox.showerror("Normalize", "Normalized output was not found.", parent=trim_prompt)
                    return
                _refresh_working_clip_state()

            def _enhance_current_clip(preset_key: str) -> None:
                label = "Denoise" if preset_key == "deepfilternet" else "Remove Music"
                processed_count, errors = _run_audio_enhancement_preset([working_audio_path], preset_key)
                if processed_count <= 0:
                    if preset_key == "deepfilternet":
                        self.messagebox.showerror(
                            label,
                            "Denoise failed in this environment (torchaudio.backend mismatch).",
                            parent=trim_prompt,
                        )
                    else:
                        self.messagebox.showerror(label, f"{label} failed ({errors} error(s)).", parent=trim_prompt)
                    return
                expected_suffix = "dfn" if preset_key == "deepfilternet" else "mbr"
                if not _adopt_generated_variant(expected_suffix):
                    self.messagebox.showerror(label, f"{label} output was not found.", parent=trim_prompt)
                    return
                _refresh_working_clip_state()

            def _cancel_trim_prompt() -> None:
                nonlocal playback_poll_after_id
                stop_media(working_audio_path)
                if playback_poll_after_id is not None:
                    try:
                        trim_prompt.after_cancel(playback_poll_after_id)
                    except Exception:
                        pass
                    playback_poll_after_id = None
                for stale_path in (working_audio_path, temp_caption_path):
                    try:
                        if stale_path.exists():
                            stale_path.unlink()
                    except OSError:
                        pass
                for stale in working_audio_path.parent.iterdir():
                    if not stale.is_file():
                        continue
                    if stale.suffix.lower() != ".wav":
                        continue
                    if stale.name.startswith(f"{working_audio_path.stem}_"):
                        try:
                            stale.unlink()
                        except OSError:
                            pass
                try:
                    shutil.rmtree(session_temp_dir, ignore_errors=True)
                except Exception:
                    pass
                trim_prompt.destroy()

            def _commit_trim_prompt() -> None:
                try:
                    stop_media(working_audio_path)
                    if normalized_audio_path.exists():
                        normalized_audio_path.unlink()
                    shutil.move(str(working_audio_path), str(normalized_audio_path))
                    if temp_caption_path.exists():
                        temp_caption_path.unlink()
                except OSError as exc:
                    self.messagebox.showerror("Trim Audio", f"Could not commit edited clip:\n{exc}", parent=trim_prompt)
                    return
                try:
                    shutil.rmtree(session_temp_dir, ignore_errors=True)
                except Exception:
                    pass
                trim_prompt.destroy()
                dialog.destroy()
                self.root.after(0, lambda: self.open(dataset_name))

            trim_holder = self.tk.Frame(values_trim_row, bg=self.bg_panel, bd=0)
            trim_holder.grid(row=0, column=9, sticky="e")
            trim_holder.pack_propagate(False)
            trim_button = self.ttk.Button(trim_holder, text="Trim", command=_apply_trim_to_working)
            trim_button.pack(fill="x", expand=True)

            actions_row = self.ttk.Frame(root_frame)
            actions_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            actions_row.columnconfigure(1, weight=1)

            play_group = self.ttk.Frame(actions_row)
            play_group.grid(row=0, column=0, sticky="w")
            self.ttk.Button(play_group, textvariable=play_pause_label_var, command=_toggle_play_pause, width=4).pack(side="left")

            tools_row = self.ttk.Frame(actions_row)
            tools_row.grid(row=0, column=1)
            self.ttk.Button(tools_row, text="Normalize", command=_normalize_current_clip).grid(row=0, column=0, padx=(0, 6))
            self.ttk.Button(tools_row, text="Denoise", command=lambda: _enhance_current_clip("deepfilternet")).grid(
                row=0,
                column=1,
                padx=(0, 6),
            )
            self.ttk.Button(
                tools_row,
                text="Remove Music",
                command=lambda: _enhance_current_clip("melband_roformer"),
            ).grid(row=0, column=2, padx=(0, 6))

            right_actions = self.ttk.Frame(actions_row)
            right_actions.grid(row=0, column=2, sticky="e")
            self.ttk.Button(right_actions, text="OK", command=_commit_trim_prompt).pack(side="left", padx=(0, 6))
            self.ttk.Button(right_actions, text="Cancel", command=_cancel_trim_prompt).pack(side="left")

            def _sync_trim_button_width() -> None:
                if not trim_prompt.winfo_exists():
                    return
                try:
                    right_actions.update_idletasks()
                    target_width = max(80, int(right_actions.winfo_reqwidth()))
                    target_height = max(26, int(right_actions.winfo_reqheight()))
                    trim_holder.configure(width=target_width, height=target_height)
                except Exception:
                    pass

            trim_prompt.after_idle(_sync_trim_button_width)

            trim_prompt.protocol("WM_DELETE_WINDOW", _cancel_trim_prompt)
            _refresh_play_pause_label()
            _poll_playback_state()
            self.center_window(trim_prompt)
            trim_prompt.deiconify()

            def _load_waveform_preview() -> None:
                nonlocal waveform_values
                if not trim_prompt.winfo_exists():
                    return
                try:
                    waveform_values = _extract_waveform_bins(working_audio_path, waveform_bins_count)
                except Exception:
                    waveform_values = []
                _clamp_points("start")

            trim_prompt.after(10, _load_waveform_preview)
            dialog.wait_window(trim_prompt)

        def _show_media_context_menu(event: Any, media_path: Path, media_kind: str) -> str:
            menu = self.tk.Menu(dialog, tearoff=0)
            menu.add_command(label="Open", command=lambda p=media_path: open_media_external(p))
            if media_kind == "audio":
                menu.add_separator()
                menu.add_command(
                    label="Transcribe",
                    command=lambda p=media_path: _start_whisper_transcription([p], bool(replace_existing_var.get())),
                )
                menu.add_command(
                    label="Normalize Gain",
                    command=lambda p=media_path: _start_normalize_audio_gain([p]),
                )
                menu.add_command(
                    label="Trim...",
                    command=lambda p=media_path: _open_trim_audio_dialog(p),
                )
                menu.add_command(
                    label="Auto-Split...",
                    command=lambda p=media_path: _start_auto_split_audio([p]),
                )
            elif media_kind == "video":
                menu.add_separator()
                menu.add_command(
                    label="Extract Audio",
                    command=lambda p=media_path: _start_extract_audio_from_videos([p]),
                )
            menu.add_separator()
            menu.add_command(label="Delete", command=lambda p=media_path: _delete_media_item(p, ask_confirmation=True))
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
            return "break"

        def build_media_thumb(media_kind: str, show_split_warning: bool = False) -> Any:
            thumb_size = (tile_size_px, tile_size_px)
            if self.Image is None or self.ImageTk is None:
                photo = self.tk.PhotoImage(master=dialog, width=thumb_size[0], height=thumb_size[1])
                photo.put("#2a2a2a", to=(0, 0, thumb_size[0], thumb_size[1]))
                return photo

            image = self.Image.new("RGB", thumb_size, color="#1f1f1f")
            draw = self.ImageDraw.Draw(image)
            if media_kind == "audio":
                mid_y = tile_size_px // 2
                margin = max(8, tile_size_px // 14)
                bar_width = max(2, tile_size_px // 28)
                spacing = max(5, tile_size_px // 17)
                heights = [0.20, 0.38, 0.55, 0.32, 0.74, 0.46, 0.62, 0.28]
                x = margin
                idx = 0
                while x < tile_size_px - margin:
                    amp = int((tile_size_px * heights[idx % len(heights)]) * 0.5)
                    draw.rectangle(
                        (x, mid_y - amp, x + bar_width, mid_y + amp),
                        fill=(118, 204, 255),
                    )
                    x += spacing
                    idx += 1
                draw.text((margin, tile_size_px - margin - 18), "AUDIO", fill=(198, 233, 255))
                if show_split_warning:
                    draw.rectangle(
                        (0, 0, tile_size_px - 1, 28),
                        fill=(90, 15, 15),
                    )
                    draw.text((10, 8), "SPLIT AUDIO >15s", fill=(255, 200, 200))
            else:
                margin = max(16, tile_size_px // 9)
                draw.rectangle(
                    (margin, margin, tile_size_px - margin, tile_size_px - margin),
                    outline=(126, 166, 230),
                    width=2,
                )
                triangle = [
                    (tile_size_px // 2 - 18, tile_size_px // 2 - 24),
                    (tile_size_px // 2 - 18, tile_size_px // 2 + 24),
                    (tile_size_px // 2 + 26, tile_size_px // 2),
                ]
                draw.polygon(triangle, fill=(126, 166, 230))
                draw.text((margin, tile_size_px - margin - 18), "VIDEO", fill=(190, 210, 240))

            image_rgba = image.convert("RGBA")
            border_overlay = self.Image.new("RGBA", thumb_size, (0, 0, 0, 0))
            border_draw = self.ImageDraw.Draw(border_overlay)
            if media_kind == "audio" and show_split_warning:
                border_draw.rectangle((0, 0, thumb_size[0] - 1, thumb_size[1] - 1), outline=(224, 72, 72, 220), width=2)
            else:
                border_draw.rectangle((0, 0, thumb_size[0] - 1, thumb_size[1] - 1), outline=(255, 255, 255, 96), width=1)
            composited = self.Image.alpha_composite(image_rgba, border_overlay).convert("RGB")
            return self.ImageTk.PhotoImage(composited, master=self.root)

        for column_index in range(columns):
            editor_inner.columnconfigure(column_index, weight=0, minsize=tile_size_px)

        for idx, (media_path, media_kind) in enumerate(media_items):
            caption_path = media_path.with_suffix(".txt")
            caption_text = ""
            if caption_path.exists() and caption_path.is_file():
                try:
                    caption_text = caption_path.read_text(encoding="utf-8")
                except OSError:
                    caption_text = ""

            item_frame = self.ttk.Frame(editor_inner, padding=(4, 4, 4, 6), style="TFrame")
            item_frame.grid(row=idx // columns, column=idx % columns, sticky="n", padx=tile_side_pad_px, pady=4)
            item_frame.columnconfigure(0, weight=1)

            long_audio_warning = media_kind == "audio" and _is_long_audio_clip(media_path)
            photo = build_media_thumb(media_kind, show_split_warning=long_audio_warning)
            thumb_refs.append(photo)
            image_label = self.ttk.Label(item_frame, image=photo, anchor="center")
            image_label.grid(row=0, column=0, sticky="n")
            if media_kind == "audio":
                image_label.bind("<Double-Button-1>", lambda _event, p=media_path: _open_trim_audio_dialog(p))
            else:
                image_label.bind("<Double-Button-1>", lambda _event, p=media_path, k=media_kind: play_media(p, k))
            image_label.bind("<MouseWheel>", on_editor_mousewheel)
            image_label.bind("<Button-4>", on_editor_linux_up)
            image_label.bind("<Button-5>", on_editor_linux_down)

            play_badge = self.tk.Button(
                item_frame,
                text="▶",
                command=lambda p=media_path, k=media_kind: play_media(p, k),
                bg="#121a27",
                fg="#d8ecff",
                activebackground="#1f2b3f",
                activeforeground="#ffffff",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI Symbol", 12, "bold"),
                padx=10,
                pady=4,
                cursor="hand2",
            )
            play_badge.place(in_=image_label, relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")

            name_label = self.ttk.Label(
                item_frame,
                text=media_path.name,
                style="CardMeta.TLabel",
                anchor="center",
                justify="center",
                wraplength=tile_size_px,
            )
            name_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))

            caption_shell = self.tk.Frame(
                item_frame,
                width=tile_size_px,
                height=64,
                bg="#111826",
                highlightthickness=1,
                highlightbackground="#2a3a50",
                highlightcolor="#4a6ea3",
                bd=0,
            )
            caption_shell.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            caption_shell.grid_propagate(False)

            caption_widget = self.tk.Text(
                caption_shell,
                width=1,
                height=3,
                wrap="word",
                bg="#111826",
                fg=self.fg_text,
                insertbackground=self.fg_text,
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                padx=6,
                pady=5,
            )
            caption_widget.pack(side="left", fill="both", expand=True)

            caption_scroll = self.ttk.Scrollbar(
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
            caption_widget.bind("<MouseWheel>", on_caption_mousewheel)
            caption_widget.bind("<Button-4>", on_caption_linux_up)
            caption_widget.bind("<Button-5>", on_caption_linux_down)
            caption_path_by_widget[caption_widget] = caption_path

            image_label.bind("<Button-3>", lambda event, p=media_path, k=media_kind: _show_media_context_menu(event, p, k))
            name_label.bind("<Button-3>", lambda event, p=media_path, k=media_kind: _show_media_context_menu(event, p, k))

            caption_widget_by_path[caption_path] = caption_widget

        transcribe_all_button.configure(
            command=lambda: _start_whisper_transcription(audio_paths, bool(replace_existing_var.get()))
        )
        extract_audio_button.configure(
            command=lambda: _start_extract_audio_from_videos(video_paths)
        )
        normalize_audio_button.configure(
            command=lambda: _start_normalize_audio_gain(audio_paths)
        )

        def close_editor() -> None:
            for text_widget in list(caption_path_by_widget.keys()):
                flush_caption_save(text_widget)
            for process in list(active_players.values()):
                try:
                    if process.poll() is None:
                        process.terminate()
                except Exception:
                    pass
            dialog.destroy()

        footer = self.tk.Frame(outer, bg=self.bg_panel, bd=0)
        footer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        footer_text_color = getattr(self, "fg_muted", self.fg_text)
        self.tk.Label(
            footer,
            text="Double-click audio tiles to open trim/tools. Use > on each tile for quick play/pause. Right-click for more actions.",
            bg=self.bg_panel,
            fg=footer_text_color,
            anchor="w",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Button(footer, text="Close", command=close_editor).grid(row=0, column=1, sticky="e")

        editor_inner.bind("<Configure>", on_editor_inner_configure)
        editor_canvas.bind("<Configure>", on_editor_canvas_configure)
        editor_canvas.bind("<MouseWheel>", on_editor_mousewheel)
        editor_canvas.bind("<Button-4>", on_editor_linux_up)
        editor_canvas.bind("<Button-5>", on_editor_linux_down)
        editor_inner.bind("<MouseWheel>", on_editor_mousewheel)
        editor_inner.bind("<Button-4>", on_editor_linux_up)
        editor_inner.bind("<Button-5>", on_editor_linux_down)
        dialog.protocol("WM_DELETE_WINDOW", close_editor)

        self.center_window(dialog)
        dialog.deiconify()
        self.root.wait_window(dialog)
