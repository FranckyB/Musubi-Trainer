from __future__ import annotations

import os
import re
import shutil
import subprocess
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
        controls.grid(row=1, column=0, sticky="w", pady=(0, 8))
        autotag_all_var = self.tk.BooleanVar(master=dialog, value=False)
        autotag_status_var = self.tk.StringVar(master=dialog, value="")
        trigger_word_var = self.tk.StringVar(master=dialog, value="")
        caption_mode_choices = ["simple", "detailed", "extra", "mixed", "extra_mixed", "analyze"]
        caption_mode_var = self.tk.StringVar(master=dialog, value="simple")
        autotag_button: Any = None
        caption_button: Any = None
        detailed_button: Any = None

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
            for action_button in (caption_button, autotag_button, detailed_button):
                if action_button is not None:
                    try:
                        action_button.configure(state=button_state)
                    except Exception:
                        pass
            autotag_status_var.set(status_text)

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
            width=24,
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
        self.ttk.Label(controls, textvariable=autotag_status_var, style="TLabel").grid(row=0, column=8, sticky="w", padx=(12, 0))

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
            text="Captions auto-save as .txt sidecar files.",
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

        self.ttk.Label(controls, textvariable=transcribe_status_var, style="TLabel").grid(
            row=0, column=6, sticky="w", padx=(10, 0)
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

        def _set_transcribe_busy(is_busy: bool, status_text: str = "") -> None:
            button_state = "disabled" if is_busy else "normal"
            if transcribe_all_button is not None:
                try:
                    transcribe_all_button.configure(state=button_state)
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

                segments = final_segments

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
                    label="Auto-Split...",
                    command=lambda p=media_path: _start_auto_split_audio([p]),
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
            image_label.bind("<Double-Button-1>", lambda _event, p=media_path, k=media_kind: play_media(p, k))
            image_label.bind("<MouseWheel>", on_editor_mousewheel)
            image_label.bind("<Button-4>", on_editor_linux_up)
            image_label.bind("<Button-5>", on_editor_linux_down)

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
            text="Double-click audio tiles to play or stop. Captions auto-save as .txt sidecar files.",
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
