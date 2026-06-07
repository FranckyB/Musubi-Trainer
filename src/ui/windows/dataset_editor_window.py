from __future__ import annotations

import os
from pathlib import Path
import threading
import traceback
import warnings
from typing import Any


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

        grid_host = self.ttk.Frame(outer)
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
            target_paths: list[Path] = []
            candidate_paths = [specific_image_path] if specific_image_path is not None else image_paths
            for image_path in candidate_paths:
                if image_path is None:
                    continue
                caption_path = image_path.with_suffix(".txt")
                if specific_image_path is not None or _eligible_for_autotag(caption_path, include_non_empty):
                    target_paths.append(image_path)

            if not target_paths:
                autotag_status_var.set("Nothing to autotag.")
                if specific_image_path is None:
                    self.log(f"[Dataset Editor] Autotag skipped for '{dataset_name}' (mode={mode}): nothing eligible.")
                else:
                    self.log(f"[Dataset Editor] Autotag skipped for '{dataset_name}' image '{specific_image_path.name}' (mode={mode}): nothing eligible.")
                return

            _set_autotag_busy(True, "Preparing Florence model...")
            if specific_image_path is None:
                self.log(f"[Dataset Editor] Autotag start for '{dataset_name}' (mode={mode}, all={include_non_empty})")
            else:
                self.log(f"[Dataset Editor] Autotag start for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})")

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

                    if torch.cuda.is_available():
                        device = "cuda"
                        dtype = torch.float16
                    else:
                        device = "cpu"
                        dtype = torch.float32

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
                        dialog.after(0, lambda: _set_autotag_busy(False, f"Autotagged {completed} image(s) [{mode}]."))
                    else:
                        self.log(f"[Dataset Editor] Autotag complete for '{dataset_name}' image '{specific_image_path.name}' (mode={mode})")
                        dialog.after(0, lambda: _set_autotag_busy(False, f"Updated '{specific_image_path.name}' [{mode}]."))
                except Exception as exc:
                    error_text = str(exc)
                    stack_text = traceback.format_exc()
                    self.log(f"[Dataset Editor] Autotag failed for '{dataset_name}': {error_text}")
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
            menu = self.tk.Menu(dialog, tearoff=0)
            menu.add_command(label="Auto Caption", command=lambda p=image_path: _run_autotag_for_images(True, "caption_selected", p))
            menu.add_command(label="Auto Tag (SDXL)", command=lambda p=image_path: _run_autotag_for_images(True, "tags", p))
            menu.add_command(label="Auto Tag+Caption", command=lambda p=image_path: _run_autotag_for_images(True, "tag_plus_caption_selected", p))
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

        def on_editor_mousewheel(event: Any) -> str:
            delta = int(-event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
            editor_canvas.yview_scroll(delta, "units")
            return "break"

        def on_editor_linux_up(_event: Any) -> str:
            editor_canvas.yview_scroll(-1, "units")
            return "break"

        def on_editor_linux_down(_event: Any) -> str:
            editor_canvas.yview_scroll(1, "units")
            return "break"

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
            caption_shell.grid(row=1, column=0, sticky="ew", pady=(4, 0))
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
            caption_widget.bind("<MouseWheel>", on_editor_mousewheel)
            caption_widget.bind("<Button-4>", on_editor_linux_up)
            caption_widget.bind("<Button-5>", on_editor_linux_down)
            caption_path_by_widget[caption_widget] = caption_path
            caption_widget_by_path[caption_path] = caption_widget

        def close_editor() -> None:
            for text_widget in list(caption_path_by_widget.keys()):
                flush_caption_save(text_widget)
            dialog.destroy()

        actions = self.ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="e", pady=(10, 0))
        self.ttk.Button(actions, text="Close", command=close_editor).grid(row=0, column=0)

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
