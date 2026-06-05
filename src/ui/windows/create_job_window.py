from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

WidgetType = Any
EventType = Any


class CreateJobWindow:
    def __init__(self, **dependencies: object) -> None:
        for name, value in dependencies.items():
            setattr(self, name, value)

    def open(self, existing_job: dict[str, str] | None = None) -> None:
        return self._open_create_job_dialog_impl(existing_job)


    def _open_create_job_dialog_impl(self, existing_job: dict[str, str] | None = None) -> None:
        if existing_job is None:
            selected = self.selected_dataset_names()
            if not selected:
                self.messagebox.showinfo("No source dataset selected", "Select at least one dataset card first.", parent=self.root)
                return
            initial_datasets: list[dict] = [{"name": n, "num_repeats": 1} for n in selected]
            dataset_name = selected[0]
        else:
            dataset_name = existing_job.get("dataset_name", "").strip()
            if not dataset_name:
                self.messagebox.showerror("Invalid job", "Job has no dataset name.", parent=self.root)
                return
            raw_datasets = existing_job.get("datasets_json", "")
            if raw_datasets:
                try:
                    initial_datasets = json.loads(raw_datasets)
                except Exception:
                    initial_datasets = [{"name": dataset_name, "num_repeats": 1}]
            else:
                initial_datasets = [{"name": dataset_name, "num_repeats": 1}]

        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Edit Job" if existing_job is not None else "Create Job")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.bg_panel)
        dialog.resizable(False, False)
        self.set_dark_title_bar(dialog)

        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        def _clear_focus(event: EventType) -> None:
            dialog.focus_set()
        
        dialog.bind_class("TEntry", "<Return>", _clear_focus)
        dialog.bind_class("TEntry", "<KP_Enter>", _clear_focus)
        dialog.bind_class("TSpinbox", "<Return>", _clear_focus)
        dialog.bind_class("TSpinbox", "<KP_Enter>", _clear_focus)

        outer = self.ttk.Frame(dialog, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # ── Header: LoRA name + model ──────────────────────────────────────
        header_frame = self.ttk.Frame(outer)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)

        def _fit_create_job_dialog_to_content() -> None:
            if not dialog.winfo_exists():
                return
            dialog.update_idletasks()
            target_width = max(740, dialog.winfo_reqwidth())
            target_height = dialog.winfo_reqheight()
            if dialog.state() == "withdrawn":
                dialog.geometry(f"{target_width}x{target_height}")
            else:
                pos_x = dialog.winfo_x()
                pos_y = dialog.winfo_y()
                dialog.geometry(f"{target_width}x{target_height}+{pos_x}+{pos_y}")

        _model_to_family: dict[str, str] = {
            mn: fam
            for fam, models in self.DOWNLOAD_MODEL_FAMILIES.items()
            for mn in models
        }

        _job_name_equivalence_source = getattr(self, "JOB_NAME_EQUIVALENCE_BY_MODEL", {})
        _job_name_equivalence_by_model: dict[str, str] = {}
        if isinstance(_job_name_equivalence_source, dict):
            _job_name_equivalence_by_model = {
                str(k).strip(): str(v).strip()
                for k, v in _job_name_equivalence_source.items()
                if str(k).strip() and str(v).strip()
            }

        def _family_label(mn: str) -> str:
            mapped_label = _job_name_equivalence_by_model.get((mn or "").strip(), "").strip()
            if mapped_label:
                return mapped_label
            fam = _model_to_family.get(mn, "")
            return fam or mn

        _MODEL_UNSELECTED_LABEL = "-------------"
        model_var = self.tk.StringVar(value=(existing_job or {}).get("model", "").strip())
        ltx_mode_var = self.tk.StringVar(value=(existing_job or {}).get("ltx_mode", "video"))
        if model_var.get().strip():
            default_job_name = f"{dataset_name}_{_family_label(model_var.get().strip())}"
        else:
            default_job_name = dataset_name
        if existing_job is not None:
            default_job_name = existing_job.get("job_name", default_job_name)
        job_name_var = self.tk.StringVar(value=default_job_name)
        _job_name_user_edited = [existing_job is not None]  # track manual edits

        # Build available model list from saved paths
        import json as _json_cj
        _mpaths_cj: dict[str, dict[str, str]] = {}
        try:
            _mpaths_cj = _json_cj.loads(self.settings_state.get(self.MODEL_PATHS_KEY, "{}"))
        except Exception:
            pass
        _all_mn = [mn for fam in self.DOWNLOAD_MODEL_FAMILIES.values() for mn in fam]
        _backend_dirs = self.configured_backend_dirs()

        def _backend_ready_for_model(mn: str) -> bool:
            kind = self.backend_kind_for_model(mn)
            path = _backend_dirs.get(kind)
            if path is None:
                return False
            return self.backend_is_valid(kind, path)

        def _model_has_saved_path(mn: str) -> bool:
            return bool(str(_mpaths_cj.get(mn, {}).get("dit", "")).strip())

        _backend_blocked_models = [
            mn for mn in _all_mn
            if _model_has_saved_path(mn) and not _backend_ready_for_model(mn)
        ]

        _avail_models = [
            mn for mn in _all_mn
            if _mpaths_cj.get(mn, {}).get("dit") and _backend_ready_for_model(mn)
        ]
        # Backward compat: legacy klein key
        if not _avail_models and self.settings_state.get(self.KLEIN_DIT_KEY, "").strip() and _backend_ready_for_model("klein-base-9b"):
            _avail_models = ["klein-base-9b"]
        if not _avail_models:
            _avail_models = [mn for mn in _all_mn if _backend_ready_for_model(mn)]
        if not _avail_models:
            self.messagebox.showinfo(
                "No models available",
                (
                    "Models were found, but their required kohya-ss backend is not ready.\n\n"
                    "Configure Musubi Main and/or Musubi LTX in Settings."
                    if _backend_blocked_models
                    else "No model backend is currently available for Jobs.\n\n"
                    "Configure Musubi Main and/or Musubi LTX in Settings."
                ),
                parent=self.root,
            )
            dialog.destroy()
            return
        if existing_job is not None and model_var.get() not in _avail_models:
            model_var.set(_avail_models[0])
        _display_values = [_MODEL_UNSELECTED_LABEL] + [self.DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn) for mn in _avail_models]
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

        def _normalize_min_one_int_text(raw_value: object, default: str = "1") -> str:
            try:
                value = int(str(raw_value).strip())
            except Exception:
                return default
            return str(value if value >= 1 else 1)

        def _normalize_non_negative_int_text(raw_value: object, default: str = "0") -> str:
            try:
                value = int(str(raw_value).strip())
            except Exception:
                return default
            return str(value if value >= 0 else 0)

        _preferred_presets_raw = self.settings_state.get(self.PREFERRED_PRESETS_BY_FAMILY_KEY, "")
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

        self.ttk.Style().configure(
            "MultiJobBarOn.TFrame",
            background="#0090d8",
        )
        self.ttk.Style().configure(
            "MultiJobBarOff.TFrame",
            background="#353535",
        )
        self.ttk.Style().configure(
            "MultiJob.TCheckbutton",
            foreground="#e8eaed",
            font=("Segoe UI", 11, "bold"),
            background="#353535",
        )
        self.ttk.Style().map(
            "MultiJob.TCheckbutton",
            background=[
                ("selected", "#0090d8"),
                ("!selected", "#353535"),
                ("active selected", "#0090d8"),
                ("active !selected", "#404040"),
            ],
            foreground=[
                ("selected", "#ffffff"),
                ("!selected", "#e8eaed"),
                ("active selected", "#ffffff"),
                ("active !selected", "#e8eaed"),
            ]
        )
        self.ttk.Style().configure(
            "CreateJobAction.TButton",
            background="#353535",
            foreground="#e8eaed",
            padding=(6, 1),
            borderwidth=1,
            relief="flat",
            bordercolor="#474747",
            lightcolor="#4b4b4b",
            darkcolor="#2b2b2b",
            focuscolor="#353535",
            font=("Segoe UI", 9),
        )
        self.ttk.Style().map(
            "CreateJobAction.TButton",
            background=[("active", "#404040"), ("disabled", "#2f2f2f")],
            foreground=[("disabled", "#8a8a8a")],
        )
        self.ttk.Style().configure(
            "CreateJobStepper.TButton",
            background="#353535",
            foreground="#e8eaed",
            padding=(0, 0),
            borderwidth=1,
            relief="flat",
            bordercolor="#474747",
            lightcolor="#4b4b4b",
            darkcolor="#2b2b2b",
            focuscolor="#353535",
            font=("Segoe UI", 9),
            anchor="center",
        )
        self.ttk.Style().map(
            "CreateJobStepper.TButton",
            background=[("active", "#404040"), ("disabled", "#2f2f2f")],
            foreground=[("disabled", "#8a8a8a")],
        )

        default_multi_job = existing_job is None and len(initial_datasets) > 1
        multi_job_var = self.tk.BooleanVar(value=default_multi_job)
        multi_job_frame = self.ttk.Frame(
            header_frame,
            style="MultiJobBarOn.TFrame" if default_multi_job else "MultiJobBarOff.TFrame",
        )
        
        self.ttk.Checkbutton(
            multi_job_frame,
            text=" Create 1 Job per Dataset",
            variable=multi_job_var,
            style="MultiJob.TCheckbutton",
            padding=(6, 4)
        ).pack(side="left", fill="both", expand=True)

        def _on_multi_job_toggle(*_args: object) -> None:
            multi_job_frame.configure(
                style="MultiJobBarOn.TFrame" if multi_job_var.get() else "MultiJobBarOff.TFrame"
            )
            if multi_job_var.get():
                _lora_name_entry.configure(state="disabled")
            else:
                _lora_name_entry.configure(state="normal")
        multi_job_var.trace_add("write", _on_multi_job_toggle)

        label_lora = self.ttk.Label(header_frame, text="LoRA name:")
        label_lora.grid(row=0, column=0, sticky="w", padx=(0, 8))
        _lora_name_entry = self.ttk.Entry(header_frame, textvariable=job_name_var, style="Flat.TEntry")
        _lora_name_entry.grid(row=0, column=1, sticky="ew")
        _lora_name_entry.bind("<Key>", lambda _e: _job_name_user_edited.__setitem__(0, True))
        self.ttk.Label(header_frame, text="Model:").grid(row=0, column=2, sticky="w", padx=(12, 8))
        _model_display_default = self.DOWNLOAD_MODEL_DISPLAY_NAMES.get(model_var.get(), model_var.get()) if model_var.get().strip() else _MODEL_UNSELECTED_LABEL
        _model_display_var = self.tk.StringVar(value=_model_display_default)

        def _on_model_display_change(*_a: object) -> None:
            disp = _model_display_var.get()
            if disp == _MODEL_UNSELECTED_LABEL:
                model_var.set("")
                _sync_model_specific_controls()
                _refresh_preset_combo()
                return
            for mn, dn in self.DOWNLOAD_MODEL_DISPLAY_NAMES.items():
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
        self.ttk.Combobox(header_frame, textvariable=_model_display_var, values=_display_values, state="readonly", width=28).grid(row=0, column=3, sticky="w")

        if _backend_blocked_models:
            blocked_display = sorted(
                {self.DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn) for mn in _backend_blocked_models},
                key=lambda name: name.casefold(),
            )
            blocked_names = ", ".join(blocked_display)
            warning_icon = self.ttk.Label(
                header_frame,
                text="⚠️",
                foreground="#d8c07a",
            )
            warning_icon.grid(row=0, column=4, sticky="w", padx=(8, 0))
            self.attach_hover_tooltip(
                warning_icon,
                (
                    "Some configured models are hidden because the matching kohya-ss backend is not set.\n\n"
                    f"Hidden models: {blocked_names}"
                ),
            )

        # ── Create Job content ─────────────────────────────────────────────
        training_tab = self.ttk.Frame(outer, padding=10)
        training_tab.grid(row=1, column=0, sticky="nsew")
        training_tab.columnconfigure(1, weight=1)
        training_tab.columnconfigure(3, weight=1)

        train_optimizer_var = self.tk.StringVar(
            value=(existing_job or {}).get("optimizer_type", "prodigy")
        )
        train_optimizer_args_var = self.tk.StringVar(
            value=(existing_job or {}).get("optimizer_args", "")
        )
        train_learning_rate_var = self.tk.StringVar(
            value=(existing_job or {}).get("learning_rate", self.DEFAULT_LEARNING_RATE)
        )
        train_steps_var = self.tk.StringVar(
            value=(existing_job or {}).get("train_steps", str(self.DEFAULT_TRAIN_STEPS))
        )
        _default_save_every_from_settings = str(
            self.get_positive_int_setting(
                self.settings_state,
                self.TRAIN_SAVE_EVERY_N_STEPS_KEY,
                self.DEFAULT_SAVE_EVERY_N_STEPS,
                minimum=1,
            )
        )
        train_save_every_var = self.tk.StringVar(
            value=(existing_job or {}).get("save_every_n_steps", _default_save_every_from_settings)
        )
        train_network_dim_var = self.tk.StringVar(
            value=(existing_job or {}).get("network_dim", str(self.DEFAULT_NETWORK_DIM))
        )
        train_network_alpha_var = self.tk.StringVar(
            value=(existing_job or {}).get("network_alpha", str(self.DEFAULT_NETWORK_ALPHA))
        )
        lr_scheduler_var = self.tk.StringVar(value=(existing_job or {}).get("lr_scheduler", "constant"))
        lr_warmup_steps_var = self.tk.StringVar(value=(existing_job or {}).get("lr_warmup_steps", "0"))
        gradient_accumulation_steps_var = self.tk.StringVar(
            value=_normalize_min_one_int_text((existing_job or {}).get("gradient_accumulation_steps", "1"))
        )
        train_batch_var = self.tk.StringVar(
            value=_normalize_min_one_int_text((existing_job or {}).get("batch_size", "1"))
        )
        blocks_to_swap_var = self.tk.StringVar(
            value=_normalize_non_negative_int_text((existing_job or {}).get("blocks_to_swap", "0"))
        )
        timestep_sampling_var = self.tk.StringVar(value=(existing_job or {}).get("timestep_sampling", "sigma"))
        ltx_lora_target_preset_var = self.tk.StringVar(value=(existing_job or {}).get("ltx_lora_target_preset", "full"))
        ltx_first_frame_conditioning_p_var = self.tk.StringVar(value=(existing_job or {}).get("ltx_first_frame_conditioning_p", "0.5"))
        ltx_gemma_load_in_4bit_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("ltx_gemma_load_in_4bit", self.bool_to_flag(True)))
        )
        sd_unet_lr_var = self.tk.StringVar(value=(existing_job or {}).get("sd_unet_lr", ""))
        sd_text_encoder_lr_var = self.tk.StringVar(value=(existing_job or {}).get("sd_text_encoder_lr", ""))

        job_presets = self.load_job_presets_from_disk()
        preset_none_label = "---------"
        preset_name_var = self.tk.StringVar(value=preset_none_label)
        default_enable_compile = False
        default_enable_tf32 = True
        default_enable_cudnn = True
        default_enable_fp8 = False
        default_enable_gc = False
        default_enable_grad_ckpt = True
        compile_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_compile", self.bool_to_flag(default_enable_compile)))
        )
        tf32_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_tf32", self.bool_to_flag(default_enable_tf32)))
        )
        cudnn_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_cudnn", self.bool_to_flag(default_enable_cudnn)))
        )
        fp8_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_fp8", self.bool_to_flag(default_enable_fp8)))
        )
        gc_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_gc", self.bool_to_flag(default_enable_gc)))
        )
        sdxl_grad_ckpt_var = self.tk.BooleanVar(
            value=self.flag_to_bool((existing_job or {}).get("enable_grad_ckpt", self.bool_to_flag(default_enable_grad_ckpt)))
        )
        global_repeats_var: Any | None = None
        _suppress_global_repeats_apply: dict[str, bool] = {"enabled": False}
        _pending_global_repeats_from_preset: dict[str, object] = {"value": None, "apply": False}

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
                "ltx_gemma_load_in_4bit": self.bool_to_flag(ltx_gemma_load_in_4bit_var.get()),
                "sd_unet_lr": sd_unet_lr_var.get().strip(),
                "sd_text_encoder_lr": sd_text_encoder_lr_var.get().strip(),
                "enable_compile": self.bool_to_flag(compile_var.get()),
                "enable_tf32": self.bool_to_flag(tf32_var.get()),
                "enable_cudnn": self.bool_to_flag(cudnn_var.get()),
                "enable_fp8": self.bool_to_flag(fp8_var.get()),
                "enable_gc": self.bool_to_flag(gc_var.get()),
                "enable_grad_ckpt": self.bool_to_flag(sdxl_grad_ckpt_var.get()),
                "global_repeats": (
                    str(global_repeats_var.get()).strip()
                    if global_repeats_var is not None
                    else "1"
                ),
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
                train_batch_var.set(_normalize_min_one_int_text(values["batch_size"]))
            if "lr_scheduler" in values:
                lr_scheduler_var.set(values["lr_scheduler"])
            if "lr_warmup_steps" in values:
                lr_warmup_steps_var.set(values["lr_warmup_steps"])
            if "gradient_accumulation_steps" in values:
                gradient_accumulation_steps_var.set(_normalize_min_one_int_text(values["gradient_accumulation_steps"]))
            if "blocks_to_swap" in values:
                blocks_to_swap_var.set(_normalize_non_negative_int_text(values["blocks_to_swap"]))
            if "timestep_sampling" in values:
                timestep_sampling_var.set(values["timestep_sampling"])
            if "ltx_mode" in values:
                ltx_mode_var.set(_normalize_ltx_mode_ui(values["ltx_mode"]))
            if "ltx_lora_target_preset" in values:
                ltx_lora_target_preset_var.set(values["ltx_lora_target_preset"])
            if "ltx_first_frame_conditioning_p" in values:
                ltx_first_frame_conditioning_p_var.set(values["ltx_first_frame_conditioning_p"])
            if "ltx_gemma_load_in_4bit" in values:
                ltx_gemma_load_in_4bit_var.set(self.flag_to_bool(values["ltx_gemma_load_in_4bit"]))
            if "sd_unet_lr" in values:
                sd_unet_lr_var.set(values["sd_unet_lr"])
            if "sd_text_encoder_lr" in values:
                sd_text_encoder_lr_var.set(values["sd_text_encoder_lr"])
            if "enable_compile" in values:
                compile_var.set(self.flag_to_bool(values["enable_compile"]))
            if "enable_tf32" in values:
                tf32_var.set(self.flag_to_bool(values["enable_tf32"]))
            if "enable_cudnn" in values:
                cudnn_var.set(self.flag_to_bool(values["enable_cudnn"]))
            if "enable_fp8" in values:
                fp8_var.set(self.flag_to_bool(values["enable_fp8"]))
            if "enable_gc" in values:
                gc_var.set(self.flag_to_bool(values["enable_gc"]))
            if "enable_grad_ckpt" in values:
                sdxl_grad_ckpt_var.set(self.flag_to_bool(values["enable_grad_ckpt"]))
            if "global_repeats" in values:
                try:
                    preset_global_repeats = int(str(values["global_repeats"]).strip())
                except Exception:
                    preset_global_repeats = 1
                if preset_global_repeats > 1:
                    if global_repeats_var is not None:
                        global_repeats_var.set(str(preset_global_repeats))
                    else:
                        _pending_global_repeats_from_preset["value"] = str(preset_global_repeats)
                        _pending_global_repeats_from_preset["apply"] = True
                else:
                    # Reset the global control to 1, but do not overwrite per-dataset repeats.
                    if global_repeats_var is not None:
                        _suppress_global_repeats_apply["enabled"] = True
                        try:
                            global_repeats_var.set("1")
                        finally:
                            _suppress_global_repeats_apply["enabled"] = False
                    else:
                        _pending_global_repeats_from_preset["value"] = "1"
                        _pending_global_repeats_from_preset["apply"] = False

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

        preset_section = self.ttk.LabelFrame(training_tab, text="Preset", padding=8)
        preset_section.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        preset_section.columnconfigure(1, weight=1)
        self.ttk.Label(preset_section, text="Preset:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        preset_combo = self.ttk.Combobox(preset_section, textvariable=preset_name_var, state="readonly")
        preset_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        def _refresh_preset_combo() -> None:
            nonlocal job_presets
            job_presets = self.load_job_presets_from_disk()
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
            if not current_model:
                self.messagebox.showerror("Save preset", "Select a model first.", parent=dialog)
                return
            current_family = _model_to_family.get(current_model, "") or current_model
            initial_name = preset_name_var.get().strip()
            if not initial_name or initial_name == preset_none_label:
                initial_name = f"{current_family} preset"
            preset_name = self.simpledialog.askstring(
                "Save preset",
                "Preset name:",
                initialvalue=initial_name,
                parent=dialog,
            )
            if preset_name is None:
                return
            preset_name = preset_name.strip()
            if not preset_name:
                self.messagebox.showerror("Invalid preset", "Preset name is required.", parent=dialog)
                return
            if _preset_payload_for_model_name(current_model, preset_name) is not None:
                if not self.messagebox.askyesno(
                    "Overwrite preset",
                    f"Preset '{preset_name}' already exists for family {current_family}. Overwrite it?",
                    parent=dialog,
                ):
                    return
            self.save_job_preset_to_disk(current_model, current_family, preset_name, _collect_preset_values())
            _refresh_preset_combo()
            preset_name_var.set(preset_name)

        def _reload_presets() -> None:
            _refresh_preset_combo()

        def _delete_preset() -> None:
            selected_name = preset_name_var.get().strip()
            if not selected_name or selected_name == preset_none_label:
                self.messagebox.showerror("Delete preset", "Select a preset to delete.", parent=dialog)
                return

            current_model = model_var.get().strip()
            if not current_model:
                self.messagebox.showerror("Delete preset", "Select a model first.", parent=dialog)
                return
            payload = _preset_payload_for_model_name(current_model, selected_name)
            if not isinstance(payload, dict):
                self.messagebox.showerror("Delete preset", "Preset could not be found on disk.", parent=dialog)
                return

            payload_family = str(payload.get("family", "")).strip() or _model_to_family.get(current_model, "")
            file_path_raw = str(payload.get("file", "")).strip()
            target_path = Path(file_path_raw) if file_path_raw else self.job_preset_file_path(payload_family, selected_name)

            if not self.messagebox.askyesno(
                "Delete preset",
                f"Delete preset '{selected_name}' for family {payload_family}?",
                parent=dialog,
            ):
                return

            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError as exc:
                self.messagebox.showerror("Delete preset", f"Could not delete preset:\n{exc}", parent=dialog)
                return

            _refresh_preset_combo()
            preset_name_var.set(preset_none_label)

        _reload_preset_button = self.ttk.Button(preset_section, text="\u21bb", command=_reload_presets, width=3, style="CreateJobAction.TButton")
        _reload_preset_button.grid(row=0, column=2, sticky="e")
        _save_preset_button = self.ttk.Button(preset_section, text="Save preset", command=_save_preset, style="CreateJobAction.TButton")
        _save_preset_button.grid(row=0, column=3, sticky="e", padx=(6, 0))
        _delete_preset_button = self.ttk.Button(preset_section, text="\U0001F5D1", command=_delete_preset, width=3, style="CreateJobAction.TButton")
        _delete_preset_button.grid(row=0, column=4, sticky="e", padx=(6, 0))
        self.attach_hover_tooltip(_reload_preset_button, "Reload presets from disk")
        self.attach_hover_tooltip(_delete_preset_button, "Delete selected preset")

        def _attach_field_tooltip(label_widget: WidgetType, input_widget: WidgetType, text: str) -> None:
            self.attach_hover_tooltip(label_widget, text)
            self.attach_hover_tooltip(input_widget, text)

        def _spinbox_positive_int_validator(proposed: str) -> bool:
            if not proposed:
                return True
            if not proposed.isdigit():
                return False
            return int(proposed) >= 1

        def _spinbox_non_negative_int_validator(proposed: str) -> bool:
            if not proposed:
                return True
            if not proposed.isdigit():
                return False
            return int(proposed) >= 0

        def _step_int_var(target_var: Any, delta: int, minimum: int, fallback: int) -> None:
            try:
                current = int(str(target_var.get()).strip())
            except Exception:
                current = fallback
            updated = current + delta
            if updated < minimum:
                updated = minimum
            target_var.set(str(updated))

        def _step_positive_int_var(target_var: Any, delta: int) -> None:
            _step_int_var(target_var, delta, minimum=1, fallback=1)

        def _step_non_negative_int_var(target_var: Any, delta: int) -> None:
            _step_int_var(target_var, delta, minimum=0, fallback=0)

        _positive_int_spin_validate_cmd = (dialog.register(_spinbox_positive_int_validator), "%P")
        _non_negative_int_spin_validate_cmd = (dialog.register(_spinbox_non_negative_int_validator), "%P")

        options = self.ttk.LabelFrame(training_tab, text="Training settings", padding=8)
        options.grid(row=1, column=0, columnspan=4, sticky="ew")
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        _optimizer_label = self.ttk.Label(options, text="Optimizer type:")
        _optimizer_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        train_optimizer_combo = self.ttk.Combobox(
            options, textvariable=train_optimizer_var, values=self.OPTIMIZER_TYPE_CHOICES, state="readonly"
        )
        train_optimizer_combo.grid(row=0, column=1, sticky="ew")
        _steps_label = self.ttk.Label(options, text="Training steps:")
        _steps_label.grid(row=0, column=2, sticky="w", padx=(12, 8))
        _steps_entry = self.ttk.Entry(options, textvariable=train_steps_var, style="Flat.TEntry")
        _steps_entry.grid(row=0, column=3, sticky="ew")

        _learning_rate_label = self.ttk.Label(options, text="Learning rate:")
        _learning_rate_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        train_learning_rate_entry = self.ttk.Entry(options, textvariable=train_learning_rate_var, style="Flat.TEntry")
        train_learning_rate_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        _network_dim_label = self.ttk.Label(options, text="LoRA network dim:")
        _network_dim_label.grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _network_dim_combo = self.ttk.Combobox(options, textvariable=train_network_dim_var, values=self.TRAIN_DIM_ALPHA_CHOICES, state="readonly")
        _network_dim_combo.grid(
            row=1, column=3, sticky="ew", pady=(6, 0)
        )
        _save_steps_label = self.ttk.Label(options, text="Save every N steps:")
        _save_steps_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _save_steps_entry = self.ttk.Entry(options, textvariable=train_save_every_var, style="Flat.TEntry")
        _save_steps_entry.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        _network_alpha_label = self.ttk.Label(options, text="LoRA network alpha:")
        _network_alpha_label.grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _network_alpha_combo = self.ttk.Combobox(options, textvariable=train_network_alpha_var, values=self.TRAIN_DIM_ALPHA_CHOICES, state="readonly")
        _network_alpha_combo.grid(
            row=2, column=3, sticky="ew", pady=(6, 0)
        )

        _optimizer_args_label = self.ttk.Label(options, text="Optimizer args:")
        _optimizer_args_label.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _optimizer_args_entry = self.ttk.Entry(options, textvariable=train_optimizer_args_var, style="Flat.TEntry")
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
            "Main step size for updates. For Prodigy, this value is ignored and the launcher uses 1 automatically.",
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

        common_advanced = self.ttk.LabelFrame(training_tab, text="Advanced settings", padding=8)
        common_advanced.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        common_advanced.columnconfigure(1, weight=1, uniform="advanced_inputs")
        common_advanced.columnconfigure(3, weight=1, uniform="advanced_inputs")

        _lr_scheduler_label = self.ttk.Label(common_advanced, text="LR scheduler:")
        _lr_scheduler_label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        _lr_scheduler_combo = self.ttk.Combobox(
            common_advanced,
            textvariable=lr_scheduler_var,
            values=("constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"),
            state="readonly",
        )
        _lr_scheduler_combo.grid(row=0, column=1, sticky="ew")
        _lr_warmup_label = self.ttk.Label(common_advanced, text="LR warmup steps:")
        _lr_warmup_label.grid(row=0, column=2, sticky="w", padx=(12, 8))
        _lr_warmup_entry = self.ttk.Entry(common_advanced, textvariable=lr_warmup_steps_var, style="Flat.TEntry")
        _lr_warmup_entry.grid(row=0, column=3, sticky="ew")

        _grad_accum_label = self.ttk.Label(common_advanced, text="Grad accumulation:")
        _grad_accum_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _grad_accum_controls = self.ttk.Frame(common_advanced)
        _grad_accum_controls.grid(row=1, column=1, sticky="w", pady=(6, 0))
        _grad_accum_controls.columnconfigure(0, weight=0)
        _grad_accum_spin = self.ttk.Spinbox(
            _grad_accum_controls,
            textvariable=gradient_accumulation_steps_var,
            from_=1,
            to=9999,
            increment=1,
            validate="key",
            validatecommand=_positive_int_spin_validate_cmd,
            width=35,
            style="Flat.TEntry",
        )
        _grad_accum_spin.grid(row=0, column=0, sticky="w")
        _grad_accum_down_button = self.ttk.Button(
            _grad_accum_controls,
            text="-",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_positive_int_var(gradient_accumulation_steps_var, -1),
        )
        _grad_accum_down_button.grid(row=0, column=1, sticky="w", padx=(4, 0))
        _grad_accum_up_button = self.ttk.Button(
            _grad_accum_controls,
            text="+",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_positive_int_var(gradient_accumulation_steps_var, 1),
        )
        _grad_accum_up_button.grid(row=0, column=2, sticky="w", padx=(2, 0))
        _batch_size_label = self.ttk.Label(common_advanced, text="Batch size:")
        _batch_size_label.grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _batch_size_controls = self.ttk.Frame(common_advanced)
        _batch_size_controls.grid(row=1, column=3, sticky="w", pady=(6, 0))
        _batch_size_controls.columnconfigure(0, weight=0)
        _batch_size_spin = self.ttk.Spinbox(
            _batch_size_controls,
            textvariable=train_batch_var,
            from_=1,
            to=9999,
            increment=1,
            validate="key",
            validatecommand=_positive_int_spin_validate_cmd,
            width=35,
            style="Flat.TEntry",
        )
        _batch_size_spin.grid(row=0, column=0, sticky="w")
        _batch_size_down_button = self.ttk.Button(
            _batch_size_controls,
            text="-",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_positive_int_var(train_batch_var, -1),
        )
        _batch_size_down_button.grid(row=0, column=1, sticky="w", padx=(4, 0))
        _batch_size_up_button = self.ttk.Button(
            _batch_size_controls,
            text="+",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_positive_int_var(train_batch_var, 1),
        )
        _batch_size_up_button.grid(row=0, column=2, sticky="w", padx=(2, 0))
        _blocks_to_swap_label = self.ttk.Label(common_advanced, text="Blocks to swap:")
        _blocks_to_swap_label.grid(row=2, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _blocks_to_swap_controls = self.ttk.Frame(common_advanced)
        _blocks_to_swap_controls.grid(row=2, column=3, sticky="w", pady=(6, 0))
        _blocks_to_swap_controls.columnconfigure(0, weight=0)
        _blocks_to_swap_spin = self.ttk.Spinbox(
            _blocks_to_swap_controls,
            textvariable=blocks_to_swap_var,
            from_=0,
            to=9999,
            increment=1,
            validate="key",
            validatecommand=_non_negative_int_spin_validate_cmd,
            width=35,
            style="Flat.TEntry",
        )
        _blocks_to_swap_spin.grid(row=0, column=0, sticky="w")
        _blocks_to_swap_down_button = self.ttk.Button(
            _blocks_to_swap_controls,
            text="-",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_non_negative_int_var(blocks_to_swap_var, -1),
        )
        _blocks_to_swap_down_button.grid(row=0, column=1, sticky="w", padx=(4, 0))
        _blocks_to_swap_up_button = self.ttk.Button(
            _blocks_to_swap_controls,
            text="+",
            style="CreateJobAction.TButton",
            width=2,
            command=lambda: _step_non_negative_int_var(blocks_to_swap_var, 1),
        )
        _blocks_to_swap_up_button.grid(row=0, column=2, sticky="w", padx=(2, 0))

        _timestep_sampling_label = self.ttk.Label(common_advanced, text="Timestep sampling:")
        _timestep_sampling_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _timestep_sampling_combo = self.ttk.Combobox(
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
            _grad_accum_spin,
            "Accumulate gradients across N steps before optimizer update. Increases effective batch size.",
        )
        self.attach_hover_tooltip(_grad_accum_down_button, "Decrease gradient accumulation (minimum 1).")
        self.attach_hover_tooltip(_grad_accum_up_button, "Increase gradient accumulation.")
        _attach_field_tooltip(
            _batch_size_label,
            _batch_size_spin,
            "Per-step micro-batch size. Use the arrows to increase/decrease; minimum is 1.",
        )
        self.attach_hover_tooltip(_batch_size_down_button, "Decrease batch size (minimum 1).")
        self.attach_hover_tooltip(_batch_size_up_button, "Increase batch size.")
        _attach_field_tooltip(
            _blocks_to_swap_label,
            _blocks_to_swap_spin,
            "Model-specific memory/perf tuning knob from Musubi scripts. Keep at profile default unless needed.",
        )
        self.attach_hover_tooltip(_blocks_to_swap_down_button, "Decrease blocks to swap (minimum 0).")
        self.attach_hover_tooltip(_blocks_to_swap_up_button, "Increase blocks to swap.")
        _attach_field_tooltip(
            _timestep_sampling_label,
            _timestep_sampling_combo,
            "How timesteps are sampled during flow-matching training. Recommended values are model-family dependent.",
        )

        model_specific = self.ttk.LabelFrame(training_tab, text="Model-specific settings", padding=8)
        model_specific.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        model_specific.columnconfigure(1, weight=1)
        model_specific.columnconfigure(3, weight=1)

        ltx_specific_frame = self.ttk.Frame(model_specific)
        ltx_specific_frame.grid(row=0, column=0, columnspan=4, sticky="ew")
        ltx_specific_frame.columnconfigure(1, weight=1)
        ltx_specific_frame.columnconfigure(3, weight=1)

        self.ttk.Label(ltx_specific_frame, text="LTX mode:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _ltx_mode_display_var = self.tk.StringVar(
            value=_ltx_mode_value_to_display.get(_normalize_ltx_mode_ui(ltx_mode_var.get()), "Image Training")
        )

        def _on_ltx_mode_display_change(*_args: object) -> None:
            ltx_mode_var.set(_ltx_mode_display_to_value.get(_ltx_mode_display_var.get(), "video"))

        _ltx_mode_display_var.trace_add("write", _on_ltx_mode_display_change)
        _ltx_mode_combo = self.ttk.Combobox(
            ltx_specific_frame,
            textvariable=_ltx_mode_display_var,
            values=list(_ltx_mode_display_to_value.keys()),
            state="readonly",
        )
        _ltx_mode_combo.grid(row=0, column=1, sticky="ew", pady=(6, 0))

        self.ttk.Label(ltx_specific_frame, text="LoRA target preset:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _ltx_lora_target_combo = self.ttk.Combobox(
            ltx_specific_frame,
            textvariable=ltx_lora_target_preset_var,
            values=_ltx_image_lora_target_choices,
            state="readonly",
        )
        _ltx_lora_target_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        self.ttk.Label(ltx_specific_frame, text="First-frame conditioning p:").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _ltx_first_frame_entry = self.ttk.Entry(ltx_specific_frame, textvariable=ltx_first_frame_conditioning_p_var, style="Flat.TEntry")
        _ltx_first_frame_entry.grid(row=1, column=3, sticky="ew", pady=(6, 0))

        _ltx_gemma_4bit_check = self.ttk.Checkbutton(
            ltx_specific_frame,
            text="Gemma load in 4-bit",
            variable=ltx_gemma_load_in_4bit_var,
        )
        _ltx_gemma_4bit_check.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.attach_hover_tooltip(
            _ltx_lora_target_combo,
            (
                "LoRA target preset controls which parts of the LTX model receive LoRA adapters.\n"
                "For image-training workflows, presets are limited to video/image-relevant targets."
            ),
        )
        self.attach_hover_tooltip(
            _ltx_first_frame_entry,
            (
                "First-frame conditioning probability.\n"
                "Higher values bias training to preserve frame-0 identity/composition guidance."
            ),
        )
        self.attach_hover_tooltip(
            _ltx_gemma_4bit_check,
            (
                "Use bitsandbytes 4-bit quantization for Gemma when a Gemma root folder is used.\n"
                "Reduces VRAM for text caching/training text path; disable if troubleshooting quantization behavior."
            ),
        )

        sd_scripts_specific_frame = self.ttk.Frame(model_specific)
        sd_scripts_specific_frame.grid(row=1, column=0, columnspan=4, sticky="ew")
        sd_scripts_specific_frame.columnconfigure(1, weight=1)
        sd_scripts_specific_frame.columnconfigure(3, weight=1)

        _sd_unet_lr_label = self.ttk.Label(sd_scripts_specific_frame, text="U-Net LR override:")
        _sd_unet_lr_label.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        _sd_unet_lr_entry = self.ttk.Entry(sd_scripts_specific_frame, textvariable=sd_unet_lr_var, style="Flat.TEntry")
        _sd_unet_lr_entry.grid(row=0, column=1, sticky="ew", pady=(6, 0))

        _sd_text_encoder_lr_label = self.ttk.Label(sd_scripts_specific_frame, text="Text Encoder LR override:")
        _sd_text_encoder_lr_label.grid(row=0, column=2, sticky="w", padx=(12, 8), pady=(6, 0))
        _sd_text_encoder_lr_entry = self.ttk.Entry(sd_scripts_specific_frame, textvariable=sd_text_encoder_lr_var, style="Flat.TEntry")
        _sd_text_encoder_lr_entry.grid(row=0, column=3, sticky="ew", pady=(6, 0))

        _attach_field_tooltip(
            _sd_unet_lr_label,
            _sd_unet_lr_entry,
            "SD-Scripts only. Optional U-Net learning-rate override. Leave blank to use the global learning rate.",
        )
        _attach_field_tooltip(
            _sd_text_encoder_lr_label,
            _sd_text_encoder_lr_entry,
            "SD-Scripts only. Optional Text Encoder learning-rate override. Leave blank to use defaults.",
        )

        flags = self.ttk.LabelFrame(training_tab, text="Advanced flags", padding=8)
        flags.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        flags.columnconfigure(0, weight=1)
        flags.columnconfigure(1, weight=1)
        _compile_check = self.ttk.Checkbutton(flags, text="Enable Torch Compile", variable=compile_var)
        _compile_check.grid(row=0, column=0, sticky="w")
        _tf32_check = self.ttk.Checkbutton(flags, text="Enable Allow TF32", variable=tf32_var)
        _tf32_check.grid(row=2, column=0, sticky="w", pady=(6, 0))
        _cudnn_check = self.ttk.Checkbutton(flags, text="Enable cuDNN Benchmark", variable=cudnn_var)
        _cudnn_check.grid(row=2, column=1, sticky="w", pady=(6, 0), padx=(12, 0))
        _fp8_check = self.ttk.Checkbutton(flags, text="Enable FP8 (Low VRAM)", variable=fp8_var)
        _fp8_check.grid(row=0, column=1, sticky="w", padx=(12, 0))
        _sdxl_grad_ckpt_check = self.ttk.Checkbutton(
            flags,
            text="Enable SDXL Gradient Checkpointing (Low VRAM, Slower)",
            variable=sdxl_grad_ckpt_var,
        )
        _sdxl_grad_ckpt_check.grid(row=1, column=0, sticky="w", pady=(6, 0))
        _gc_check = self.ttk.Checkbutton(flags, text="Enable CPU Offload for Checkpointing (Low VRAM)", variable=gc_var)
        _gc_check.grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(12, 0))

        self.attach_hover_tooltip(
            _compile_check,
            (
                "Enable PyTorch compile graph optimizations.\n"
                "Can improve training throughput after warmup, with extra startup/compile overhead."
            ),
        )
        self.attach_hover_tooltip(
            _fp8_check,
            (
                "Low-VRAM mode using FP8 where supported by backend/model.\n"
                "May reduce memory usage but can be slower or less stable on some setups."
            ),
        )
        self.attach_hover_tooltip(
            _tf32_check,
            (
                "Allow TF32 math on supported NVIDIA GPUs (typically Ampere+).\n"
                "Usually faster with a small precision tradeoff. Non-SD-scripts models only in this UI."
            ),
        )
        self.attach_hover_tooltip(
            _cudnn_check,
            (
                "Enable cuDNN benchmark autotuning for convolution kernels.\n"
                "Can improve speed for stable input shapes. Non-SD-scripts models only in this UI."
            ),
        )
        self.attach_hover_tooltip(
            _sdxl_grad_ckpt_check,
            (
                "SD-Scripts SDXL only. Reduces VRAM use by recomputing activations during backward pass.\n"
                "Usually slower per step; disable on high-VRAM GPUs for higher throughput."
            ),
        )
        self.attach_hover_tooltip(
            _gc_check,
            (
                "Offloads gradient-checkpointing state to CPU for extra memory savings.\n"
                "Can be significantly slower and may increase host RAM / PCIe traffic."
            ),
        )
        _sd_scripts_hidden_widgets: list[WidgetType] = [
            _blocks_to_swap_label,
            _blocks_to_swap_controls,
            _timestep_sampling_label,
            _timestep_sampling_combo,
        ]
        _advanced_state_specs: list[tuple[WidgetType, str]] = [
            (_lr_scheduler_combo, "readonly"),
            (_lr_warmup_entry, "normal"),
            (_grad_accum_spin, "normal"),
            (_grad_accum_down_button, "normal"),
            (_grad_accum_up_button, "normal"),
            (_batch_size_spin, "normal"),
            (_batch_size_down_button, "normal"),
            (_batch_size_up_button, "normal"),
            (_blocks_to_swap_spin, "normal"),
            (_blocks_to_swap_down_button, "normal"),
            (_blocks_to_swap_up_button, "normal"),
        ]
        _flags_state_specs: list[tuple[WidgetType, str]] = [(_compile_check, "normal"), (_fp8_check, "normal"), (_sdxl_grad_ckpt_check, "normal"), (_gc_check, "normal"), (_tf32_check, "normal"), (_cudnn_check, "normal")]
        _flag_display_order: list[WidgetType] = [
            _compile_check,
            _fp8_check,
            _sdxl_grad_ckpt_check,
            _gc_check,
            _tf32_check,
            _cudnn_check,
        ]

        def _layout_visible_flags(is_sd_scripts: bool) -> None:
            for widget in _flag_display_order:
                widget.grid_remove()

            visible_flags: list[WidgetType] = [_compile_check, _fp8_check]
            if is_sd_scripts:
                visible_flags.extend([_sdxl_grad_ckpt_check, _gc_check])
            else:
                visible_flags.extend([_tf32_check, _gc_check, _cudnn_check])

            for index, widget in enumerate(visible_flags):
                row = index // 2
                column = index % 2
                widget.grid(
                    row=row,
                    column=column,
                    sticky="w",
                    pady=((0, 0) if row == 0 else (6, 0)),
                    padx=((0, 0) if column == 0 else (12, 0)),
                )

        def _sync_backend_specific_controls() -> None:
            model_name = model_var.get().strip()
            is_sd_scripts = self.backend_kind_for_model(model_name) == "sd-scripts"

            for widget, enabled_state in _advanced_state_specs:
                widget.configure(state=enabled_state)
            for widget, enabled_state in _flags_state_specs:
                widget.configure(state=enabled_state)

            for widget in _sd_scripts_hidden_widgets:
                if is_sd_scripts:
                    widget.grid_remove()
                else:
                    widget.grid()

            _layout_visible_flags(is_sd_scripts)

        def sync_optimizer_controls() -> None:
            optimizer_value = train_optimizer_var.get().strip().lower()
            if optimizer_value == "prodigy":
                if not train_optimizer_args_var.get().strip():
                    train_optimizer_args_var.set(self.DEFAULT_PRODIGY_OPTIMIZER_ARGS)
            else:
                if train_optimizer_args_var.get().strip() == self.DEFAULT_PRODIGY_OPTIMIZER_ARGS:
                    train_optimizer_args_var.set("")

        def _sync_model_specific_controls() -> None:
            model_name = model_var.get().strip()
            is_ltx = _model_to_family.get(model_name, "") == "LTX"
            is_sd_scripts = self.backend_kind_for_model(model_name) == "sd-scripts"
            if is_ltx:
                model_specific.grid()
                ltx_specific_frame.grid()
                _ltx_mode_combo.configure(state="disabled")
                ltx_mode_var.set("video")
                _ltx_mode_display_var.set(_ltx_mode_value_to_display["video"])
                if ltx_lora_target_preset_var.get().strip() not in _ltx_image_lora_target_choices:
                    ltx_lora_target_preset_var.set("full")
            else:
                ltx_specific_frame.grid_remove()

            if is_sd_scripts:
                model_specific.grid()
                sd_scripts_specific_frame.grid()
            else:
                sd_scripts_specific_frame.grid_remove()

            if not is_ltx and not is_sd_scripts:
                model_specific.grid_remove()
            _sync_backend_specific_controls()
            _fit_create_job_dialog_to_content()

        train_optimizer_var.trace_add("write", lambda *_args: sync_optimizer_controls())
        sync_optimizer_controls()
        _sync_model_specific_controls()
        _refresh_preset_combo()

        # ── Datasets (bottom section) ─────────────────────────────────────
        datasets_section = self.ttk.LabelFrame(training_tab, text="Datasets", padding=8)
        datasets_section.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        datasets_section.columnconfigure(0, weight=1)
        datasets_section.rowconfigure(1, weight=1)

        # Resolution row (written to [general] section of dataset.toml)
        res_row = self.ttk.Frame(datasets_section)
        res_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        res_row.columnconfigure(1, weight=1)
        _saved_res = int((existing_job or {}).get("resolution", str(self.DEFAULT_RESOLUTION)))
        _saved_res_str = str(_saved_res) if _saved_res in self.RESOLUTION_CHOICES else str(self.RESOLUTION_CHOICES[self.RESOLUTION_CHOICES.index(1024)])
        train_resolution_var = self.tk.StringVar(value=_saved_res_str)
        _ltx_resolution_choices = (1280, 1920)
        _ltx_resolution_choice_values = [str(r) for r in _ltx_resolution_choices]
        _default_resolution_choice_values = [str(r) for r in self.RESOLUTION_CHOICES]
        
        res_left = self.ttk.Frame(res_row)
        res_left.grid(row=0, column=0, sticky="w")
        self.ttk.Label(res_left, text="Resolution:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        _resolution_combo = self.ttk.Combobox(
            res_left, textvariable=train_resolution_var,
            values=_default_resolution_choice_values,
            state="readonly", width=7,
        )
        _resolution_combo.grid(row=0, column=1, sticky="w")
        
        # Apply to all repeats controls
        global_repeats_var = self.tk.StringVar(value="1")
        global_repeats_frame = self.ttk.Frame(res_row)
        global_repeats_frame.grid(row=0, column=2, sticky="e", padx=(0, 8))
        global_repeats_frame.columnconfigure(4, minsize=26)
        
        def _apply_global_repeats(*_args: object) -> None:
            if _suppress_global_repeats_apply["enabled"]:
                return
            val = global_repeats_var.get().strip()
            if not val or not val.isdigit():
                return
            for entry in dataset_entries:
                entry["num_repeats_var"].set(val)
        
        self.ttk.Label(global_repeats_frame, text="Repeats All:", style="CardMeta.TLabel").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.ttk.Spinbox(
            global_repeats_frame,
            textvariable=global_repeats_var,
            from_=1, to=9999, increment=1,
            validate="key", validatecommand=_positive_int_spin_validate_cmd,
            width=8, style="Flat.TEntry",
        ).grid(row=0, column=1, sticky="e")
        self.ttk.Button(
            global_repeats_frame, text="-", style="CreateJobStepper.TButton", width=2,
            command=lambda: (_step_positive_int_var(global_repeats_var, -1), _apply_global_repeats())
        ).grid(row=0, column=2, sticky="e", padx=(4, 0))
        self.ttk.Button(
            global_repeats_frame, text="+", style="CreateJobStepper.TButton", width=2,
            command=lambda: (_step_positive_int_var(global_repeats_var, 1), _apply_global_repeats())
        ).grid(row=0, column=3, sticky="e", padx=(2, 0))
        # Reserve the same trailing slot used by per-row remove buttons so +/- align vertically.
        self.ttk.Frame(global_repeats_frame).grid(row=0, column=4, sticky="e")
        
        # We bind return key to unfocus, and trace write so direct typing also syncs
        global_repeats_var.trace_add("write", _apply_global_repeats)

        def _sync_resolution_controls() -> None:
            model_name = model_var.get().strip()
            is_ltx = _model_to_family.get(model_name, "") == "LTX"
            allowed_values = _ltx_resolution_choice_values if is_ltx else _default_resolution_choice_values
            _resolution_combo.configure(values=allowed_values)

            current_value = train_resolution_var.get().strip()
            if current_value not in allowed_values:
                train_resolution_var.set("1920" if is_ltx else _saved_res_str)

        _family_default_profiles: dict[str, dict[str, str]] = {
            "SDXL": {
                "optimizer_type": "prodigy",
                "optimizer_args": self.DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": str(self.DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(self.DEFAULT_NETWORK_DIM),
                "network_alpha": str(self.DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant_with_warmup",
                "lr_warmup_steps": "100",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "sigma",
                "resolution": "1024",
                "batch_size": "1",
            },
            "FLUX.2": {
                "optimizer_type": "prodigy",
                "optimizer_args": self.DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": str(self.DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(self.DEFAULT_NETWORK_DIM),
                "network_alpha": str(self.DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "flux2_shift",
                "resolution": str(self.DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "LTX": {
                "optimizer_type": "adamw8bit",
                "optimizer_args": "",
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": "400",
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": "16",
                "network_alpha": "16",
                "lr_scheduler": "constant_with_warmup",
                "lr_warmup_steps": "100",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "shifted_logit_normal",
                "resolution": "1920",
                "batch_size": "1",
                "ltx_lora_target_preset": "full",
                "ltx_first_frame_conditioning_p": "0.5",
                "ltx_gemma_load_in_4bit": "1",
            },
            "Wan": {
                "optimizer_type": "prodigy",
                "optimizer_args": self.DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": str(self.DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(self.DEFAULT_NETWORK_DIM),
                "network_alpha": str(self.DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "shift",
                "resolution": str(self.DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "Z-Image": {
                "optimizer_type": "prodigy",
                "optimizer_args": self.DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": str(self.DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(self.DEFAULT_NETWORK_DIM),
                "network_alpha": str(self.DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "shift",
                "resolution": str(self.DEFAULT_RESOLUTION),
                "batch_size": "1",
            },
            "Qwen-Image": {
                "optimizer_type": "prodigy",
                "optimizer_args": self.DEFAULT_PRODIGY_OPTIMIZER_ARGS,
                "learning_rate": self.DEFAULT_LEARNING_RATE,
                "train_steps": str(self.DEFAULT_TRAIN_STEPS),
                "save_every_n_steps": _default_save_every_from_settings,
                "network_dim": str(self.DEFAULT_NETWORK_DIM),
                "network_alpha": str(self.DEFAULT_NETWORK_ALPHA),
                "lr_scheduler": "constant",
                "lr_warmup_steps": "0",
                "gradient_accumulation_steps": "1",
                "blocks_to_swap": "0",
                "timestep_sampling": "qwen_shift",
                "resolution": str(self.DEFAULT_RESOLUTION),
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
            if "ltx_gemma_load_in_4bit" in profile:
                ltx_gemma_load_in_4bit_var.set(self.flag_to_bool(profile["ltx_gemma_load_in_4bit"]))

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
        list_host = self.ttk.Frame(datasets_section)
        list_host.grid(row=1, column=0, sticky="nsew")
        list_host.columnconfigure(0, weight=1)
        list_host.rowconfigure(0, weight=1)

        ds_canvas = self.tk.Canvas(list_host, bg=self.bg_panel, highlightthickness=0, height=120)
        ds_canvas.grid(row=0, column=0, sticky="nsew")
        ds_scrollbar = self.ttk.Scrollbar(list_host, orient="vertical", command=ds_canvas.yview, style="Dark.Vertical.TScrollbar")
        ds_scrollbar.grid(row=0, column=1, sticky="ns")
        ds_canvas.configure(yscrollcommand=ds_scrollbar.set)
        try:
            _scrollbar_width = int(str(ds_scrollbar.cget("width") or "12"))
        except Exception:
            _scrollbar_width = 12
        # Match right edge with rows inside the canvas (which exclude scrollbar width).
        global_repeats_frame.grid_configure(padx=(0, 12 + _scrollbar_width))

        ds_inner = self.ttk.Frame(ds_canvas)
        ds_inner_id = ds_canvas.create_window((0, 0), window=ds_inner, anchor="nw")
        ds_inner.columnconfigure(0, weight=1)

        def _sync_ds_scroll(_e: object = None) -> None:
            ds_canvas.configure(scrollregion=ds_canvas.bbox("all"))

        def _sync_ds_canvas_width(e: EventType) -> None:
            ds_canvas.itemconfigure(ds_inner_id, width=e.width)

        ds_inner.bind("<Configure>", _sync_ds_scroll)
        ds_canvas.bind("<Configure>", _sync_ds_canvas_width)

        # dataset_entries: list of {"name", "num_repeats_var", "frame"}
        dataset_entries: list[dict] = []
        dataset_image_count_cache: dict[str, int] = {}
        dataset_thumbnail_cache: dict[tuple[str, int, str, int], Any] = {}
        estimated_epochs_var = self.tk.StringVar(value="Est. epochs: n/a")

        def _dataset_preview_image_path(name: str) -> Path | None:
            image_paths = self.dataset_image_files(self.datasets_root_dir(), name)
            return image_paths[0] if image_paths else None

        def _dataset_thumbnail(name: str, thumb_px: int) -> Any:
            image_path = _dataset_preview_image_path(name)
            cache_path = str(image_path) if image_path is not None else "__none__"
            cache_mtime_ns = 0
            if image_path is not None:
                try:
                    cache_mtime_ns = image_path.stat().st_mtime_ns
                except OSError:
                    cache_mtime_ns = 0
            cache_key = (name, thumb_px, cache_path, cache_mtime_ns)
            cached = dataset_thumbnail_cache.get(cache_key)
            if cached is not None:
                return cached

            if Image is None or ImageTk is None:
                photo = self.tk.PhotoImage(master=dialog, width=thumb_px, height=thumb_px)
                photo.put("#3a3a3a", to=(0, 0, thumb_px, thumb_px))
                dataset_thumbnail_cache[cache_key] = photo
                return photo

            thumb_size = (thumb_px, thumb_px)
            if image_path is None:
                image = Image.new("RGB", thumb_size, color="#3a3a3a")
            else:
                try:
                    image = Image.open(image_path).convert("RGB")
                    src_w, src_h = image.size
                    dst_w, dst_h = thumb_size
                    scale = max(dst_w / src_w, dst_h / src_h)
                    resized_w = max(1, int(round(src_w * scale)))
                    resized_h = max(1, int(round(src_h * scale)))
                    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    image = image.resize((resized_w, resized_h), resample)
                    crop_x = max(0, (resized_w - dst_w) // 2)
                    crop_y = max(0, (resized_h - dst_h) // 2)
                    image = image.crop((crop_x, crop_y, crop_x + dst_w, crop_y + dst_h))
                except Exception:
                    image = Image.new("RGB", thumb_size, color="#3a3a3a")

            photo = ImageTk.PhotoImage(image, master=dialog)
            dataset_thumbnail_cache[cache_key] = photo
            return photo

        def _dataset_image_count(name: str) -> int:
            cached = dataset_image_count_cache.get(name)
            if cached is not None:
                return cached
            count = len(self.dataset_image_files(self.datasets_root_dir(), name))
            dataset_image_count_cache[name] = count
            return count

        def _refresh_estimated_epochs() -> None:
            try:
                steps_value = int(train_steps_var.get().strip())
                batch_value = int(train_batch_var.get().strip())
                grad_accum_value = int(gradient_accumulation_steps_var.get().strip())
            except ValueError:
                estimated_epochs_var.set("Est. epochs: n/a")
                return

            if steps_value <= 0 or batch_value <= 0 or grad_accum_value <= 0:
                estimated_epochs_var.set("Est. epochs: n/a")
                return

            total_images = 0
            effective_samples = 0
            for entry in dataset_entries:
                try:
                    repeats = int(entry["num_repeats_var"].get().strip())
                except ValueError:
                    estimated_epochs_var.set("Est. epochs: n/a")
                    return
                if repeats <= 0:
                    estimated_epochs_var.set("Est. epochs: n/a")
                    return
                image_count = _dataset_image_count(entry["name"])
                total_images += image_count
                effective_samples += image_count * repeats

            if effective_samples <= 0:
                estimated_epochs_var.set("Est. epochs: n/a (no images)")
                return

            denom = batch_value * grad_accum_value
            steps_per_epoch = max(1, (effective_samples + denom - 1) // denom)
            epochs_value = steps_value / steps_per_epoch
            estimated_epochs_var.set(
                f"Est. epochs: {epochs_value:.2f} (img: {total_images}, repeated: {effective_samples}, eff batch: {denom}, steps/epoch: {steps_per_epoch})"
            )

        def _rebuild_ds_rows() -> None:
            if len(dataset_entries) > 1:
                multi_job_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            else:
                multi_job_var.set(False)
                multi_job_frame.grid_remove()

            for child in ds_inner.winfo_children():
                child.destroy()
            for idx, entry in enumerate(dataset_entries):
                row_frame = self.ttk.Frame(ds_inner, padding=(4, 3, 4, 3), style="Card.TFrame")
                row_frame.grid(row=idx, column=0, sticky="ew", pady=2)
                row_frame.columnconfigure(1, weight=1)
                entry["frame"] = row_frame

                thumb = _dataset_thumbnail(entry["name"], 28)
                self.ttk.Label(row_frame, image=thumb).grid(
                    row=0, column=0, sticky="w", padx=(4, 8)
                )
                self.ttk.Label(row_frame, text=entry["name"], style="CardTitle.TLabel").grid(
                    row=0, column=1, sticky="w", padx=(0, 12)
                )
                self.ttk.Label(row_frame, text="Repeats:", style="CardMeta.TLabel").grid(
                    row=0, column=2, sticky="e", padx=(0, 6)
                )
                _repeats_controls = self.ttk.Frame(row_frame)
                _repeats_controls.grid(row=0, column=3, sticky="e", padx=(0, 8))
                self.ttk.Spinbox(
                    _repeats_controls,
                    textvariable=entry["num_repeats_var"],
                    from_=1,
                    to=9999,
                    increment=1,
                    validate="key",
                    validatecommand=_positive_int_spin_validate_cmd,
                    width=8,
                    style="Flat.TEntry",
                ).grid(
                    row=0, column=0, sticky="e"
                )
                self.ttk.Button(
                    _repeats_controls,
                    text="-",
                    style="CreateJobStepper.TButton",
                    width=2,
                    command=lambda v=entry["num_repeats_var"]: _step_positive_int_var(v, -1),
                ).grid(row=0, column=1, sticky="e", padx=(4, 0))
                self.ttk.Button(
                    _repeats_controls,
                    text="+",
                    style="CreateJobStepper.TButton",
                    width=2,
                    command=lambda v=entry["num_repeats_var"]: _step_positive_int_var(v, 1),
                ).grid(row=0, column=2, sticky="e", padx=(2, 0))

                def _make_remove(e: dict = entry) -> None:
                    dataset_entries.remove(e)
                    _rebuild_ds_rows()
                    _refresh_add_combo()
                    _refresh_estimated_epochs()

                self.ttk.Button(row_frame, text="✕", style="CreateJobAction.TButton", command=_make_remove, width=2).grid(
                    row=0, column=4, sticky="e"
                )

            _refresh_estimated_epochs()

        def _available_datasets() -> list[str]:
            return sorted(self.scan_training_folders(self.datasets_root_dir()), key=str.casefold)

        def _available_datasets_to_add() -> list[str]:
            already = {e["name"] for e in dataset_entries}
            return [n for n in _available_datasets() if n not in already]

        def _refresh_add_combo() -> None:
            available = _available_datasets_to_add()
            add_combo["values"] = available
            if available and add_combo.get() not in available:
                add_combo.set(available[0])
            elif not available:
                add_combo.set("")

        # Populate from initial_datasets
        for _ds in initial_datasets:
            _repeats_var = self.tk.StringVar(value=_normalize_min_one_int_text(_ds.get("num_repeats", 1)))
            _repeats_var.trace_add("write", lambda *_args: _refresh_estimated_epochs())
            dataset_entries.append({"name": _ds["name"], "num_repeats_var": _repeats_var, "frame": None})
        _rebuild_ds_rows()
        if _pending_global_repeats_from_preset["value"] is not None:
            pending_value = str(_pending_global_repeats_from_preset["value"])
            pending_apply = bool(_pending_global_repeats_from_preset.get("apply", False))
            if pending_apply:
                global_repeats_var.set(pending_value)
            else:
                _suppress_global_repeats_apply["enabled"] = True
                try:
                    global_repeats_var.set(pending_value)
                finally:
                    _suppress_global_repeats_apply["enabled"] = False
            _pending_global_repeats_from_preset["value"] = None
            _pending_global_repeats_from_preset["apply"] = False

        # Add dataset row
        add_row = self.ttk.Frame(datasets_section, padding=(0, 8, 0, 0))
        add_row.grid(row=2, column=0, sticky="ew")
        add_row.columnconfigure(1, weight=1)
        self.ttk.Label(add_row, text="Add dataset:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        add_combo = self.ttk.Combobox(add_row, state="readonly", width=24)
        add_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        _refresh_add_combo()

        def _add_dataset() -> None:
            name = add_combo.get().strip()
            if not name or any(e["name"] == name for e in dataset_entries):
                return
            _repeats_var = self.tk.StringVar(value="1")
            _repeats_var.trace_add("write", lambda *_args: _refresh_estimated_epochs())
            dataset_entries.append({"name": name, "num_repeats_var": _repeats_var, "frame": None})
            _rebuild_ds_rows()
            _refresh_add_combo()
            _refresh_estimated_epochs()

        self.ttk.Button(add_row, text="Add", command=_add_dataset, style="CreateJobAction.TButton").grid(row=0, column=2, sticky="w")
        train_steps_var.trace_add("write", lambda *_args: _refresh_estimated_epochs())
        train_batch_var.trace_add("write", lambda *_args: _refresh_estimated_epochs())
        gradient_accumulation_steps_var.trace_add("write", lambda *_args: _refresh_estimated_epochs())
        _refresh_estimated_epochs()

        # ── create_job / save ──────────────────────────────────────────────
        def create_job() -> None:
            job_name = job_name_var.get().strip()
            if not job_name:
                self.messagebox.showerror("Missing value", "LoRA name is required.", parent=dialog)
                return
            selected_model_name = model_var.get().strip()
            if not selected_model_name or selected_model_name not in _avail_models:
                self.messagebox.showerror("Missing value", "Select a model before creating the job.", parent=dialog)
                return
            if not self.is_valid_folder_name(job_name):
                self.messagebox.showerror(
                    "Invalid name",
                    "LoRA name must be a valid folder name. Spaces and '-' are allowed.",
                    parent=dialog,
                )
                return
            if not dataset_entries:
                self.messagebox.showerror("No datasets", "Add at least one dataset in the Datasets section.", parent=dialog)
                return

            resolution_value = int(train_resolution_var.get())

            try:
                batch_size_value = int(train_batch_var.get().strip())
                if batch_size_value < 1:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "Batch size must be a positive integer.", parent=dialog)
                return

            try:
                _ = int(train_steps_var.get().strip())
            except ValueError:
                self.messagebox.showerror("Invalid value", "Steps must be numeric.", parent=dialog)
                return

            try:
                save_every_n_steps_value = int(train_save_every_var.get().strip())
                if save_every_n_steps_value < 1:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "Save every N steps must be a positive integer.", parent=dialog)
                return

            train_optimizer = train_optimizer_var.get().strip().lower()
            if train_optimizer not in set(self.OPTIMIZER_TYPE_CHOICES):
                self.messagebox.showerror(
                    "Invalid value",
                    "Optimizer type must be one of: " + ", ".join(self.OPTIMIZER_TYPE_CHOICES),
                    parent=dialog,
                )
                return
            train_optimizer_args = train_optimizer_args_var.get().strip()
            if train_optimizer == "prodigy" and not train_optimizer_args:
                train_optimizer_args = self.DEFAULT_PRODIGY_OPTIMIZER_ARGS
                train_optimizer_args_var.set(train_optimizer_args)

            train_learning_rate = train_learning_rate_var.get().strip()
            if train_optimizer == "prodigy":
                train_learning_rate = "1"
            if not train_learning_rate:
                self.messagebox.showerror("Invalid value", "Learning rate is required.", parent=dialog)
                return
            try:
                learning_rate_number = float(train_learning_rate)
            except ValueError:
                self.messagebox.showerror("Invalid value", "Learning rate must be numeric (example: 1e-4).", parent=dialog)
                return
            if learning_rate_number <= 0:
                self.messagebox.showerror("Invalid value", "Learning rate must be greater than 0.", parent=dialog)
                return

            lr_scheduler_value = lr_scheduler_var.get().strip().lower() or "constant"
            try:
                lr_warmup_steps_value = int(lr_warmup_steps_var.get().strip())
                if lr_warmup_steps_value < 0:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "LR warmup steps must be a non-negative integer.", parent=dialog)
                return

            try:
                grad_accum_value = int(gradient_accumulation_steps_var.get().strip())
                if grad_accum_value < 1:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "Gradient accumulation must be a positive integer.", parent=dialog)
                return

            try:
                blocks_to_swap_value = int(blocks_to_swap_var.get().strip())
                if blocks_to_swap_value < 0:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "Blocks to swap must be a non-negative integer.", parent=dialog)
                return

            timestep_sampling_value = timestep_sampling_var.get().strip().lower() or "sigma"
            ltx_lora_target_preset_value = ltx_lora_target_preset_var.get().strip().lower() or "t2v"
            try:
                ltx_first_frame_conditioning_p_value = float(ltx_first_frame_conditioning_p_var.get().strip())
                if ltx_first_frame_conditioning_p_value < 0 or ltx_first_frame_conditioning_p_value > 1:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Invalid value", "First-frame conditioning p must be a number between 0 and 1.", parent=dialog)
                return

            sd_unet_lr_value = sd_unet_lr_var.get().strip()
            if sd_unet_lr_value:
                try:
                    if float(sd_unet_lr_value) <= 0:
                        raise ValueError
                except ValueError:
                    self.messagebox.showerror("Invalid value", "SD-Scripts U-Net LR must be a number greater than 0.", parent=dialog)
                    return

            sd_text_encoder_lr_value = sd_text_encoder_lr_var.get().strip()
            if sd_text_encoder_lr_value:
                try:
                    if float(sd_text_encoder_lr_value) <= 0:
                        raise ValueError
                except ValueError:
                    self.messagebox.showerror("Invalid value", "SD-Scripts Text Encoder LR must be a number greater than 0.", parent=dialog)
                    return

            # Validate and collect per-dataset config
            datasets_config: list[dict] = []
            for entry in dataset_entries:
                try:
                    repeats = int(entry["num_repeats_var"].get().strip())
                    if repeats < 1:
                        raise ValueError
                except ValueError:
                    self.messagebox.showerror(
                        "Invalid value",
                        f"Repeats for '{entry['name']}' must be a positive integer.",
                        parent=dialog,
                    )
                    return
                datasets_config.append({"name": entry["name"], "num_repeats": repeats})

            is_multi_job = multi_job_var.get() and existing_job is None
            jobs_to_create: list[tuple[str, str, list[dict]]] = []

            if is_multi_job:
                selected_model = selected_model_name
                fam_label = _family_label(selected_model)
                for ds in datasets_config:
                    base_name = f"{ds['name']}_{fam_label}"
                    final_name = self.unique_job_name(base_name)
                    jobs_to_create.append((final_name, final_name, [ds]))
            else:
                existing_index: int | None = None
                if existing_job is not None:
                    try:
                        existing_index = self.job_queue.index(existing_job)
                    except ValueError:
                        existing_index = None

                for idx, queued_job in enumerate(self.job_queue):
                    if existing_index is not None and idx == existing_index:
                        continue
                    if queued_job.get("job_name", "").strip().lower() == job_name.lower():
                        self.messagebox.showerror("Duplicate name", "LoRA name already exists in queue.", parent=dialog)
                        return

                existing_training_name = ""
                if existing_job is not None:
                    existing_training_name = (
                        (existing_job or {}).get("training_name", "").strip()
                        or (existing_job or {}).get("job_name", "").strip()
                    )

                training_name = job_name
                jobs_to_create.append((job_name, training_name, datasets_config))

            created_count = 0

            # ── Async creation loop so UI and loader can animate ──
            loading_overlay = self.tk.Toplevel(dialog)
            loading_overlay.overrideredirect(True)
            loading_overlay.configure(bg=self.bg_panel, bd=2, relief="solid")
            loading_overlay.attributes("-topmost", True)
            
            dialog.update_idletasks()
            ox = dialog.winfo_x() + max(0, (dialog.winfo_width() - 320) // 2)
            oy = dialog.winfo_y() + max(0, (dialog.winfo_height() - 120) // 2)
            loading_overlay.geometry(f"320x120+{ox}+{oy}")
            
            total_jobs = len(jobs_to_create)
            
            _lbl = self.ttk.Label(
                loading_overlay, 
                text=f"Creating Jobs... 0/{total_jobs}", 
                font=("Segoe UI", 11, "bold"), 
                background=self.bg_panel,
                foreground="#ffffff"
            )
            _lbl.pack(pady=(25, 6))

            _pb_frame = self.tk.Frame(loading_overlay, height=8, width=240, bg=self.bg_panel)
            _pb_frame.pack_propagate(False)
            _pb_frame.pack(pady=(0, 12))

            _pb_bg = self.tk.Canvas(
                _pb_frame,
                width=240,
                height=8,
                bg="#1e1e1e",
                bd=0,
                highlightthickness=0,
            )
            _pb_bg.pack(fill="both", expand=True)
            _pb_fill = _pb_bg.create_rectangle(0, 0, 0, 8, fill="#0090d8", width=0)

            def _set_progress(value: int) -> None:
                clamped = max(0, min(value, total_jobs))
                ratio = 0.0 if total_jobs <= 0 else (clamped / total_jobs)
                width = int(240 * ratio)
                _pb_bg.coords(_pb_fill, 0, 0, width, 8)
                _pb_bg.update_idletasks()

            _set_progress(0)
            
            loading_overlay.update()  # Force render immediately 

            jobs_to_process = list(jobs_to_create)
            
            def _process_next_job() -> None:
                nonlocal created_count
                if not jobs_to_process:
                    _finalize_jobs()
                    return

                current_job_name, current_training_name, current_datasets_config = jobs_to_process.pop(0)

                renamed_training_folder = False

                if existing_job is not None and existing_training_name and current_training_name != existing_training_name:
                    source_training_dir = self.training_job_dir_path(existing_training_name).expanduser()
                    target_training_dir = self.training_job_dir_path(current_training_name).expanduser()

                    if target_training_dir.exists():
                        loading_overlay.destroy()
                        self.messagebox.showerror(
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
                            loading_overlay.destroy()
                            self.messagebox.showerror(
                                "Rename failed",
                                f"Could not rename job folder:\n{exc}",
                                parent=dialog,
                            )
                            return

                try:
                    training_dir_path, output_root, created_captions = self.ensure_training_job_structure(
                        training_name=current_training_name,
                        datasets=current_datasets_config,
                        resolution=resolution_value,
                        batch_size=batch_size_value,
                        default_caption_keyword=self.settings_state.get(self.DEFAULT_CAPTION_KEYWORD_KEY, ""),
                        model_name=selected_model_name,
                        create_missing_captions=False,
                    )
                except Exception as exc:
                    loading_overlay.destroy()
                    self.messagebox.showerror("Create job failed", str(exc), parent=dialog)
                    return

                primary_dataset = current_datasets_config[0]["name"]
                tracker_name = self.settings_state.get(self.TRAIN_LOG_TRACKER_NAME_KEY, "").strip() or current_job_name

                new_job = {
                    "id": current_training_name,
                    "dataset_name": primary_dataset,
                    "datasets_json": json.dumps(current_datasets_config),
                    "training_name": current_training_name,
                    "training_dir": str(training_dir_path),
                    "job_name": current_job_name,
                    "model": selected_model_name,
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
                    "ltx_gemma_load_in_4bit": self.bool_to_flag(ltx_gemma_load_in_4bit_var.get()),
                    "sd_unet_lr": sd_unet_lr_value,
                    "sd_text_encoder_lr": sd_text_encoder_lr_value,
                    "enable_compile": self.bool_to_flag(compile_var.get()),
                    "enable_tf32": self.bool_to_flag(tf32_var.get()),
                    "enable_cudnn": self.bool_to_flag(cudnn_var.get()),
                    "enable_fp8": self.bool_to_flag(fp8_var.get()),
                    "enable_gc": self.bool_to_flag(gc_var.get()),
                    "enable_grad_ckpt": self.bool_to_flag(sdxl_grad_ckpt_var.get()),
                    "enable_logging": self.bool_to_flag(self.is_truthy(self.settings_state.get(self.TRAIN_ENABLE_LOGGING_KEY), default=True)),
                    "tracker_name": tracker_name,
                    "stream_output": self.bool_to_flag(self.is_truthy(self.settings_state.get(self.TRAIN_STREAM_TO_LOGGER_KEY), default=False)),
                    "auto_cleanup": (existing_job or {}).get("auto_cleanup", "1"),
                    "hold": (existing_job or {}).get("hold", "1"),
                    "status": "queued",
                }

                new_job["status"] = self.detect_job_status(new_job)

                auto_fixed_elements = 0
                if existing_job is not None:
                    for _attempt in range(5):
                        has_mismatch, source_base = self.detect_job_element_base_mismatch(new_job)
                        if not has_mismatch or source_base is None:
                            break
                        renamed_count, _conflicts = self.rename_job_elements_to_training_name(new_job, source_base)
                        auto_fixed_elements += renamed_count
                        if renamed_count == 0:
                            break
                
                if existing_job is None:
                    self.job_queue.append(new_job)
                else:
                    if existing_index is None:
                        self.job_queue.append(new_job)
                    else:
                        self.job_queue[existing_index] = new_job

                self.save_job_to_disk(new_job)

                generated_training_args = True
                generator = getattr(self, "ensure_job_training_args_toml", None)
                if callable(generator):
                    try:
                        generated_training_args = bool(generator(new_job))
                    except Exception:
                        generated_training_args = False

                if not generated_training_args:
                    self.log(
                        f"[Queue] Note: Could not generate training_args.toml for {current_job_name}; "
                        "it will be generated at launch if missing."
                    )

                ds_names = ", ".join(d["name"] for d in current_datasets_config)
                if existing_job is None:
                    self.log(f"[Queue] Created job: {current_job_name} (datasets: {ds_names}, training: {current_training_name}, captions added: {created_captions})")
                else:
                    if auto_fixed_elements > 0:
                        self.log(f"[Queue] Updated job: {current_job_name} (datasets: {ds_names}, training: {current_training_name}, captions added: {created_captions}, renamed elements: {auto_fixed_elements})")
                    else:
                        self.log(f"[Queue] Updated job: {current_job_name} (datasets: {ds_names}, training: {current_training_name}, captions added: {created_captions})")
                
                created_count += 1
                
                # Update animated UI
                _set_progress(created_count)
                _lbl.configure(text=f"Creating Jobs... {created_count}/{total_jobs}")
                loading_overlay.update_idletasks()
                
                # Re-schedule safely back into the event loop
                dialog.after_idle(_process_next_job)
                
            def _finalize_jobs() -> None:
                loading_overlay.destroy()
                self.save_job_order()
                self.refresh_job_queue_list()
                self.update_start_button_state()
                if existing_job is None:
                    for var in self.vars_by_name.values():
                        var.set(False)
                    for card in self.card_frame_by_name.values():
                        if callable(getattr(self, "dataset_name_from_widget", None)):
                            self.apply_card_style(self.dataset_name_from_widget(card))
                    for name in list(self.card_frame_by_name.keys()):
                        self.apply_card_style(name)
                
                dialog.destroy()

            _process_next_job()

        # ── Footer buttons ─────────────────────────────────────────────────
        buttons = self.ttk.Frame(outer, padding=(0, 10, 0, 0))
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        self.ttk.Label(buttons, textvariable=estimated_epochs_var, style="Dim.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Button(buttons, text="Cancel", command=dialog.destroy, style="CreateJobAction.TButton").grid(row=0, column=1, padx=(0, 8))
        self.ttk.Button(buttons, text="Save" if existing_job is not None else "Create Job", command=create_job, style="CreateJobAction.TButton").grid(row=0, column=2)

        _fit_create_job_dialog_to_content()
        self.center_window(dialog)
        dialog.deiconify()
        self.root.wait_window(dialog)


