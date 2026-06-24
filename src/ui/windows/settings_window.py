from __future__ import annotations
# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false

import os
from pathlib import Path
import json
import re
import shutil
import subprocess
import threading
import time
from typing import Any

from ...runtime_config import RuntimeConfig


class SettingsWindow:
    def __init__(self, **dependencies: object) -> None:
        for name, value in dependencies.items():
            setattr(self, name, value)

    def open(self, required: bool) -> RuntimeConfig | None:
        return self._open_settings_dialog_impl(required)


    def _open_settings_dialog_impl(self, required: bool) -> RuntimeConfig | None:
        backend_dirs = self.configured_backend_dirs()
        current_main_dir = str(backend_dirs["musubi-main"])
        current_ltx_dir = str(backend_dirs["musubi-ltx"])
        current_sd_scripts_dir = str(backend_dirs["sd-scripts"])
        current_trainers_root = str(self.configured_trainers_root())
        current_musubi_python_path = self.resolve_musubi_python(self.Path(current_main_dir).expanduser()) if current_main_dir else None
        current_default_caption_keyword = self.settings_state.get(self.app_settings.DEFAULT_CAPTION_KEYWORD_KEY, "")
        current_enable_training_logging = self.settings_state.get(
            self.app_settings.TRAIN_ENABLE_LOGGING_KEY,
            "1",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_train_log_tracker_name = self.settings_state.get(self.app_settings.TRAIN_LOG_TRACKER_NAME_KEY, "").strip()
        current_train_stream_to_logger = self.settings_state.get(
            self.app_settings.TRAIN_STREAM_TO_LOGGER_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_auto_start_tensorboard = self.settings_state.get(
            self.app_settings.TRAIN_AUTO_START_TENSORBOARD_KEY,
            "0",
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        current_train_save_every_n_steps = str(
            self.get_positive_int_setting(
                self.settings_state,
                self.app_settings.TRAIN_SAVE_EVERY_N_STEPS_KEY,
                self.DEFAULT_SAVE_EVERY_N_STEPS,
                minimum=1,
            )
        )

        def _settings_log(message: str) -> None:
            msg = str(message)
            try:
                self.log(msg)  # type: ignore[name-defined]
            except Exception:
                print(msg)

        result: RuntimeConfig | None = None
        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("Settings")
        dialog.transient(self.root)
        dialog.resizable(True, True)
        dialog.configure(bg=self.bg_panel)
        self.set_dark_title_bar(dialog)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=0)

        notebook = self.ttk.Notebook(dialog)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        general_tab = self.ttk.Frame(notebook, padding=10)
        general_tab.columnconfigure(0, weight=1)
        notebook.add(general_tab, text="  General  ")

        models_tab = self.ttk.Frame(notebook, padding=10)
        models_tab.columnconfigure(0, weight=1)
        notebook.add(models_tab, text="  Models  ")

        footer = self.ttk.Frame(dialog, padding=(10, 8, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        trainers_section = self.ttk.LabelFrame(general_tab, text="Trainers", padding=8)
        trainers_section.grid(row=0, column=0, sticky="ew")
        trainers_section.columnconfigure(1, weight=1)

        captions_section = self.ttk.LabelFrame(general_tab, text="Captions", padding=8)
        captions_section.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        captions_section.columnconfigure(1, weight=1)

        advanced_section = self.ttk.LabelFrame(general_tab, text="Training", padding=8)
        advanced_section.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        advanced_section.columnconfigure(0, weight=0)
        advanced_section.columnconfigure(1, weight=1)
        advanced_section.columnconfigure(2, weight=0)
        advanced_section.columnconfigure(3, weight=1)

        # ── Models tab ────────────────────────────────────────────────────
        model_loc_frame = self.ttk.Frame(models_tab)
        model_loc_frame.grid(row=0, column=0, sticky="ew")
        model_location_var = self.tk.StringVar(
            value=self.settings_state.get(self.app_settings.MODEL_DOWNLOAD_LOCATION_KEY, self.DOWNLOAD_LOCATION_MODELS_FOLDER)
        )
        self.ttk.Label(model_loc_frame, text="Download location:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Combobox(
            model_loc_frame,
            textvariable=model_location_var,
            values=list(self.DOWNLOAD_LOCATIONS),
            state="readonly",
            width=22,
        ).grid(row=0, column=1, sticky="w")
        hf_token_var = self.tk.StringVar(value=self.settings_state.get(self.app_settings.HF_TOKEN_KEY, ""))
        self.ttk.Label(model_loc_frame, text="HuggingFace token:").grid(row=0, column=2, sticky="w", padx=(24, 8))
        self.ttk.Entry(model_loc_frame, textvariable=hf_token_var, show="*", width=36, style="Flat.TEntry").grid(row=0, column=3, sticky="ew")
        model_loc_frame.columnconfigure(3, weight=1)

        # ── ComfyUI path + scan ───────────────────────────────────────────
        _raw_extra = self.settings_state.get(self.app_settings.EXTRA_SEARCH_PATHS_KEY, "")
        import json as _json_extra
        try:
            _extra_paths_list: list[str] = _json_extra.loads(_raw_extra) if _raw_extra else []
        except Exception:
            _extra_paths_list = []

        extra_paths_frame = self.ttk.Frame(models_tab)
        extra_paths_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        extra_paths_frame.columnconfigure(1, weight=1)
        self.ttk.Label(extra_paths_frame, text="ComfyUI models path:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        extra_path_var = self.tk.StringVar(value=_extra_paths_list[0] if _extra_paths_list else "")
        self.ttk.Entry(extra_paths_frame, textvariable=extra_path_var, style="Flat.TEntry").grid(row=0, column=1, sticky="ew")

        def _browse_extra_path() -> None:
            picked = self.filedialog.askdirectory(
                parent=dialog,
                title="Select ComfyUI models folder",
                initialdir=extra_path_var.get().strip() or str(self.Path.home()),
            )
            if picked:
                extra_path_var.set(picked)

        def _scan_all_sources() -> None:
            """Scan Models folder, HF cache, and optionally the ComfyUI path for any missing files."""
            ws_root = self.download_workspace_root()
            comfy_dir = extra_path_var.get().strip()
            extra = [comfy_dir] if comfy_dir and self.Path(comfy_dir).is_dir() else []
            found_count = 0
            for mn, comps in self.DOWNLOAD_MODELS.items():
                for comp in comps:
                    existing = pending_model_paths.get(mn, {}).get(comp, "").strip()
                    if _is_existing_file_path(existing):
                        continue  # already set and valid
                    hit = self.find_component(mn, comp, ws_root, extra or None)
                    if hit is not None:
                        pending_model_paths.setdefault(mn, {})[comp] = str(hit)
                        found_count += 1
            # Refresh all entry StringVars and status labels
            for mn in self.DOWNLOAD_MODELS:
                for comp, cpv in _comp_path_vars_all.get(mn, {}).items():
                    new_val = pending_model_paths.get(mn, {}).get(comp, "")
                    if cpv.get() != new_val:
                        cpv.set(new_val)
                _refresh_status(mn)
            if found_count:
                self.messagebox.showinfo("Scan complete", f"Found {found_count} new file(s). Click Save to apply.", parent=dialog)
            else:
                self.messagebox.showinfo("Scan complete", "No new files found.", parent=dialog)

        self.ttk.Button(extra_paths_frame, text="Browse…", command=_browse_extra_path).grid(row=0, column=2, padx=(6, 0))
        self.ttk.Button(extra_paths_frame, text="Scan for models", command=_scan_all_sources).grid(row=0, column=3, padx=(6, 0))

        # Registry of all component StringVars so _scan_extra_path can update them
        _comp_path_vars_all: dict[str, dict[str, Any]] = {}

        models_canvas = self.tk.Canvas(models_tab, bg=self.bg_panel, highlightthickness=0)
        models_scrollbar = self.ttk.Scrollbar(models_tab, orient="vertical", command=models_canvas.yview)
        models_canvas.configure(yscrollcommand=models_scrollbar.set)
        models_canvas.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        models_scrollbar.grid(row=2, column=1, sticky="ns", pady=(10, 0))
        models_tab.rowconfigure(2, weight=1)

        models_inner = self.ttk.Frame(models_canvas)
        models_inner.columnconfigure(0, weight=1)
        _mw_id = models_canvas.create_window((0, 0), window=models_inner, anchor="nw")

        def _models_canvas_has_scrollable_content() -> bool:
            try:
                region = models_canvas.cget("scrollregion")
                if not region:
                    return False
                parts = [float(v) for v in str(region).split()]
                if len(parts) != 4:
                    return False
                content_h = max(0.0, parts[3] - parts[1])
                viewport_h = float(max(1, models_canvas.winfo_height()))
                return content_h > (viewport_h + 1.0)
            except Exception:
                return False

        def _clamp_models_scroll() -> None:
            if not _models_canvas_has_scrollable_content():
                try:
                    models_canvas.yview_moveto(0.0)
                except Exception:
                    pass

        def _on_models_mousewheel(event: Any) -> str:
            if not _models_canvas_has_scrollable_content():
                _clamp_models_scroll()
                return "break"
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta != 0:
                models_canvas.yview_scroll(delta, "units")
            return "break"

        def _on_models_inner_configure(event: Any) -> None:
            models_canvas.configure(scrollregion=models_canvas.bbox("all"))
            _clamp_models_scroll()

        def _on_models_canvas_configure(event: Any) -> None:
            models_canvas.itemconfig(_mw_id, width=event.width)

        models_inner.bind("<Configure>", _on_models_inner_configure)
        models_canvas.bind("<Configure>", _on_models_canvas_configure)
        models_canvas.bind("<MouseWheel>", _on_models_mousewheel)

        def _bind_mousewheel(widget: Any) -> None:
            widget.bind("<MouseWheel>", _on_models_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        selected_trainers_root = current_trainers_root
        selected_musubi_main_path = current_main_dir
        selected_musubi_ltx_path = current_ltx_dir
        selected_sd_scripts_path = current_sd_scripts_dir
        selected_musubi_python = str(current_musubi_python_path) if current_musubi_python_path is not None else ""

        # pending_model_paths: model_name → {component: path_str}
        import json as _json_settings
        _raw_model_paths = self.settings_state.get(self.app_settings.MODEL_PATHS_KEY, "")
        try:
            pending_model_paths: dict[str, dict[str, str]] = _json_settings.loads(_raw_model_paths) if _raw_model_paths else {}
        except Exception:
            pending_model_paths = {}
        preset_none_label = "---------"
        _raw_preferred_presets = self.settings_state.get(self.app_settings.PREFERRED_PRESETS_BY_FAMILY_KEY, "")
        try:
            _preferred_presets_loaded = _json_settings.loads(_raw_preferred_presets) if _raw_preferred_presets else {}
        except Exception:
            _preferred_presets_loaded = {}
        preferred_preset_by_family: dict[str, str] = {}
        if isinstance(_preferred_presets_loaded, dict):
            preferred_preset_by_family = {
                str(k): str(v)
                for k, v in _preferred_presets_loaded.items()
                if isinstance(k, str) and isinstance(v, str)
            }

        def _preset_names_for_family_settings(family_name: str) -> list[str]:
            names: set[str] = set()
            presets_dir = self.download_workspace_root() / "Presets"
            if not presets_dir.exists() or not presets_dir.is_dir():
                return []
            for path in sorted(presets_dir.glob(f"*{self.JOB_PRESET_FILE_SUFFIX}"), key=lambda p: p.name.casefold()):
                try:
                    payload = self.json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                payload_family = str(payload.get("family", "")).strip()
                preset_name = str(payload.get("name", "")).strip()
                if payload_family == family_name and preset_name:
                    names.add(preset_name)
            return sorted(names, key=str.casefold)

        preferred_preset_vars: dict[str, Any] = {}
        trainers_root_var = self.tk.StringVar(value=selected_trainers_root)
        musubi_main_display_var = self.tk.StringVar(value=selected_musubi_main_path if selected_musubi_main_path else "(none)")
        musubi_ltx_display_var = self.tk.StringVar(value=selected_musubi_ltx_path if selected_musubi_ltx_path else "(none)")
        sd_scripts_display_var = self.tk.StringVar(value=selected_sd_scripts_path if selected_sd_scripts_path else "(none)")
        musubi_main_status_var = self.tk.StringVar(value="")
        musubi_ltx_status_var = self.tk.StringVar(value="")
        sd_scripts_status_var = self.tk.StringVar(value="")
        musubi_main_action_var = self.tk.StringVar(value="Auto Download")
        musubi_ltx_action_var = self.tk.StringVar(value="Auto Download")
        sd_scripts_action_var = self.tk.StringVar(value="Auto Download")
        default_caption_keyword_var = self.tk.StringVar(value=current_default_caption_keyword)
        enable_training_logging_var = self.tk.BooleanVar(value=current_enable_training_logging)
        train_log_tracker_name_var = self.tk.StringVar(value=current_train_log_tracker_name)
        stream_to_logger_var = self.tk.BooleanVar(value=current_train_stream_to_logger)
        auto_start_tensorboard_var = self.tk.BooleanVar(value=current_auto_start_tensorboard)
        train_save_every_default_var = self.tk.StringVar(value=current_train_save_every_n_steps)

        self.ttk.Label(trainers_section, text="Trainers root:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Label(
            trainers_section,
            textvariable=trainers_root_var,
            anchor="w",
            style="PathDisplay.TLabel",
            padding=(6, 4),
        ).grid(row=0, column=1, sticky="ew")
        self.ttk.Button(trainers_section, text="Browse Root", command=lambda: browse_trainers_root()).grid(
            row=0,
            column=2,
            padx=(8, 0),
        )

        self.ttk.Label(trainers_section, text="Musubi Main:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.ttk.Label(
            trainers_section,
            textvariable=musubi_main_display_var,
            anchor="w",
            style="PathDisplay.TLabel",
            padding=(6, 4),
        ).grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.ttk.Button(trainers_section, text="Browse", command=lambda: browse_backend("musubi-main")).grid(
            row=1,
            column=2,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Button(trainers_section, textvariable=musubi_main_action_var, command=lambda: auto_download_backend("musubi-main")).grid(
            row=1,
            column=3,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Label(trainers_section, textvariable=musubi_main_status_var).grid(row=2, column=1, sticky="w")
        self.ttk.Label(
            trainers_section,
            text="Used for: FLUX2, KREA2, QWEN, ZIMAGE, WAN",
            foreground=self.fg_muted,
        ).grid(row=2, column=2, columnspan=2, sticky="w")

        self.ttk.Label(trainers_section, text="Musubi LTX:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.ttk.Label(
            trainers_section,
            textvariable=musubi_ltx_display_var,
            anchor="w",
            style="PathDisplay.TLabel",
            padding=(6, 4),
        ).grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.ttk.Button(trainers_section, text="Browse", command=lambda: browse_backend("musubi-ltx")).grid(
            row=3,
            column=2,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Button(trainers_section, textvariable=musubi_ltx_action_var, command=lambda: auto_download_backend("musubi-ltx")).grid(
            row=3,
            column=3,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Label(trainers_section, textvariable=musubi_ltx_status_var).grid(row=4, column=1, sticky="w")
        self.ttk.Label(
            trainers_section,
            text="Used for: LTX",
            foreground=self.fg_muted,
        ).grid(row=4, column=2, columnspan=2, sticky="w")

        self.ttk.Label(trainers_section, text="sd-scripts:").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.ttk.Label(
            trainers_section,
            textvariable=sd_scripts_display_var,
            anchor="w",
            style="PathDisplay.TLabel",
            padding=(6, 4),
        ).grid(row=5, column=1, sticky="ew", pady=(8, 0))
        self.ttk.Button(trainers_section, text="Browse", command=lambda: browse_backend("sd-scripts")).grid(
            row=5,
            column=2,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Button(trainers_section, textvariable=sd_scripts_action_var, command=lambda: auto_download_backend("sd-scripts")).grid(
            row=5,
            column=3,
            padx=(8, 0),
            pady=(8, 0),
        )
        self.ttk.Label(trainers_section, textvariable=sd_scripts_status_var).grid(row=6, column=1, sticky="w")
        self.ttk.Label(
            trainers_section,
            text="Used for: ANIMA, FLUX, SDXL",
            foreground=self.fg_muted,
        ).grid(row=6, column=2, columnspan=2, sticky="w")

        self.ttk.Label(
            trainers_section,
            text="Model families are shown based on backend availability. Python interpreter is managed by this app.",
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.ttk.Label(captions_section, text="Default caption keyword:").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        default_caption_keyword_entry = self.ttk.Entry(
            captions_section,
            textvariable=default_caption_keyword_var,
            style="Flat.TEntry",
        )
        default_caption_keyword_entry.grid(row=0, column=1, sticky="ew")
        self.ttk.Label(captions_section, text="Leave blank to create empty .txt captions.").grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

        training_defaults_section = self.ttk.LabelFrame(advanced_section, text="Training defaults", padding=8)
        training_defaults_section.grid(row=0, column=0, columnspan=4, sticky="ew")
        training_defaults_section.columnconfigure(1, weight=1)
        self.ttk.Label(training_defaults_section, text="Save every N steps:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Entry(training_defaults_section, textvariable=train_save_every_default_var, style="Flat.TEntry").grid(
            row=0,
            column=1,
            sticky="w",
        )

        logging_section = self.ttk.LabelFrame(advanced_section, text="Logging & metadata", padding=8)
        logging_section.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        logging_section.columnconfigure(0, weight=0)
        logging_section.columnconfigure(1, weight=1)
        logging_section.columnconfigure(2, weight=0)
        logging_section.columnconfigure(3, weight=1)

        self.ttk.Checkbutton(
            logging_section,
            text="Enable TensorBoard",
            variable=enable_training_logging_var,
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        self.ttk.Label(logging_section, text="Tracker name:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        self.ttk.Entry(logging_section, textvariable=train_log_tracker_name_var, style="Flat.TEntry").grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )
        self.ttk.Checkbutton(
            logging_section,
            text="Show full training output in app console",
            variable=stream_to_logger_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(
            logging_section,
            text="Keep TensorBoard running in background",
            variable=auto_start_tensorboard_var,
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.ttk.Label(
            logging_section,
            text="Logs are stored per job under each Training/<job>/logs folder and can be viewed via TensorBoard.",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # ── Model family sections ─────────────────────────────────────────
        _status_vars: dict[str, Any] = {}
        _component_label_widgets: dict[tuple[str, str], Any] = {}

        _COMPONENT_LABELS: dict[str, str] = {
            "dit": "Model",
            "vae": "VAE",
            "text_encoder": "Text Encoder",
            "t5": "T5",
            "clip": "CLIP",
        }

        def _normalize_candidate_path(path_value: str) -> str:
            raw = str(path_value or "").strip()
            if not raw:
                return ""
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1].strip()
            return raw

        def _is_existing_file_path(path_value: str) -> bool:
            candidate_raw = _normalize_candidate_path(path_value)
            if not candidate_raw:
                return False
            expanded = os.path.expandvars(candidate_raw)
            p = self.Path(expanded).expanduser()
            if p.is_file():
                return True
            if not p.is_absolute():
                ws_candidate = self.download_workspace_root() / p
                if ws_candidate.is_file():
                    return True
            return False

        def _component_display_label(model_name: str, component: str) -> str:
            comp_label = _COMPONENT_LABELS.get(component, component.capitalize())
            comp_info = self.DOWNLOAD_MODELS.get(model_name, {}).get(component, {})
            folder_name = comp_info.get("folder_name", "")
            friendly = self.DOWNLOAD_COMPONENT_FRIENDLY_NAMES.get(folder_name, "")
            return f"{comp_label} ({friendly})" if friendly else comp_label

        def _model_found_missing_labels(model_name: str) -> tuple[list[str], list[str]]:
            components = list(self.DOWNLOAD_MODELS.get(model_name, {}).keys())
            stored = pending_model_paths.get(model_name, {})
            found_labels: list[str] = []
            missing_labels: list[str] = []
            for comp in components:
                label = _component_display_label(model_name, comp)
                comp_path = str(stored.get(comp, "") or "").strip()
                if _is_existing_file_path(comp_path):
                    found_labels.append(label)
                else:
                    missing_labels.append(label)
            return found_labels, missing_labels

        def _model_status_str(model_name: str) -> str:
            components = list(self.DOWNLOAD_MODELS.get(model_name, {}).keys())
            if not components:
                return "Unknown"
            found_labels, _missing_labels = _model_found_missing_labels(model_name)
            found = len(found_labels)
            if found == 0:
                return "Not configured"
            if found < len(components):
                return f"Partial ({found}/{len(components)})"
            return "✓ Ready"

        def _apply_component_label_color(model_name: str, component: str) -> None:
            lbl = _component_label_widgets.get((model_name, component))
            if lbl is None:
                return
            comp_path = str(pending_model_paths.get(model_name, {}).get(component, "") or "").strip()
            if _is_existing_file_path(comp_path):
                lbl.configure(foreground="#e6e6e6")
            else:
                lbl.configure(foreground="#f0b429")

        def _refresh_status(model_name: str) -> None:
            sv = _status_vars.get(model_name)
            if sv:
                sv.set(_model_status_str(model_name))
            for comp_name in self.DOWNLOAD_MODELS.get(model_name, {}).keys():
                _apply_component_label_color(model_name, comp_name)

        def _apply_status_color(lbl: Any, sv: Any, *_a: object) -> None:
            val = sv.get()
            if val.startswith("✓"):
                lbl.configure(foreground="#6fcf6f")
            elif val.startswith("Partial"):
                lbl.configure(foreground="#f0b429")
            else:
                lbl.configure(foreground=self.fg_muted)

        def _make_family_section(parent: Any, family_name: str, model_names: list[str], row: int, expanded: bool) -> None:
            section_frame = self.ttk.Frame(parent)
            section_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            section_frame.columnconfigure(0, weight=1)

            fam_expanded_var = self.tk.BooleanVar(value=expanded)

            fam_header_btn = self.ttk.Button(
                section_frame,
                text=f"{'▼' if expanded else '▶'}  {family_name}",
                style="FamilyHeader.TButton",
                command=lambda: _toggle_family(fam_header_btn, fam_body, fam_expanded_var, family_name),
            )
            fam_header_btn.grid(row=0, column=0, sticky="ew")

            fam_body = self.ttk.Frame(section_frame, padding=(4, 2, 4, 2))
            fam_body.columnconfigure(0, weight=1)
            if expanded:
                fam_body.grid(row=1, column=0, sticky="ew")

            family_preset_names = _preset_names_for_family_settings(family_name)
            preferred_initial = preferred_preset_by_family.get(family_name, "").strip()
            if preferred_initial and preferred_initial not in family_preset_names:
                preferred_initial = ""
            preferred_var = self.tk.StringVar(value=(preferred_initial or preset_none_label))
            preferred_preset_vars[family_name] = preferred_var

            preset_row = self.ttk.Frame(fam_body)
            preset_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
            preset_row.columnconfigure(1, weight=1)
            self.ttk.Label(preset_row, text="Preferred Preset:").grid(row=0, column=0, sticky="w", padx=(22, 8))
            self.ttk.Combobox(
                preset_row,
                textvariable=preferred_var,
                values=[preset_none_label] + family_preset_names,
                state="readonly",
                width=28,
            ).grid(row=0, column=1, sticky="w")

            for r, mn in enumerate(model_names, start=1):
                display_name = self.DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn)
                sv = self.tk.StringVar(value=_model_status_str(mn))
                _status_vars[mn] = sv

                model_block = self.ttk.Frame(fam_body)
                model_block.grid(row=r, column=0, sticky="ew", pady=(1, 0))
                model_block.columnconfigure(0, weight=1)

                # ─ Model header row ───────────────────────────────────────
                hdr = self.ttk.Frame(model_block)
                hdr.grid(row=0, column=0, sticky="ew")
                hdr.columnconfigure(0, weight=1)

                detail_expanded_var = self.tk.BooleanVar(value=False)
                detail_frame = self.ttk.Frame(model_block, padding=(20, 2, 0, 2))
                detail_frame.columnconfigure(1, weight=1)
                # detail_frame is NOT gridded yet (hidden by default)

                expand_btn = self.ttk.Button(
                    hdr,
                    text=f"▶  {display_name}",
                    style="FamilyHeader.TButton",
                )
                expand_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

                status_lbl = self.ttk.Label(hdr, textvariable=sv, anchor="w", width=18)
                status_lbl.grid(row=0, column=1, padx=(8, 0), sticky="w")
                sv.trace_add("write", lambda *a, lbl=status_lbl, s=sv: _apply_status_color(lbl, s))
                _apply_status_color(status_lbl, sv)

                self.ttk.Button(
                    hdr, text="Auto-download",
                    command=lambda mn=mn: _auto_download_model(mn),
                ).grid(row=0, column=2, padx=(8, 0))

                # ─ Detail rows (component paths) ──────────────────────────
                components = list(self.DOWNLOAD_MODELS.get(mn, {}).keys())
                _comp_path_vars: dict[str, Any] = {}

                for cr, comp in enumerate(components):
                    comp_label = _COMPONENT_LABELS.get(comp, comp.capitalize())
                    # Derive the friendly name for this specific component slot
                    comp_info = self.DOWNLOAD_MODELS.get(mn, {}).get(comp, {})
                    folder_name = comp_info.get("folder_name", "")
                    friendly = self.DOWNLOAD_COMPONENT_FRIENDLY_NAMES.get(folder_name, "")
                    label_text = f"{comp_label} ({friendly}):" if friendly else f"{comp_label}:"

                    stored_path = pending_model_paths.get(mn, {}).get(comp, "")
                    cpv = self.tk.StringVar(value=stored_path)
                    _comp_path_vars[comp] = cpv

                    comp_lbl = self.ttk.Label(detail_frame, text=label_text, anchor="e")
                    comp_lbl.grid(
                        row=cr, column=0, sticky="e", padx=(0, 6), pady=1
                    )
                    _component_label_widgets[(mn, comp)] = comp_lbl
                    path_entry = self.ttk.Entry(
                        detail_frame, textvariable=cpv, style="Flat.TEntry",
                    )
                    path_entry.grid(row=cr, column=1, sticky="ew", pady=1)

                    def _save_path(mn: str = mn, comp: str = comp, cpv: Any = cpv) -> None:
                        val = cpv.get().strip()
                        if val:
                            pending_model_paths.setdefault(mn, {})[comp] = val
                        elif mn in pending_model_paths and comp in pending_model_paths[mn]:
                            del pending_model_paths[mn][comp]
                        _refresh_status(mn)

                    path_entry.bind("<FocusOut>", lambda e, mn=mn, comp=comp, cpv=cpv: _save_path(mn, comp, cpv))
                    path_entry.bind("<Return>", lambda e, mn=mn, comp=comp, cpv=cpv: _save_path(mn, comp, cpv))

                    def _browse_comp(mn: str = mn, comp: str = comp, cpv: Any = cpv, friendly: str = friendly) -> None:
                        cur = cpv.get().strip()
                        initial = str(self.Path(cur).parent) if cur and self.Path(cur).parent.exists() else str(self.default_models_dir if self.default_models_dir.exists() else self.Path.home())
                        title_label = friendly or _COMPONENT_LABELS.get(comp, comp)
                        picked = self.filedialog.askopenfilename(
                            parent=dialog,
                            title=f"Select {title_label} for {self.DOWNLOAD_MODEL_DISPLAY_NAMES.get(mn, mn)}",
                            initialdir=initial,
                            filetypes=[("Safetensors / PTH", "*.safetensors *.pth"), ("All files", "*.*")],
                        )
                        if picked:
                            pending_model_paths.setdefault(mn, {})[comp] = picked
                            cpv.set(picked)
                            _refresh_status(mn)

                    self.ttk.Button(detail_frame, text="Browse", command=_browse_comp).grid(
                        row=cr, column=2, padx=(6, 0), pady=1
                    )

                # Register vars so _scan_extra_path can update them
                _comp_path_vars_all[mn] = _comp_path_vars
                _refresh_status(mn)

                def _toggle_detail(
                    btn: Any = expand_btn,
                    det: Any = detail_frame,
                    var: Any = detail_expanded_var,
                    dn: str = display_name,
                ) -> None:
                    if var.get():
                        det.grid_remove()
                        var.set(False)
                        btn.configure(text=f"▶  {dn}")
                    else:
                        det.grid(row=1, column=0, sticky="ew")
                        var.set(True)
                        btn.configure(text=f"▼  {dn}")
                    _clamp_models_scroll()

                expand_btn.configure(command=_toggle_detail)
                _bind_mousewheel(hdr)
                _bind_mousewheel(detail_frame)

            _bind_mousewheel(fam_body)

        def _toggle_family(
            btn: Any,
            body: Any,
            var: Any,
            family_name: str,
        ) -> None:
            if var.get():
                body.grid_remove()
                var.set(False)
                btn.configure(text=f"▶  {family_name}")
            else:
                body.grid(row=1, column=0, sticky="ew")
                var.set(True)
                btn.configure(text=f"▼  {family_name}")
            _clamp_models_scroll()

        _family_row = 0
        for _fam_name, _fam_models in self.DOWNLOAD_MODEL_FAMILIES.items():
            _make_family_section(models_inner, _fam_name, _fam_models, _family_row, expanded=(_fam_name == "FLUX.2"))
            _family_row += 1

        _bind_mousewheel(models_inner)

        # ── Auto-download handler ─────────────────────────────────────────
        def _auto_download_model(model_name: str) -> None:
            location = model_location_var.get()
            ws_root = self.download_workspace_root()
            hf_token = hf_token_var.get().strip() or None
            components = list(self.DOWNLOAD_MODELS.get(model_name, {}).keys())

            missing = [c for c in components if self.find_component(model_name, c, ws_root) is None]
            # Also check pending_model_paths
            stored = pending_model_paths.get(model_name, {})
            missing = [c for c in missing if not _is_existing_file_path(str(stored.get(c, "") or ""))]

            if not missing:
                self.messagebox.showinfo(
                    "Models found",
                    f"All '{model_name}' files are already available.",
                    parent=dialog,
                )
                _refresh_status(model_name)
                return

            confirmed = self.messagebox.askyesno(
                "Download models",
                f"The following files for '{model_name}' were not found:\n\n"
                + "\n".join(f"  \u2022 {m}" for m in missing)
                + f"\n\nDownload to: {location}?\n\nThis may take a while.",
                parent=dialog,
            )
            if not confirmed:
                return

            component_units: dict[str, int] = {}
            total_units = 0
            for comp in missing:
                comp_info = self.DOWNLOAD_MODELS.get(model_name, {}).get(comp, {})
                units = int(comp_info.get("shard_count", 1)) if comp_info.get("shards") else 1
                units = units if units > 0 else 1
                component_units[comp] = units
                total_units += units
            if total_units <= 0:
                total_units = max(1, len(missing))

            progress_state: dict[str, float] = {"completed_units": 0.0}
            progress_label_var = self.tk.StringVar(value=f"Preparing download for {model_name}...")
            progress_pct_var = self.tk.StringVar(value="0%")
            cancel_download_event = self.threading.Event()
            cancelled_holder: dict[str, bool] = {"value": False}
            active_proc_holder: dict[str, Any] = {"proc": None}
            component_runtime: dict[str, Any] = {
                "name": None,
                "started_at": 0.0,
                "last_pct": 0.0,
                "last_pct_at": 0.0,
            }

            progress_dialog = self.tk.Toplevel(dialog)
            progress_dialog.title("Downloading models")
            progress_dialog.transient(dialog)
            progress_dialog.resizable(False, False)
            progress_dialog.configure(bg=self.bg_panel)
            self.set_dark_title_bar(progress_dialog)

            def _safe_after_ui(callback: Any) -> None:
                try:
                    if dialog.winfo_exists():
                        dialog.after(0, callback)
                except Exception:
                    pass

            def _request_cancel_download() -> None:
                if cancel_download_event.is_set():
                    return
                cancel_download_event.set()
                cancelled_holder["value"] = True
                progress_label_var.set(f"Cancelling download for {model_name}...")
                self.log(f"━━━ Cancelling download: {model_name} ━━━")
                proc = active_proc_holder.get("proc")
                if proc is not None:
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                    except Exception:
                        pass
                try:
                    if progress_dialog.winfo_exists():
                        progress_dialog.destroy()
                except Exception:
                    pass

            progress_dialog.protocol("WM_DELETE_WINDOW", _request_cancel_download)

            progress_frame = self.ttk.Frame(progress_dialog, padding=12)
            progress_frame.grid(row=0, column=0, sticky="nsew")
            progress_frame.columnconfigure(0, weight=1)
            self.ttk.Label(progress_frame, textvariable=progress_label_var, anchor="w").grid(row=0, column=0, sticky="ew")
            progress_bar = self.ttk.Progressbar(
                progress_frame,
                mode="determinate",
                maximum=float(total_units),
                value=0.0,
                length=420,
            )
            progress_bar.grid(row=1, column=0, sticky="ew", pady=(8, 2))
            self.ttk.Label(progress_frame, textvariable=progress_pct_var, anchor="e").grid(row=2, column=0, sticky="e")
            progress_dialog.update_idletasks()
            self.center_window(progress_dialog)

            def _set_progress_ui(label_text: str, value_units: float) -> None:
                clamped = max(0.0, min(float(total_units), value_units))
                progress_label_var.set(label_text)
                progress_bar.configure(value=clamped)
                pct = int(round((clamped / float(total_units)) * 100.0)) if total_units else 0
                progress_pct_var.set(f"{pct}%")

            def _set_component_progress(comp: str, pct: float) -> None:
                units = float(component_units.get(comp, 1))
                bounded_pct = max(0.0, min(100.0, pct))
                value_units = progress_state["completed_units"] + ((bounded_pct / 100.0) * units)
                component_runtime["last_pct"] = bounded_pct
                component_runtime["last_pct_at"] = time.time()
                _set_progress_ui(f"Downloading {model_name}: {comp} ({int(round(bounded_pct))}%)", value_units)

            def _complete_component_progress(comp: str) -> None:
                progress_state["completed_units"] += float(component_units.get(comp, 1))
                if component_runtime.get("name") == comp:
                    component_runtime["name"] = None
                _set_progress_ui(f"Completed {model_name}: {comp}", progress_state["completed_units"])

            def _progress_tick() -> None:
                if cancel_download_event.is_set():
                    return
                try:
                    if not progress_dialog.winfo_exists():
                        return
                except Exception:
                    return

                active_component = component_runtime.get("name")
                if active_component:
                    elapsed = max(0.0, time.time() - float(component_runtime.get("started_at", 0.0) or 0.0))
                    inferred_pct = min(95.0, elapsed * 1.2)
                    known_pct = float(component_runtime.get("last_pct", 0.0) or 0.0)
                    show_pct = max(known_pct, inferred_pct)
                    units = float(component_units.get(active_component, 1))
                    value_units = progress_state["completed_units"] + ((show_pct / 100.0) * units)
                    _set_progress_ui(
                        f"Downloading {model_name}: {active_component} ({int(round(show_pct))}%)",
                        value_units,
                    )

                try:
                    progress_dialog.after(700, _progress_tick)
                except Exception:
                    pass

            progress_dialog.after(700, _progress_tick)

            # Resolve which Python to use — prefer the configured Musubi-Tuner venv
            # so that huggingface_hub is available and the process has a real stdout.
            python_exe = selected_musubi_python
            if not python_exe:
                self.messagebox.showerror(
                    "Python not found",
                    "App venv Python was not found. Run Setup.bat and try again.",
                    parent=dialog,
                )
                return
            cli_script = str(self.download_cli_script_path)

            error_holder: list[str] = []
            result_holder: dict[str, object] = {}

            self.log(f"━━━ Downloading {model_name} ({', '.join(missing)}) ━━━")

            def _do_download() -> None:
                for comp in missing:
                    if cancel_download_event.is_set():
                        cancelled_holder["value"] = True
                        break
                    cmd = [
                        python_exe, cli_script,
                        "--model", model_name,
                        "--component", comp,
                        "--ws-root", str(ws_root) if ws_root else "",
                        "--location", location,
                    ]
                    if hf_token:
                        cmd += ["--token", hf_token]

                    self.log(f"  ↓ {comp}…")
                    _safe_after_ui(lambda c=comp: _set_progress_ui(f"Starting {model_name}: {c}...", progress_state["completed_units"]))
                    component_runtime["name"] = comp
                    component_runtime["started_at"] = time.time()
                    component_runtime["last_pct"] = 0.0
                    component_runtime["last_pct_at"] = 0.0
                    _progress_stop = self.threading.Event()

                    def _progress_heartbeat(component: str = comp) -> None:
                        started = time.time()
                        while not _progress_stop.wait(12.0):
                            elapsed = int(time.time() - started)
                            self.log(f"    ... {component} download in progress ({elapsed}s elapsed)")

                    heartbeat_thread = self.threading.Thread(target=_progress_heartbeat, daemon=True)
                    heartbeat_thread.start()
                    try:
                        proc = self.subprocess.Popen(
                            cmd,
                            stdout=self.subprocess.PIPE,
                            stderr=self.subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        )
                        active_proc_holder["proc"] = proc
                        comp_result: str | None = None
                        assert proc.stdout is not None
                        # Read with \r awareness so tqdm in-place updates stream through.
                        _buf = ""
                        while True:
                            if cancel_download_event.is_set():
                                cancelled_holder["value"] = True
                                try:
                                    if proc.poll() is None:
                                        proc.terminate()
                                except Exception:
                                    pass
                                break
                            chunk = proc.stdout.read(256)
                            if not chunk:
                                break
                            _buf += chunk
                            while True:
                                nl = _buf.find("\n")
                                cr = _buf.find("\r")
                                if nl == -1 and cr == -1:
                                    break
                                if cr != -1 and (nl == -1 or cr < nl):
                                    line = _buf[:cr]
                                    _buf = _buf[cr + 1:]
                                    if line.startswith("RESULT:"):
                                        comp_result = line[7:]
                                    elif line.strip():
                                        trimmed = line.strip()
                                        self.log(f"    {trimmed}")
                                        match = self.re.search(r"(\d{1,3})%", trimmed)
                                        if match:
                                            pct = float(match.group(1))
                                            _safe_after_ui(lambda c=comp, p=pct: _set_component_progress(c, p))
                                else:
                                    line = _buf[:nl].rstrip("\r")
                                    _buf = _buf[nl + 1:]
                                    if line.startswith("RESULT:"):
                                        comp_result = line[7:]
                                    elif line.strip():
                                        trimmed = line.strip()
                                        self.log(f"    {trimmed}")
                                        match = self.re.search(r"(\d{1,3})%", trimmed)
                                        if match:
                                            pct = float(match.group(1))
                                            _safe_after_ui(lambda c=comp, p=pct: _set_component_progress(c, p))
                        if _buf.strip():
                            self.log(f"    {_buf.strip()}")
                        proc.wait()
                        if cancel_download_event.is_set() or cancelled_holder["value"]:
                            cancelled_holder["value"] = True
                            break
                        if proc.returncode != 0:
                            error_holder.append(
                                f"Download of '{comp}' failed (exit {proc.returncode})"
                            )
                            break
                        self.log(f"  ✓ {comp} complete")
                        _safe_after_ui(lambda c=comp: _complete_component_progress(c))
                        if comp_result:
                            result_holder[comp] = comp_result
                            # Persist each component immediately after it downloads.
                            def _save_comp(c: str = comp, p: str = comp_result) -> None:
                                import json as _json_save
                                pending_model_paths.setdefault(model_name, {})[c] = p
                                self.settings_state[self.app_settings.MODEL_PATHS_KEY] = _json_save.dumps(pending_model_paths)
                                self.app_settings.save_settings(self.settings_state)
                                _refresh_status(model_name)
                                # Update entry box if the registry has a StringVar for it
                                sv = _comp_path_vars_all.get(model_name, {}).get(c)
                                if sv is not None:
                                    sv.set(p)
                            _safe_after_ui(_save_comp)
                    except Exception as exc:
                        error_holder.append(str(exc))
                        break
                    finally:
                        active_proc_holder["proc"] = None
                        _progress_stop.set()

                _safe_after_ui(_on_dl_done)

            def _on_dl_done() -> None:
                if progress_dialog.winfo_exists():
                    progress_dialog.destroy()
                if cancelled_holder["value"]:
                    self.log(f"━━━ Cancelled: {model_name} ━━━")
                    self.messagebox.showinfo(
                        "Download cancelled",
                        f"Download for '{model_name}' was cancelled.",
                        parent=dialog,
                    )
                    return
                if error_holder:
                    self.log(f"[ERROR] {error_holder[0]}")
                    self.messagebox.showerror("Download failed", error_holder[0], parent=dialog)
                    return
                _refresh_status(model_name)
                self.log(f"━━━ Complete: {model_name} ━━━")
                self.messagebox.showinfo(
                    "Download complete",
                    f"'{model_name}' is ready.",
                    parent=dialog,
                )

            self.threading.Thread(target=_do_download, daemon=True).start()

        def _refresh_backend_status_labels() -> None:
            main_path = self.Path(selected_musubi_main_path).expanduser()
            ltx_path = self.Path(selected_musubi_ltx_path).expanduser()
            sd_path = self.Path(selected_sd_scripts_path).expanduser()
            preferred_main = self.Path(selected_trainers_root).expanduser() / "musubi-main"
            preferred_ltx = self.Path(selected_trainers_root).expanduser() / "musubi-ltx"
            preferred_sd = self.Path(selected_trainers_root).expanduser() / "sd-scripts"

            main_ok = self.backend_is_valid("musubi-main", main_path)
            ltx_ok = self.backend_is_valid("musubi-ltx", ltx_path)
            sd_ok = self.backend_is_valid("sd-scripts", sd_path)
            musubi_main_status_var.set("Configured" if main_ok else "Missing or invalid")
            musubi_ltx_status_var.set("Configured" if ltx_ok else "Missing or invalid")
            sd_scripts_status_var.set("Configured" if sd_ok else "Missing or invalid")

            main_is_preferred = main_ok and main_path.resolve() == preferred_main.resolve()
            ltx_is_preferred = ltx_ok and ltx_path.resolve() == preferred_ltx.resolve()
            sd_is_preferred = sd_ok and sd_path.resolve() == preferred_sd.resolve()

            musubi_main_action_var.set("Update" if main_is_preferred else "Auto Download")
            musubi_ltx_action_var.set("Update" if ltx_is_preferred else "Auto Download")
            sd_scripts_action_var.set("Update" if sd_is_preferred else "Auto Download")

        def _persist_backend_paths() -> None:
            trainers_root_value = str(self.Path(selected_trainers_root).expanduser())
            self.settings_state[self.app_settings.TRAINERS_ROOT_KEY] = trainers_root_value
            self.settings_state[self.app_settings.MUSUBI_MAIN_DIR_KEY] = str(self.Path(selected_musubi_main_path).expanduser())
            self.settings_state[self.app_settings.MUSUBI_LTX_DIR_KEY] = str(self.Path(selected_musubi_ltx_path).expanduser())
            self.settings_state[self.app_settings.SD_SCRIPTS_DIR_KEY] = str(self.Path(selected_sd_scripts_path).expanduser())
            # Backward compatibility for legacy single-musubi key.
            self.settings_state[self.app_settings.MUSUBI_DIR_KEY] = self.settings_state[self.app_settings.MUSUBI_MAIN_DIR_KEY]
            self.app_settings.save_settings(self.settings_state)

        def browse_trainers_root() -> None:
            nonlocal selected_trainers_root, selected_musubi_main_path, selected_musubi_ltx_path, selected_sd_scripts_path
            picked = self.filedialog.askdirectory(
                parent=dialog,
                title="Select Trainers root folder",
                initialdir=selected_trainers_root or str(self.default_trainers_root()),
            )
            if not picked:
                return

            selected_trainers_root = picked
            trainers_root_var.set(picked)

            if not self.settings_state.get(self.app_settings.MUSUBI_MAIN_DIR_KEY, "").strip():
                selected_musubi_main_path = str(self.Path(picked) / "musubi-main")
                musubi_main_display_var.set(selected_musubi_main_path)
            if not self.settings_state.get(self.app_settings.MUSUBI_LTX_DIR_KEY, "").strip():
                selected_musubi_ltx_path = str(self.Path(picked) / "musubi-ltx")
                musubi_ltx_display_var.set(selected_musubi_ltx_path)
            if not self.settings_state.get(self.app_settings.SD_SCRIPTS_DIR_KEY, "").strip():
                selected_sd_scripts_path = str(self.Path(picked) / "sd-scripts")
                sd_scripts_display_var.set(selected_sd_scripts_path)
            _refresh_backend_status_labels()

        def browse_backend(kind: str) -> None:
            nonlocal selected_musubi_main_path, selected_musubi_ltx_path, selected_sd_scripts_path, selected_musubi_python
            if kind == "musubi-main":
                current = selected_musubi_main_path
                title = "Select Musubi Main folder"
            elif kind == "musubi-ltx":
                current = selected_musubi_ltx_path
                title = "Select Musubi LTX folder"
            else:
                current = selected_sd_scripts_path
                title = "Select sd-scripts folder"

            picked = self.filedialog.askdirectory(
                parent=dialog,
                title=title,
                initialdir=current or str(self.Path.home()),
            )
            if not picked:
                return

            if kind == "musubi-main":
                selected_musubi_main_path = picked
                musubi_main_display_var.set(picked)
                detected_python = self.resolve_musubi_python(self.Path(picked).expanduser())
                selected_musubi_python = str(detected_python) if detected_python is not None else ""
            elif kind == "musubi-ltx":
                selected_musubi_ltx_path = picked
                musubi_ltx_display_var.set(picked)
            else:
                selected_sd_scripts_path = picked
                sd_scripts_display_var.set(picked)

            _refresh_backend_status_labels()

        def auto_download_backend(kind: str) -> None:
            nonlocal selected_musubi_main_path, selected_musubi_ltx_path, selected_sd_scripts_path, selected_musubi_python
            row_map = {row_kind: (label, repo_url, branch) for label, row_kind, _k, _d, repo_url, branch in self.backend_repo_rows()}
            label, repo_url, branch = row_map[kind]
            clone_confirmed = False

            if kind == "musubi-main":
                current_target = self.Path(selected_musubi_main_path).expanduser() if selected_musubi_main_path else self.Path(selected_trainers_root) / "musubi-main"
                preferred_target = self.Path(selected_trainers_root).expanduser() / "musubi-main"
            elif kind == "musubi-ltx":
                current_target = self.Path(selected_musubi_ltx_path).expanduser() if selected_musubi_ltx_path else self.Path(selected_trainers_root) / "musubi-ltx"
                preferred_target = self.Path(selected_trainers_root).expanduser() / "musubi-ltx"
            else:
                current_target = self.Path(selected_sd_scripts_path).expanduser() if selected_sd_scripts_path else self.Path(selected_trainers_root) / "sd-scripts"
                preferred_target = self.Path(selected_trainers_root).expanduser() / "sd-scripts"

            target = preferred_target

            selected_is_preferred = False
            if self.backend_is_valid(kind, current_target):
                try:
                    selected_is_preferred = current_target.resolve() == target.resolve()
                except OSError:
                    selected_is_preferred = False

            def _apply_selected_backend(path: Path) -> None:
                nonlocal selected_musubi_main_path, selected_musubi_ltx_path, selected_sd_scripts_path, selected_musubi_python
                if kind == "musubi-main":
                    selected_musubi_main_path = str(path)
                    musubi_main_display_var.set(str(path))
                    detected_python = self.resolve_musubi_python(path)
                    selected_musubi_python = str(detected_python) if detected_python is not None else ""
                elif kind == "musubi-ltx":
                    selected_musubi_ltx_path = str(path)
                    musubi_ltx_display_var.set(str(path))
                else:
                    selected_sd_scripts_path = str(path)
                    sd_scripts_display_var.set(str(path))
                _refresh_backend_status_labels()
                _persist_backend_paths()

            if selected_is_preferred:
                _settings_log(f"[Trainers] {label}: update started")
                confirm_update = self.messagebox.askyesno(
                    "Update backend",
                    f"Run git pull for {label}?\n\nTarget:\n{target}",
                    parent=dialog,
                )
                if not confirm_update:
                    _settings_log(f"[Trainers] {label}: update cancelled")
                    return

                ok, output = self.update_backend_repo(target, logger=_settings_log)
                if ok:
                    _apply_selected_backend(target)
                    _settings_log(f"[Trainers] {label}: update done")
                    self.messagebox.showinfo(
                        "Update complete",
                        f"{label} updated successfully.",
                        parent=dialog,
                    )
                    return

                _settings_log(f"[Trainers] {label}: update failed")

                reset_now = self.messagebox.askyesno(
                    "Update failed",
                    f"git pull failed for {label}.\n\n"
                    "See the app console for details.\n\n"
                    "Do you want to reset this backend (delete folder and re-download)?",
                    parent=dialog,
                )
                if not reset_now:
                    _settings_log(f"[Trainers] {label}: reset cancelled")
                    return

                _settings_log(f"[Trainers] {label}: reset started")
                try:
                    self.shutil.rmtree(target)
                except OSError as exc:
                    _settings_log(f"[Trainers] {label}: reset failed")
                    self.messagebox.showerror("Reset failed", f"Could not remove folder:\n{exc}", parent=dialog)
                    return

                ok, err = self.clone_backend_repo(target, repo_url, branch, logger=_settings_log)
                if not ok:
                    _settings_log(f"[Trainers] {label}: re-download failed")
                    self.messagebox.showerror("Re-download failed", err, parent=dialog)
                    return

                _apply_selected_backend(target)
                _settings_log(f"[Trainers] {label}: reset done")
                self.messagebox.showinfo("Reset complete", f"Re-downloaded {label}:\n{target}", parent=dialog)
                return

            if self.backend_is_valid(kind, target):
                _apply_selected_backend(target)
                return

            if self.backend_is_valid(kind, current_target) and current_target.resolve() != target.resolve():
                clone_here = self.messagebox.askyesno(
                    "Download backend",
                    f"{label} is currently configured outside Trainers:\n{current_target}\n\n"
                    f"Clone another copy into Trainers and switch to it?\n\nTarget:\n{target}",
                    parent=dialog,
                )
                if not clone_here:
                    _apply_selected_backend(current_target)
                    return
                clone_confirmed = True

            if not clone_confirmed:
                confirm = self.messagebox.askyesno(
                    "Download backend",
                    f"Clone {label} into:\n{target}\n\nRepository:\n{repo_url}" + (f"\nBranch: {branch}" if branch else ""),
                    parent=dialog,
                )
                if not confirm:
                    _settings_log(f"[Trainers] {label}: download cancelled")
                    return

            _settings_log(f"[Trainers] {label}: download started")
            ok, err = self.clone_backend_repo(target, repo_url, branch, logger=_settings_log)
            if not ok:
                _settings_log(f"[Trainers] {label}: download failed")
                self.messagebox.showerror("Clone failed", err, parent=dialog)
                return

            _apply_selected_backend(target)
            _settings_log(f"[Trainers] {label}: download done")
            self.messagebox.showinfo("Clone complete", f"Configured {label}:\n{target}", parent=dialog)

        _refresh_backend_status_labels()

        def browse_file(current_path: str, initial_dir_hint: str, title: str) -> str | None:
            initial_dir = initial_dir_hint
            if self.default_models_dir.exists() and self.default_models_dir.is_dir():
                initial_dir = str(self.default_models_dir)
            if current_path:
                current_parent = self.Path(current_path).expanduser().parent
                initial_dir = str(current_parent)
            picked = self.filedialog.askopenfilename(
                parent=dialog,
                title=title,
                initialdir=initial_dir or str(self.Path.home()),
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
            )
            return picked if picked else None

        def normalize_model_checkpoint_path(raw_path: str | None) -> str:
            if not raw_path:
                return ""
            candidate = self.Path(raw_path).expanduser()
            if candidate.is_file() and candidate.name.lower().endswith(".safetensors.index.json"):
                try:
                    payload = self.json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, self.json.JSONDecodeError):
                    return str(candidate)
                weight_map = payload.get("weight_map", {})
                if not isinstance(weight_map, dict) or not weight_map:
                    return str(candidate)
                shard_names = sorted({str(v) for v in weight_map.values() if isinstance(v, str) and v.lower().endswith(".safetensors")})
                preferred = next((name for name in shard_names if self.re.search(r"-00001-of-\d+\.safetensors$", name, flags=self.re.IGNORECASE)), None)
                shard_name = preferred or (shard_names[0] if shard_names else None)
                if shard_name:
                    shard_path = candidate.parent / shard_name
                    if shard_path.is_file():
                        return str(shard_path)
            return str(candidate)

        def save_and_close() -> None:
            nonlocal result
            # Force any focused entry widget to commit (triggers FocusOut → _save_path)
            dialog.focus_set()
            main_dir = self.Path(selected_musubi_main_path).expanduser()
            ltx_dir = self.Path(selected_musubi_ltx_path).expanduser()
            sd_scripts_dir = self.Path(selected_sd_scripts_path).expanduser()

            if required:
                if not self.backend_is_valid("musubi-main", main_dir) and not self.backend_is_valid("musubi-ltx", ltx_dir):
                    self.messagebox.showerror(
                        "Missing backend",
                        "Set at least one Musubi backend (Main or LTX) before continuing.",
                        parent=dialog,
                    )
                    return

            try:
                save_every_default_value = int(train_save_every_default_var.get().strip())
                if save_every_default_value < 1:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror(
                    "Invalid value",
                    "Training default 'Save every N steps' must be a positive integer.",
                    parent=dialog,
                )
                return

            if not main_dir.exists() or not main_dir.is_dir():
                self.messagebox.showerror("Invalid folder", "Choose a valid Musubi Main folder.", parent=dialog)
                return

            musubi_python_path = self.resolve_musubi_python(main_dir)
            if musubi_python_path is None:
                self.messagebox.showerror(
                    "Python venv not found",
                    "App venv Python was not found. Run Setup.bat first.",
                    parent=dialog,
                )
                return

            import json as _json_save
            trainers_root_value = str(self.Path(selected_trainers_root).expanduser())
            self.settings_state[self.app_settings.TRAINERS_ROOT_KEY] = trainers_root_value
            self.settings_state[self.app_settings.MUSUBI_MAIN_DIR_KEY] = str(main_dir)
            self.settings_state[self.app_settings.MUSUBI_LTX_DIR_KEY] = str(ltx_dir)
            self.settings_state[self.app_settings.SD_SCRIPTS_DIR_KEY] = str(sd_scripts_dir)
            self.settings_state[self.app_settings.MUSUBI_DIR_KEY] = str(main_dir)
            self.settings_state[self.app_settings.MUSUBI_PYTHON_KEY] = ""
            self.settings_state[self.app_settings.MODEL_PATHS_KEY] = _json_save.dumps(pending_model_paths)
            # Backward compat: derive legacy keys from pending_model_paths
            _active_klein = self.settings_state.get(self.app_settings.KLEIN_MODEL_VERSION_KEY, "klein-base-9b") or "klein-base-9b"
            _kpaths = pending_model_paths.get(_active_klein, {})
            self.settings_state[self.app_settings.KLEIN_MODEL_VERSION_KEY] = _active_klein
            self.settings_state[self.app_settings.KLEIN_DIT_KEY] = _kpaths.get("dit", "")
            self.settings_state[self.app_settings.KLEIN_VAE_KEY] = _kpaths.get("vae", "")
            self.settings_state[self.app_settings.KLEIN_TEXT_ENCODER_KEY] = _kpaths.get("text_encoder", "")
            _ltx_paths = pending_model_paths.get("ltx-2.3", {})
            self.settings_state[self.app_settings.LTX_MODEL_VERSION_KEY] = "ltx-2.3" if _ltx_paths else ""
            self.settings_state[self.app_settings.LTX_DIT_KEY] = _ltx_paths.get("dit", "")
            self.settings_state[self.app_settings.LTX_VAE_KEY] = _ltx_paths.get("vae", "")
            self.settings_state[self.app_settings.LTX_TEXT_ENCODER_KEY] = _ltx_paths.get("text_encoder", "")
            self.settings_state[self.app_settings.DEFAULT_CAPTION_KEYWORD_KEY] = default_caption_keyword_var.get().strip()
            self.settings_state[self.app_settings.TRAIN_ENABLE_LOGGING_KEY] = "1" if enable_training_logging_var.get() else "0"
            self.settings_state[self.app_settings.TRAIN_LOG_BACKEND_KEY] = "tensorboard"
            self.settings_state[self.app_settings.TRAIN_LOG_TRACKER_NAME_KEY] = train_log_tracker_name_var.get().strip()
            self.settings_state[self.app_settings.TRAIN_STREAM_TO_LOGGER_KEY] = "1" if stream_to_logger_var.get() else "0"
            self.settings_state[self.app_settings.TRAIN_AUTO_START_TENSORBOARD_KEY] = "1" if auto_start_tensorboard_var.get() else "0"
            self.settings_state[self.app_settings.TRAIN_SAVE_EVERY_N_STEPS_KEY] = str(save_every_default_value)
            preferred_to_save = {
                family_name: var.get().strip()
                for family_name, var in preferred_preset_vars.items()
                if var.get().strip() and var.get().strip() != preset_none_label
            }
            self.settings_state[self.app_settings.PREFERRED_PRESETS_BY_FAMILY_KEY] = _json_save.dumps(preferred_to_save)
            self.settings_state[self.app_settings.MODEL_DOWNLOAD_LOCATION_KEY] = model_location_var.get()
            self.settings_state[self.app_settings.HF_TOKEN_KEY] = hf_token_var.get().strip()
            _ep = extra_path_var.get().strip()
            self.settings_state[self.app_settings.EXTRA_SEARCH_PATHS_KEY] = _json_save.dumps([_ep] if _ep else [])
            self.app_settings.save_settings(self.settings_state)
            result = self.runtime_config_from_settings(self.settings_state)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        def reset_settings() -> None:
            nonlocal result
            confirmed = self.messagebox.askyesno(
                "Reset settings",
                "Delete settings.json and reset all saved settings?",
                parent=dialog,
            )
            if not confirmed:
                return

            try:
                if self.app_settings.SETTINGS_FILE.exists():
                    self.app_settings.SETTINGS_FILE.unlink()
            except OSError as exc:
                self.messagebox.showerror("Reset failed", f"Could not delete settings file:\n{exc}", parent=dialog)
                return

            self.set_settings_reset_requested(True)
            self.settings_state.clear()
            result = None
            dialog.destroy()
            self.root.after_idle(self.root.destroy)

        button_row = self.ttk.Frame(footer)
        button_row.grid(row=0, column=0, sticky="ew")
        button_row.columnconfigure(0, weight=1)
        self.ttk.Button(button_row, text="Reset Settings", command=reset_settings).grid(row=0, column=0, sticky="w")
        self.ttk.Button(button_row, text="Cancel", command=cancel_and_close).grid(row=0, column=1, padx=(0, 8))
        self.ttk.Button(button_row, text="Save", command=save_and_close).grid(row=0, column=2)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.update_idletasks()

        content_w = max(notebook.winfo_reqwidth(), footer.winfo_reqwidth()) + 44
        content_h = notebook.winfo_reqheight() + footer.winfo_reqheight() + 20
        max_w = max(760, dialog.winfo_screenwidth() - 80)
        max_h = max(480, dialog.winfo_screenheight() - 80)
        win_w = max(780, min(1080, content_w))
        win_w = min(win_w, max_w)
        win_h = min(max(520, content_h), max_h)
        dialog.geometry(f"{win_w}x{win_h}")
        self.center_window(dialog)
        dialog.deiconify()
        dialog.grab_set()
        dialog.focus_set()
        self.root.wait_window(dialog)

        if required and result is None:
            return None

        return result



