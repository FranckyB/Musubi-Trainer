from __future__ import annotations

from pathlib import Path


class LoraMergeWindow:
    def __init__(self, **dependencies: object) -> None:
        for name, value in dependencies.items():
            setattr(self, name, value)

    def ask_lora_merge_options(
        self,
        dataset_name: str,
        available_loras: list[Path],
    ) -> tuple[list[str], list[tuple[str, str, list[str], str]]] | None:
        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("LoRA Post-Hoc EMA Merge")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=self.bg_panel)
        self.set_dark_title_bar(dialog)
        dialog.minsize(380, 460)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        frame = self.ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        self.ttk.Label(frame, text=f"LoRAs in output for {dataset_name}:").grid(row=0, column=0, sticky="w")
        self.ttk.Label(frame, text="Select the LoRAs you want to merge.").grid(row=1, column=0, sticky="w", pady=(4, 0))

        def _is_comfy_variant(path: Path) -> bool:
            return path.name.casefold().endswith(".comfy.safetensors")

        def _is_standard_variant(path: Path) -> bool:
            name_lower = path.name.casefold()
            return name_lower.endswith(".safetensors") and not name_lower.endswith(".comfy.safetensors")

        has_comfy_variants = any(_is_comfy_variant(path) for path in available_loras)
        has_standard_variants = any(_is_standard_variant(path) for path in available_loras)
        ltx_toggle_enabled = has_comfy_variants and has_standard_variants
        variant_mode_var = self.tk.StringVar(value="comfy")
        visible_loras: list[Path] = []

        toggle_row = self.ttk.Frame(frame)
        toggle_row.grid(row=2, column=0, sticky="w", pady=(6, 0))
        if ltx_toggle_enabled:
            self.ttk.Label(toggle_row, text="Show:").grid(row=0, column=0, sticky="w", padx=(0, 8))
            comfy_toggle_border = self.tk.Frame(toggle_row, bg="#4a4a4a", bd=0, highlightthickness=0)
            comfy_toggle_border.grid(row=0, column=1, sticky="w")
            comfy_toggle = self.tk.Button(
                comfy_toggle_border,
                text="Comfy",
                command=lambda: variant_mode_var.set("comfy"),
                bg="#2b2b2b",
                fg="#e6e6e6",
                activebackground="#3a3a3a",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=10,
                pady=2,
            )
            comfy_toggle.pack(padx=1, pady=1)
            standard_toggle_border = self.tk.Frame(toggle_row, bg="#4a4a4a", bd=0, highlightthickness=0)
            standard_toggle_border.grid(row=0, column=2, sticky="w", padx=(6, 0))
            standard_toggle = self.tk.Button(
                standard_toggle_border,
                text="Standard",
                command=lambda: variant_mode_var.set("standard"),
                bg="#2b2b2b",
                fg="#e6e6e6",
                activebackground="#3a3a3a",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=10,
                pady=2,
            )
            standard_toggle.pack(padx=1, pady=1)

            def _style_toggle(border: object, button: object, selected: bool) -> None:
                if selected:
                    border.configure(bg="#4aa3ff")
                    button.configure(bg="#31485f", fg="#ffffff")
                else:
                    border.configure(bg="#4a4a4a")
                    button.configure(bg="#2b2b2b", fg="#e6e6e6")

        selected_box_frame = self.ttk.Frame(frame)
        selected_box_frame.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        selected_box_frame.columnconfigure(0, weight=1)
        selected_box_frame.rowconfigure(0, weight=1)

        selected_list = self.tk.Listbox(
            selected_box_frame,
            selectmode="extended",
            exportselection=False,
            activestyle="none",
            width=1,
            bg="#1f1f1f",
            fg=self.fg_text,
            highlightthickness=1,
            highlightbackground=self.border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
            height=8,
        )
        selected_list.grid(row=0, column=0, sticky="nsew")
        selected_scroll = self.ttk.Scrollbar(
            selected_box_frame,
            orient="vertical",
            command=selected_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        selected_scroll.grid(row=0, column=1, sticky="ns")
        selected_list.configure(yscrollcommand=selected_scroll.set)

        def _refresh_visible_loras(*_args: object) -> None:
            nonlocal visible_loras
            if not ltx_toggle_enabled:
                visible_loras = list(available_loras)
            elif variant_mode_var.get() == "standard":
                visible_loras = [p for p in available_loras if _is_standard_variant(p)]
            else:
                visible_loras = [p for p in available_loras if _is_comfy_variant(p)]

            selected_list.delete(0, "end")
            for lora_path in visible_loras:
                selected_list.insert("end", lora_path.name)

            if ltx_toggle_enabled:
                mode_is_comfy = variant_mode_var.get() == "comfy"
                _style_toggle(comfy_toggle_border, comfy_toggle, mode_is_comfy)
                _style_toggle(standard_toggle_border, standard_toggle, not mode_is_comfy)

        if ltx_toggle_enabled:
            variant_mode_var.trace_add("write", _refresh_visible_loras)
        _refresh_visible_loras()

        self.ttk.Label(
            frame,
            text="Post-Hoc EMA smooths checkpoints from the same run into one more stable LoRA.",
        ).grid(row=4, column=0, sticky="ew", pady=(10, 0))

        options_frame = self.ttk.Frame(frame)
        options_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

        mode_section = self.ttk.LabelFrame(options_frame, text="Mode(s)", padding=6)
        mode_section.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.attach_hover_tooltip(mode_section, self.merge_mode_tooltip_text)

        preset_section = self.ttk.LabelFrame(options_frame, text="Preset(s)", padding=6)
        preset_section.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.attach_hover_tooltip(preset_section, self.merge_preset_tooltip_text)

        preset_balanced_var = self.tk.BooleanVar(value=False)
        preset_smooth_var = self.tk.BooleanVar(value=False)
        preset_anti_overfit_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(preset_section, text="Balanced", variable=preset_balanced_var).grid(row=0, column=0, sticky="w")
        self.ttk.Checkbutton(preset_section, text="Smooth", variable=preset_smooth_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(preset_section, text="Anti-overfit", variable=preset_anti_overfit_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        mode_beta_var = self.tk.BooleanVar(value=False)
        mode_beta2_var = self.tk.BooleanVar(value=False)
        mode_sigma_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(mode_section, text="SIGMA_REL", variable=mode_sigma_var).grid(row=0, column=0, sticky="w")
        self.ttk.Checkbutton(mode_section, text="BETA", variable=mode_beta_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(mode_section, text="BETA + BETA2 (Interpolated)", variable=mode_beta2_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        button_row = self.ttk.Frame(frame)
        button_row.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)

        choice: tuple[list[str], list[tuple[str, str, list[str], str]]] | None = None

        def choose_and_close() -> None:
            nonlocal choice
            picked_indices = selected_list.curselection()
            selected_file_paths = [str(visible_loras[i]) for i in picked_indices]
            if len(selected_file_paths) < 2:
                self.messagebox.showerror(
                    "Merge unavailable",
                    "Select at least 2 .safetensors files for Post-Hoc EMA merge.",
                    parent=dialog,
                )
                return

            selected_preset_names: list[str] = []
            if preset_balanced_var.get():
                selected_preset_names.append("Balanced")
            if preset_smooth_var.get():
                selected_preset_names.append("Smooth")
            if preset_anti_overfit_var.get():
                selected_preset_names.append("Anti-overfit")
            if not selected_preset_names:
                self.messagebox.showerror(
                    "Merge unavailable",
                    "Select at least one preset.",
                    parent=dialog,
                )
                return

            mode_defs: list[tuple[str, str, str]] = []
            if mode_beta_var.get():
                mode_defs.append(("BETA", "Beta", "beta"))
            if mode_beta2_var.get():
                mode_defs.append(("BETA2", "Beta2", "beta2"))
            if mode_sigma_var.get():
                mode_defs.append(("SIGMA_REL", "Sigma", "sigma_rel"))

            if not mode_defs:
                self.messagebox.showerror(
                    "Merge unavailable",
                    "Select at least one merge mode.",
                    parent=dialog,
                )
                return

            selected_jobs: list[tuple[str, str, list[str], str]] = []
            for preset_name in selected_preset_names:
                preset_args = self.post_hoc_ema_mode_args_for_preset(preset_name)
                for mode_label, mode_suffix, mode_key in mode_defs:
                    selected_jobs.append((mode_label, mode_suffix, preset_args[mode_key], preset_name))

            choice = (selected_file_paths, selected_jobs)
            dialog.destroy()

        def cancel_and_close() -> None:
            dialog.destroy()

        go_button = self.ttk.Button(button_row, text="Go", command=choose_and_close)
        go_button.grid(row=0, column=0)

        dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dialog.bind("<Escape>", lambda _e: cancel_and_close())
        dialog.bind("<Return>", lambda _e: choose_and_close())

        dialog.update_idletasks()
        requested_width = max(390, dialog.winfo_reqwidth())
        requested_height = max(500, dialog.winfo_reqheight())
        dialog.geometry(f"{requested_width}x{requested_height}")
        self.center_window(dialog)
        dialog.deiconify()
        dialog.focus_set()
        selected_list.focus_set()
        self.root.wait_window(dialog)
        return choice

    def lora_post_hoc_ema_merge(self, dataset_name: str) -> None:
        output_dir = self.runtime_config.training_dir / dataset_name / "output"
        self.lora_post_hoc_ema_merge_for_output(dataset_name, output_dir)

    def lora_post_hoc_ema_merge_for_output(
        self,
        target_name: str,
        output_dir: Path,
        merge_output_dir: Path | None = None,
    ) -> None:
        if merge_output_dir is None:
            merge_output_dir = output_dir
        if not output_dir.exists() or not output_dir.is_dir():
            self.messagebox.showerror(
                "Merge unavailable",
                "No output folder was found for this job.",
                parent=self.root,
            )
            return

        def _merge_candidate_sort_key(path: Path) -> tuple[str, int, str]:
            name_lower = path.name.casefold()
            comfy_suffix = ".comfy.safetensors"
            std_suffix = ".safetensors"

            if name_lower.endswith(comfy_suffix):
                base = name_lower[: -len(comfy_suffix)]
                variant_rank = 0
            elif name_lower.endswith(std_suffix):
                base = name_lower[: -len(std_suffix)]
                variant_rank = 1
            else:
                base = name_lower
                variant_rank = 2

            return (base, variant_rank, name_lower)

        available = sorted(
            [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors"],
            key=_merge_candidate_sort_key,
        )
        if not available:
            self.messagebox.showerror(
                "Merge unavailable",
                "No .safetensors files were found in this output folder.",
                parent=self.root,
            )
            return
        module_script = self.runtime_config.musubi_dir / "src" / "musubi_tuner" / "lora_post_hoc_ema.py"
        root_script = self.runtime_config.musubi_dir / "lora_post_hoc_ema.py"
        if not module_script.exists() and not root_script.exists():
            self.messagebox.showerror(
                "Merge unavailable",
                "Could not find lora_post_hoc_ema.py in Musubi-Tuner.",
                parent=self.root,
            )
            return

        merge_options = self.ask_lora_merge_options(target_name, available)
        if merge_options is None:
            return
        selected_files, selected_jobs = merge_options

        musubi_python = self.runtime_config.musubi_python
        if musubi_python is None or not musubi_python.is_file():
            self.messagebox.showerror(
                "Merge unavailable",
                (
                    "Musubi-Tuner Python was not found in its venv.\n"
                    "Expected: .venv/Scripts/python.exe inside your Musubi-Tuner folder."
                ),
                parent=self.root,
            )
            return

        run_env = self.os.environ.copy()
        musubi_src = str(self.runtime_config.musubi_dir / "src")
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = musubi_src if not existing_pythonpath else f"{musubi_src}{self.os.pathsep}{existing_pythonpath}"

        self.log("")
        created_paths: list[Path] = []
        for merge_mode_label, merge_mode_suffix, merge_mode_args, preset_name in selected_jobs:
            output_path = self.next_merged_output_path(
                target_name,
                merge_output_dir,
                merge_mode_suffix,
                preset_name,
                selected_files,
            )
            command: list[str]
            if module_script.exists():
                command = [
                    str(musubi_python),
                    "-m",
                    "musubi_tuner.lora_post_hoc_ema",
                    *selected_files,
                    "--output_file",
                    str(output_path),
                    *merge_mode_args,
                ]
            else:
                command = [
                    str(musubi_python),
                    str(root_script),
                    *selected_files,
                    "--output_file",
                    str(output_path),
                    *merge_mode_args,
                ]

            self.log(
                f"[Post-Hoc EMA] Merging {len(selected_files)} checkpoint(s) for '{target_name}' "
                f"using {merge_mode_label} / {preset_name}..."
            )
            result = self.subprocess.run(
                command,
                cwd=str(self.runtime_config.musubi_dir),
                env=run_env,
                capture_output=True,
                text=True,
            )

            stdout_text = result.stdout.strip()
            stderr_text = result.stderr.strip()
            if stdout_text:
                self.log(stdout_text)

            if result.returncode != 0:
                message = stderr_text if stderr_text else "lora_post_hoc_ema.py failed with no error output."
                self.log(f"[Post-Hoc EMA] Failed ({result.returncode}) while running {merge_mode_label} / {preset_name}.")
                if stderr_text:
                    self.log(stderr_text)
                self.messagebox.showerror("Post-Hoc EMA merge failed", message, parent=self.root)
                return

            self.log(f"[Post-Hoc EMA] Created ({merge_mode_suffix} / {preset_name}): {output_path}")
            if stderr_text:
                self.log(stderr_text)
            created_paths.append(output_path)

        if created_paths:
            created_text = "\n".join(path.stem if path.suffix.lower() == ".safetensors" else path.name for path in created_paths)
            self.messagebox.showinfo("Post-Hoc EMA merge complete", f"Created:\n{created_text}", parent=self.root)

        self.checkpoint_cache.pop(target_name, None)
        self.rebuild_folder_list(force=True)

    def open_lora_merge_tool_dialog(self) -> None:
        musubi_python = self.runtime_config.musubi_python
        if musubi_python is None or not musubi_python.is_file():
            self.messagebox.showerror(
                "Merge unavailable",
                "App venv Python was not found. Run Setup.bat first.",
                parent=self.root,
            )
            return

        module_script = self.runtime_config.musubi_dir / "src" / "musubi_tuner" / "lora_post_hoc_ema.py"
        root_script = self.runtime_config.musubi_dir / "lora_post_hoc_ema.py"
        if not module_script.exists() and not root_script.exists():
            self.messagebox.showerror(
                "Merge unavailable",
                "Could not find lora_post_hoc_ema.py in Musubi-Tuner.",
                parent=self.root,
            )
            return

        dialog = self.tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title("LoRA Post-Hoc EMA Merge")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.bg_panel)
        dialog.resizable(False, False)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        self.set_dark_title_bar(dialog)

        candidate_loras: list[Path] = []
        merge_loras: list[Path] = []

        frame = self.ttk.Frame(dialog, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        candidate_section = self.ttk.LabelFrame(frame, text="LoRAs", padding=8)
        candidate_section.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        candidate_section.columnconfigure(0, weight=1)
        candidate_section.rowconfigure(0, weight=1)

        candidate_list = self.tk.Listbox(
            candidate_section,
            selectmode="extended",
            exportselection=False,
            height=10,
            activestyle="none",
            bg="#1f1f1f",
            fg=self.fg_text,
            highlightthickness=1,
            highlightbackground=self.border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
        )
        candidate_list.grid(row=0, column=0, sticky="nsew")
        candidate_scroll = self.ttk.Scrollbar(
            candidate_section,
            orient="vertical",
            command=candidate_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        candidate_scroll.grid(row=0, column=1, sticky="ns")
        candidate_list.configure(yscrollcommand=candidate_scroll.set)

        merge_section = self.ttk.LabelFrame(frame, text="Merge order", padding=8)
        merge_section.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        merge_section.columnconfigure(0, weight=1)
        merge_section.rowconfigure(0, weight=1)

        merge_list = self.tk.Listbox(
            merge_section,
            selectmode="extended",
            exportselection=False,
            height=10,
            activestyle="none",
            bg="#1f1f1f",
            fg=self.fg_text,
            highlightthickness=1,
            highlightbackground=self.border_dark,
            selectbackground="#2f4f66",
            selectforeground="#ffffff",
            relief="flat",
        )
        merge_list.grid(row=0, column=0, sticky="nsew")
        merge_scroll = self.ttk.Scrollbar(
            merge_section,
            orient="vertical",
            command=merge_list.yview,
            style="Dark.Vertical.TScrollbar",
        )
        merge_scroll.grid(row=0, column=1, sticky="ns")
        merge_list.configure(yscrollcommand=merge_scroll.set)

        mode_section = self.ttk.LabelFrame(frame, text="Merge options", padding=8)
        mode_section.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        mode_section.columnconfigure(0, weight=1)
        mode_section.columnconfigure(1, weight=1)

        self.ttk.Label(
            mode_section,
            text="Post-Hoc EMA smooths checkpoints from the same run into one more stable LoRA.",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        mode_beta_var = self.tk.BooleanVar(value=False)
        mode_beta2_var = self.tk.BooleanVar(value=False)
        mode_sigma_var = self.tk.BooleanVar(value=False)
        preset_balanced_var = self.tk.BooleanVar(value=False)
        preset_smooth_var = self.tk.BooleanVar(value=False)
        preset_anti_overfit_var = self.tk.BooleanVar(value=False)

        mode_group = self.ttk.LabelFrame(mode_section, text="Mode(s)", padding=6)
        mode_group.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(8, 0))
        self.attach_hover_tooltip(mode_group, self.merge_mode_tooltip_text)
        self.ttk.Checkbutton(mode_group, text="SIGMA_REL", variable=mode_sigma_var).grid(row=0, column=0, sticky="w")
        self.ttk.Checkbutton(mode_group, text="BETA", variable=mode_beta_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(mode_group, text="BETA + BETA2 (Interpolated)", variable=mode_beta2_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        preset_section = self.ttk.LabelFrame(mode_section, text="Preset(s)", padding=6)
        preset_section.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(8, 0))
        self.attach_hover_tooltip(preset_section, self.merge_preset_tooltip_text)
        self.ttk.Checkbutton(preset_section, text="Balanced", variable=preset_balanced_var).grid(row=0, column=0, sticky="w")
        self.ttk.Checkbutton(preset_section, text="Smooth", variable=preset_smooth_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ttk.Checkbutton(preset_section, text="Anti-overfit", variable=preset_anti_overfit_var).grid(row=2, column=0, sticky="w", pady=(6, 0))

        output_name_var = self.tk.StringVar(value="merged_lora")
        output_dir_var = self.tk.StringVar(value="")

        self.ttk.Label(mode_section, text="Output name:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.ttk.Entry(mode_section, textvariable=output_name_var, style="Flat.TEntry").grid(row=2, column=1, sticky="ew", pady=(8, 0))

        self.ttk.Label(mode_section, text="Output folder:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.ttk.Label(mode_section, textvariable=output_dir_var, style="PathDisplay.TLabel", anchor="w", padding=(6, 4)).grid(
            row=3,
            column=1,
            sticky="ew",
            pady=(8, 0),
        )

        actions = self.ttk.Frame(frame)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)

        def refresh_candidate_list() -> None:
            candidate_list.delete(0, "end")
            for path in candidate_loras:
                candidate_list.insert("end", path.name)

        def refresh_merge_list() -> None:
            merge_list.delete(0, "end")
            for path in merge_loras:
                merge_list.insert("end", path.name)

        def _normalize_lora_paths(raw_paths: list[str]) -> list[Path]:
            normalized: list[Path] = []
            for raw_path in raw_paths:
                value = raw_path.strip().strip('"')
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.exists() or not path.is_file() or path.suffix.lower() != ".safetensors":
                    continue
                normalized.append(path)
            return normalized

        def add_loras_to_candidate_pool(paths: list[Path]) -> int:
            if not paths:
                return 0
            existing = {str(path.resolve()) for path in candidate_loras}
            added = 0
            for path in paths:
                key = str(path.resolve())
                if key in existing:
                    continue
                candidate_loras.append(path)
                existing.add(key)
                added += 1

            if added > 0:
                candidate_loras.sort(key=lambda p: p.name.lower())
                refresh_candidate_list()
            return added

        def add_paths_to_merge_list(paths: list[Path]) -> int:
            if not paths:
                return 0
            existing = {str(path.resolve()) for path in merge_loras}
            added = 0
            for path in paths:
                key = str(path.resolve())
                if key in existing:
                    continue
                merge_loras.append(path)
                existing.add(key)
                added += 1

            if added > 0:
                if merge_loras and not output_dir_var.get().strip():
                    output_dir_var.set(str(merge_loras[0].parent))
                refresh_merge_list()
            return added

        def add_raw_paths(raw_paths: list[str], to_merge: bool = False) -> int:
            paths = _normalize_lora_paths(raw_paths)
            if not paths:
                return 0
            add_loras_to_candidate_pool(paths)
            return add_paths_to_merge_list(paths) if to_merge else len(paths)

        def add_loras_to_pool() -> None:
            initial_dir = str(self.runtime_config.training_dir)
            if candidate_loras:
                initial_dir = str(candidate_loras[-1].parent)
            picked = self.filedialog.askopenfilenames(
                parent=dialog,
                title="Select LoRA files",
                initialdir=initial_dir,
                filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")],
            )
            if not picked:
                return

            add_raw_paths([str(p) for p in picked], to_merge=False)

        def add_to_merge_list() -> None:
            raw_selected_indices = list(candidate_list.curselection())
            if not raw_selected_indices:
                return

            selected_indices: list[int] = []
            for raw_index in raw_selected_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(candidate_loras):
                    selected_indices.append(index)

            if not selected_indices:
                return

            selected_paths = [candidate_loras[index] for index in selected_indices]
            add_paths_to_merge_list(selected_paths)

        def on_candidate_double_click(event: Any) -> str:
            # Only treat double-clicks on an actual item row as "add to merge".
            clicked_index = candidate_list.nearest(event.y)
            if clicked_index < 0 or clicked_index >= len(candidate_loras):
                return "break"

            row_bbox = candidate_list.bbox(clicked_index)
            if row_bbox is None:
                return "break"
            _x, y, _w, h = row_bbox
            if not (y <= event.y < y + h):
                return "break"

            candidate_list.selection_clear(0, "end")
            candidate_list.selection_set(clicked_index)
            candidate_list.activate(clicked_index)
            add_to_merge_list()
            return "break"

        def try_enable_file_dnd() -> bool:
            if not self.tkdnd_available or self.DND_FILES is None:
                return False

            def process_drop_on_ui_thread(raw_paths: list[str], to_merge: bool, target_name: str) -> None:
                if not dialog.winfo_exists():
                    return

                def _apply_drop() -> None:
                    try:
                        added = add_raw_paths(raw_paths, to_merge=to_merge)
                        if added > 0:
                            destination = "merge list" if to_merge else "pool"
                            self.log(f"[LoRA Post-Hoc EMA Merge] Added {added} dropped LoRA file(s) to {destination}.")
                    except Exception as exc:
                        self.log(f"[LoRA Post-Hoc EMA Merge] Drop handling failed on {target_name}: {exc}")

                dialog.after(0, _apply_drop)

            def decode_dropped_paths(event_data: str) -> list[str]:
                if not event_data:
                    return []
                try:
                    split_values = dialog.tk.splitlist(event_data)
                except Exception:
                    split_values = [event_data]
                return [str(item) for item in split_values if str(item).strip()]

            def on_drop_to_pool(event: Any) -> str:
                raw_paths = decode_dropped_paths(str(getattr(event, "data", "")))
                process_drop_on_ui_thread(raw_paths, to_merge=False, target_name="LoRAs")
                return "break"

            def on_drop_to_merge(event: Any) -> str:
                raw_paths = decode_dropped_paths(str(getattr(event, "data", "")))
                process_drop_on_ui_thread(raw_paths, to_merge=True, target_name="Merge order")
                return "break"

            try:
                candidate_list.drop_target_register(self.DND_FILES)
                candidate_list.dnd_bind("<<Drop>>", on_drop_to_pool)
                merge_list.drop_target_register(self.DND_FILES)
                merge_list.dnd_bind("<<Drop>>", on_drop_to_merge)
                return True
            except Exception:
                return False

        def remove_from_merge_list() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices:
                return
            for index in reversed(selected_indices):
                merge_loras.pop(index)
            refresh_merge_list()

        def move_merge_up() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices or selected_indices[0] == 0:
                return
            for index in selected_indices:
                merge_loras[index - 1], merge_loras[index] = merge_loras[index], merge_loras[index - 1]
            refresh_merge_list()
            for index in [i - 1 for i in selected_indices]:
                merge_list.selection_set(index)

        def move_merge_down() -> None:
            selected_indices = list(merge_list.curselection())
            if not selected_indices or selected_indices[-1] >= len(merge_loras) - 1:
                return
            for index in reversed(selected_indices):
                merge_loras[index + 1], merge_loras[index] = merge_loras[index], merge_loras[index + 1]
            refresh_merge_list()
            for index in [i + 1 for i in selected_indices]:
                merge_list.selection_set(index)

        def clear_merge_list() -> None:
            merge_loras.clear()
            refresh_merge_list()

        def browse_output_folder() -> None:
            picked = self.filedialog.askdirectory(parent=dialog, title="Select output folder")
            if picked:
                output_dir_var.set(picked)

        def resolve_output_base() -> tuple[Path, str] | None:
            output_name = output_name_var.get().strip()
            if not output_name:
                self.messagebox.showerror("Missing value", "Output name is required.", parent=dialog)
                return None

            output_folder_raw = output_dir_var.get().strip()
            if not output_folder_raw:
                self.messagebox.showerror("Missing value", "Output folder is required.", parent=dialog)
                return None

            output_folder = Path(output_folder_raw).expanduser()
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.messagebox.showerror("Invalid folder", f"Could not use output folder:\n{exc}", parent=dialog)
                return None

            output_name = output_name[:-12] if output_name.lower().endswith(".safetensors") else output_name
            return output_folder, output_name

        def build_output_path(output_folder: Path, output_name: str, mode_suffix: str, preset_name: str) -> Path:
            preset_token = self.merge_preset_file_token(preset_name)
            return output_folder / f"{output_name}_{mode_suffix}_{preset_token}.safetensors"

        def run_merge() -> None:
            if len(merge_loras) < 2:
                self.messagebox.showerror("Merge unavailable", "Add at least 2 LoRAs to merge list.", parent=dialog)
                return

            selected_preset_names: list[str] = []
            if preset_balanced_var.get():
                selected_preset_names.append("Balanced")
            if preset_smooth_var.get():
                selected_preset_names.append("Smooth")
            if preset_anti_overfit_var.get():
                selected_preset_names.append("Anti-overfit")
            if not selected_preset_names:
                self.messagebox.showerror("Merge unavailable", "Select at least one preset.", parent=dialog)
                return

            mode_defs: list[tuple[str, str, str]] = []
            if mode_beta_var.get():
                mode_defs.append(("BETA", "Beta", "beta"))
            if mode_beta2_var.get():
                mode_defs.append(("BETA2", "Beta2", "beta2"))
            if mode_sigma_var.get():
                mode_defs.append(("SIGMA_REL", "Sigma", "sigma_rel"))
            if not mode_defs:
                self.messagebox.showerror("Merge unavailable", "Select at least one merge mode.", parent=dialog)
                return

            output_base = resolve_output_base()
            if output_base is None:
                return
            output_folder, output_name = output_base

            selected_jobs: list[tuple[str, str, list[str], str]] = []
            for preset_name in selected_preset_names:
                preset_args = self.post_hoc_ema_mode_args_for_preset(preset_name)
                for mode_label, mode_suffix, mode_key in mode_defs:
                    selected_jobs.append((mode_label, mode_suffix, preset_args[mode_key], preset_name))

            existing_outputs: list[Path] = []
            for _merge_mode_label, merge_mode_suffix, _merge_mode_args, preset_name in selected_jobs:
                candidate = build_output_path(output_folder, output_name, merge_mode_suffix, preset_name)
                if candidate.exists():
                    existing_outputs.append(candidate)
            if existing_outputs:
                existing_text = "\n".join(path.stem if path.suffix.lower() == ".safetensors" else path.name for path in existing_outputs)
                self.messagebox.showerror(
                    "Name already exists",
                    f"One or more output files already exist:\n{existing_text}\n\nChoose a different output name.",
                    parent=dialog,
                )
                return

            selected_files = [str(path) for path in merge_loras]

            run_env = self.os.environ.copy()
            musubi_src = str(self.runtime_config.musubi_dir / "src")
            existing_pythonpath = run_env.get("PYTHONPATH", "")
            run_env["PYTHONPATH"] = musubi_src if not existing_pythonpath else f"{musubi_src}{self.os.pathsep}{existing_pythonpath}"

            self.log("")
            created_paths: list[Path] = []
            for merge_mode_label, merge_mode_suffix, merge_mode_args, preset_name in selected_jobs:
                output_path = build_output_path(output_folder, output_name, merge_mode_suffix, preset_name)
                command: list[str]
                if module_script.exists():
                    command = [
                        str(musubi_python),
                        "-m",
                        "musubi_tuner.lora_post_hoc_ema",
                        *selected_files,
                        "--output_file",
                        str(output_path),
                        "--no_sort",
                        *merge_mode_args,
                    ]
                else:
                    command = [
                        str(musubi_python),
                        str(root_script),
                        *selected_files,
                        "--output_file",
                        str(output_path),
                        "--no_sort",
                        *merge_mode_args,
                    ]

                self.log(
                    f"[LoRA Post-Hoc EMA Merge] Merging {len(selected_files)} LoRA(s) "
                    f"using {merge_mode_label} / {preset_name}..."
                )
                self.log(f"[LoRA Post-Hoc EMA Merge] Output: {output_path}")

                result = self.subprocess.run(
                    command,
                    cwd=str(self.runtime_config.musubi_dir),
                    env=run_env,
                    capture_output=True,
                    text=True,
                )

                if result.stdout.strip():
                    self.log(result.stdout.strip())

                if result.returncode != 0:
                    self.log(
                        f"[LoRA Post-Hoc EMA Merge] Failed ({result.returncode}) "
                        f"while running {merge_mode_label} / {preset_name}."
                    )
                    if result.stderr.strip():
                        self.log(result.stderr.strip())
                    self.messagebox.showerror(
                        "Merge failed",
                        result.stderr.strip() or "lora_post_hoc_ema.py failed with no error output.",
                        parent=dialog,
                    )
                    return

                if result.stderr.strip():
                    self.log(result.stderr.strip())
                self.log(f"[LoRA Post-Hoc EMA Merge] Created ({merge_mode_suffix} / {preset_name}): {output_path}")
                created_paths.append(output_path)

            created_text = "\n".join(path.stem if path.suffix.lower() == ".safetensors" else path.name for path in created_paths)
            self.messagebox.showinfo("Merge complete", f"Created:\n{created_text}", parent=dialog)

        candidate_list.bind("<Double-Button-1>", on_candidate_double_click)

        candidate_actions = self.ttk.Frame(candidate_section)
        candidate_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.ttk.Button(candidate_actions, text="Add LoRA", command=add_loras_to_pool).grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.ttk.Button(candidate_actions, text="Add to Merge List >>", command=add_to_merge_list).grid(row=0, column=1, sticky="w")

        dnd_enabled = try_enable_file_dnd()
        dnd_hint = (
            "Tip: Drag .safetensors files onto LoRAs or Merge order lists."
            if dnd_enabled
            else "Tip: Drag-and-drop requires tkinterdnd2 in app Python (pip install tkinterdnd2). Use Add LoRA for now."
        )
        self.ttk.Label(candidate_section, text=dnd_hint, style="Dim.TLabel", wraplength=320).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        merge_actions = self.ttk.Frame(merge_section)
        merge_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.ttk.Button(merge_actions, text="Remove", command=remove_from_merge_list).grid(row=0, column=0, padx=(0, 6))
        self.ttk.Button(merge_actions, text="Up", command=move_merge_up).grid(row=0, column=1, padx=(0, 6))
        self.ttk.Button(merge_actions, text="Down", command=move_merge_down).grid(row=0, column=2, padx=(0, 6))
        self.ttk.Button(merge_actions, text="Clear", command=clear_merge_list).grid(row=0, column=3)

        self.ttk.Button(mode_section, text="Browse", command=browse_output_folder).grid(row=6, column=2, padx=(8, 0), pady=(8, 0))

        self.ttk.Button(actions, text="Close", command=dialog.destroy).grid(row=0, column=1, padx=(0, 8))
        self.ttk.Button(actions, text="Go", command=run_merge).grid(row=0, column=2)

        dialog.geometry("820x620")
        self.center_window(dialog)
        dialog.deiconify()
        self.root.wait_window(dialog)
