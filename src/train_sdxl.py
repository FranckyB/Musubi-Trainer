"""SDXL LoRA training module using sd-scripts backend.

Uses:
  - tools/cache_latents.py (--sdxl)
  - tools/cache_text_encoder_outputs.py (--sdxl)
  - sdxl_train_network.py
"""

from __future__ import annotations

import os
import shlex
import tomllib
from pathlib import Path
from typing import Callable

from .runtime_config import RuntimeConfig
from .train_utils import (
    build_config_file_command,
    clear_dataset_cache_directories,
    run_command,
    TrainingCancelledError,
    JOB_EXIT_SUCCESS,
    JOB_EXIT_FAILED,
    JOB_EXIT_CANCELLED,
    DEFAULT_NETWORK_DIM,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_SAVE_EVERY_N_STEPS,
    DEFAULT_RESOLUTION,
    prep_dataset_minimal,
    format_command_for_log,
    latest_checkpoint_for_output,
    latest_resume_state_for_output,
    finished_checkpoint_for_output,
    remap_resume_artifacts_for_output,
    cleanup_step_states_for_completed_output,
    cleanup_step_states_for_cancel_output,
    read_recorded_completed_steps,
    write_recorded_completed_steps,
    next_dataset_log_run_dir,
    require_model_file,
)


LATENT_SUFFIX = "sdxl"


def _output_name_default(dataset_name: str) -> str:
    return f"{dataset_name}_SDXL"


def _toml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _toml_list(values: list[object]) -> str:
    return "[" + ", ".join(_toml_scalar(v) for v in values) + "]"


def _network_settings(network_type: str, lora_module: str) -> tuple[str, bool]:
    selected = (network_type or "lora").strip().lower()
    use_lokr = selected == "lokr"
    return ("networks.lokr" if use_lokr else lora_module, use_lokr)


def _recommended_sdxl_dataloader_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 2))


def _sd_scripts_dataset_config(dataset_config: Path, logger: Callable[[str], None]) -> Path:
    try:
        parsed = tomllib.loads(dataset_config.read_text(encoding="utf-8"))
    except Exception:
        return dataset_config

    datasets_raw = parsed.get("datasets")
    if not isinstance(datasets_raw, list):
        return dataset_config

    for entry in datasets_raw:
        if not isinstance(entry, dict):
            continue
        subsets = entry.get("subsets")
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if isinstance(subset, dict) and str(subset.get("image_dir", "")).strip():
                return dataset_config

    general = parsed.get("general") if isinstance(parsed.get("general"), dict) else {}

    subset_rows: list[tuple[str, int]] = []
    for entry in datasets_raw:
        if not isinstance(entry, dict):
            continue
        image_dir = str(entry.get("image_dir") or entry.get("image_directory") or "").strip()
        if not image_dir:
            continue
        repeats_raw = entry.get("num_repeats", 1)
        try:
            repeats = max(1, int(repeats_raw))
        except Exception:
            repeats = 1
        subset_rows.append((image_dir, repeats))

    if not subset_rows:
        return dataset_config

    out_path = dataset_config.with_name("dataset.sdxl.toml")
    lines: list[str] = []
    if general:
        lines.append("[general]")
        for key in (
            "resolution",
            "caption_extension",
            "batch_size",
            "enable_bucket",
            "bucket_no_upscale",
        ):
            if key not in general:
                continue
            value = general[key]
            if isinstance(value, list):
                lines.append(f"{key} = {_toml_list(list(value))}")
            else:
                lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")

    lines.append("[[datasets]]")
    for image_dir, repeats in subset_rows:
        lines.append("  [[datasets.subsets]]")
        lines.append(f"  image_dir = {_toml_scalar(image_dir)}")
        lines.append(f"  num_repeats = {repeats}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger(f"  normalized dataset config for sd-scripts: {out_path.name}")
    return out_path


def run_steps_for_model(
    runtime_config: RuntimeConfig,
    model_name: str,
    *,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    network_type: str,
    lokr_factor: int,
    optimizer_type: str,
    optimizer_args: str,
    learning_rate: str,
    train_steps: int,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    unet_lr: str = "",
    text_encoder_lr: str = "",
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    enable_compile_optimizations: bool = False,
    enable_cuda_allow_tf32: bool = False,
    enable_cuda_cudnn_benchmark: bool = False,
    enable_fp8_dit: bool = False,
    enable_gradient_checkpointing: bool = True,
    enable_gradient_checkpointing_cpu_offload: bool = False,
    enable_training_logging: bool = False,
    training_log_backend: str = "tensorboard",
    training_log_tracker_name: str = "",
    stream_training_output: bool = True,
    do_prep_dataset: bool = True,
    do_cache_latents: bool = True,
    do_cache_text: bool = True,
    do_train: bool = True,
    max_data_loader_n_workers: int | None = None,
    resume_state_dir: Path | None = None,
    resume_step_offset: int = 0,
    warmstart_checkpoint: Path | None = None,
    train_steps_override: int | None = None,
    output_name_override: str | None = None,
    output_dir_override: Path | None = None,
    generate_training_args_only: bool = False,
    logger: Callable[[str], None] = print,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    musubi_python = runtime_config.musubi_python
    backend_dir = Path(runtime_config.musubi_dir)
    training_dir = Path(runtime_config.training_dir)

    if musubi_python is None or not musubi_python.is_file():
        raise RuntimeError("Python executable was not found for the configured backend environment.")

    output_dir = (output_dir_override or training_dir / model_name / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (output_name_override or "").strip() or _output_name_default(model_name)
    train_steps_for_run = train_steps_override if train_steps_override is not None else train_steps

    dataset_config = training_dir / model_name / "dataset.toml"
    model_label = (model_name or "SDXL").strip()

    # Step 1: Prepare dataset
    if do_prep_dataset:
        logger(f"[1/4] Preparing dataset: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        result = prep_dataset_minimal(training_dir, model_name, default_caption_keyword, resolution)
        logger(f"  dataset.toml {'already existed' if result['had_dataset_toml'] else 'created'}")
        logger(f"  captions created: {result['created']}")

    if not dataset_config.is_file():
        raise RuntimeError(
            f"dataset.toml not found for '{model_name}' at {dataset_config}. "
            "Run Step 1 (Prepare Dataset) first."
        )

    if do_cache_latents or do_cache_text:
        cleared_cache_dirs = clear_dataset_cache_directories(dataset_config, logger)
        if cleared_cache_dirs > 0:
            plural = "y" if cleared_cache_dirs == 1 else "ies"
            logger(f"  cache reset: cleared {cleared_cache_dirs} cache director{plural}")

    dataset_config_for_sd = _sd_scripts_dataset_config(dataset_config, logger)

    # Resolve model paths
    sdxl_base = require_model_file(runtime_config.dit, "SDXL base checkpoint")
    vae_override: Path | None = None
    if runtime_config.vae is not None:
        vae_override = require_model_file(runtime_config.vae, "SDXL VAE")

    # Step 2: Cache latents
    if do_cache_latents:
        logger(f"[2/4] Caching latents: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError

        cache_latents_args = [
            str(musubi_python),
            "-m",
            "tools.cache_latents",
            "--dataset_config", str(dataset_config_for_sd),
            "--pretrained_model_name_or_path", str(sdxl_base),
            "--sdxl",
            "--no_half_vae",
        ]
        if vae_override is not None:
            cache_latents_args += ["--vae", str(vae_override)]

        logger(f"  command: {format_command_for_log(cache_latents_args)}")
        try:
            run_command(
                cache_latents_args,
                cwd=backend_dir,
                cancel_requested=cancel_requested,
                logger=logger,
                stream_to_logger=stream_training_output,
                stream_mode="cache_progress",
                inherit_io=not stream_training_output,
            )
        except TrainingCancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Latent caching failed for {model_label}.\n"
                "Verify SDXL-family base checkpoint and optional VAE path in Settings.\n"
                f"Details: {exc}"
            ) from exc

    # Step 3: Cache text encoder outputs
    if do_cache_text:
        logger(f"[3/4] Caching text encoder outputs: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError

        cache_text_args = [
            str(musubi_python),
            "-m",
            "tools.cache_text_encoder_outputs",
            "--dataset_config", str(dataset_config_for_sd),
            "--pretrained_model_name_or_path", str(sdxl_base),
            "--sdxl",
        ]

        logger(f"  command: {format_command_for_log(cache_text_args)}")
        try:
            run_command(
                cache_text_args,
                cwd=backend_dir,
                cancel_requested=cancel_requested,
                logger=logger,
                stream_to_logger=stream_training_output,
                stream_mode="cache_progress",
                inherit_io=not stream_training_output,
            )
        except TrainingCancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Text encoder caching failed for {model_label}.\n"
                "Verify SDXL-family base checkpoint path in Settings.\n"
                f"Details: {exc}"
            ) from exc

    # Step 4: Train
    if do_train:
        logger(f"[4/4] Training: {model_name}  output_name={output_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError

        log_backend = "tensorboard"
        logging_dir: Path | None = None
        tracker_name = ""
        if enable_training_logging:
            log_backend = training_log_backend.strip().lower() or "tensorboard"
            logging_dir, auto_tracker_name = next_dataset_log_run_dir(training_dir, model_name)
            tracker_name = training_log_tracker_name.strip() or auto_tracker_name

        optimizer_key = (optimizer_type or "adamw8bit").strip().lower()
        optimizer_arg = "Prodigy" if optimizer_key == "prodigy" else (optimizer_type or "adamw8bit").strip()
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate
        scheduler_key = str(lr_scheduler or "constant").strip().lower()
        warmup_steps = max(0, int(lr_warmup_steps))
        if scheduler_key == "constant" and warmup_steps > 0:
            logger("  lr_warmup_steps is ignored for constant scheduler; forcing to 0")
            warmup_steps = 0

        model_lines = [
            f"pretrained_model_name_or_path = {_toml_scalar(str(sdxl_base))}",
        ]
        if vae_override is not None:
            model_lines.append(f"vae = {_toml_scalar(str(vae_override))}")

        data_output_lines = [
            f"dataset_config = {_toml_scalar(str(dataset_config_for_sd))}",
            f"output_dir = {_toml_scalar(str(output_dir))}",
            f"output_name = {_toml_scalar(output_name)}",
        ]
        selected_network_module, is_lokr = _network_settings(network_type, "networks.lora")
        network_lines = [
            f"network_module = {_toml_scalar(selected_network_module)}",
            f"network_dim = {network_dim}",
            f"network_alpha = {network_alpha}",
            "network_train_unet_only = true",
        ]
        if is_lokr and lokr_factor != -1:
            network_lines.append(f"network_args = {_toml_list([f'factor={lokr_factor}'])}")
        optimization_lines = [
            f"optimizer_type = {_toml_scalar(optimizer_arg)}",
            f"learning_rate = {learning_rate_for_run}",
            f"lr_scheduler = {_toml_scalar(scheduler_key)}",
            f"lr_warmup_steps = {warmup_steps}",
            f"gradient_accumulation_steps = {max(1, int(gradient_accumulation_steps))}",
            f"max_train_steps = {train_steps_for_run}",
        ]

        if enable_cuda_allow_tf32 or enable_cuda_cudnn_benchmark:
            logger("  cuda tuning note: sd-scripts SDXL path does not expose tf32/cudnn flags here; using script defaults")

        if optimizer_args.strip():
            parsed = shlex.split(optimizer_args)
            if parsed:
                optimization_lines.append(f"optimizer_args = {_toml_list(parsed)}")

        if unet_lr.strip():
            optimization_lines.append(f"unet_lr = {_toml_scalar(unet_lr.strip())}")
        if text_encoder_lr.strip():
            optimization_lines.append(f"text_encoder_lr = {_toml_scalar(text_encoder_lr.strip())}")

        resolved_workers = max_data_loader_n_workers
        if resolved_workers is None:
            resolved_workers = _recommended_sdxl_dataloader_workers()
        else:
            resolved_workers = max(1, int(resolved_workers))

        runtime_lines = [
            'mixed_precision = "bf16"',
            "sdpa = true",
            "cache_latents = true",
            "cache_text_encoder_outputs = true",
            "cache_text_encoder_outputs_to_disk = true",
            f"gradient_checkpointing = {'true' if enable_gradient_checkpointing else 'false'}",
            f"cpu_offload_checkpointing = {'true' if enable_gradient_checkpointing_cpu_offload else 'false'}",
            f"torch_compile = {'true' if enable_compile_optimizations else 'false'}",
            f"fp8_base_unet = {'true' if enable_fp8_dit else 'false'}",
            "persistent_data_loader_workers = true",
            f"max_data_loader_n_workers = {resolved_workers}",
        ]
        checkpoint_lines = [
            f"save_every_n_steps = {save_every_n_steps}",
            "save_state = true",
            "save_state_on_train_end = true",
            "seed = 42",
        ]

        restore_lines: list[str] = []
        if resume_state_dir is not None:
            restore_lines.append(f"resume = {_toml_scalar(str(resume_state_dir))}")
        if warmstart_checkpoint is not None:
            restore_lines.append(f"network_weights = {_toml_scalar(str(warmstart_checkpoint))}")

        logging_lines: list[str] = []
        if enable_training_logging and logging_dir is not None:
            logging_lines.extend(
                [
                    f"log_with = {_toml_scalar(log_backend)}",
                    f"logging_dir = {_toml_scalar(str(logging_dir))}",
                ]
            )
            if tracker_name:
                logging_lines.append(f"log_tracker_name = {_toml_scalar(tracker_name)}")

        config_lines: list[str] = []
        config_sections: list[tuple[str, list[str]]] = [
            ("Model", model_lines),
            ("Data and Output", data_output_lines),
            ("Network", network_lines),
            ("Optimization", optimization_lines),
            ("Runtime", runtime_lines),
            ("Checkpointing", checkpoint_lines),
        ]
        if restore_lines:
            config_sections.append(("Resume and Warmstart", restore_lines))
        if logging_lines:
            config_sections.append(("Logging", logging_lines))

        for section_name, section_lines in config_sections:
            if config_lines:
                config_lines.append("")
            config_lines.append(f"# {section_name}")
            config_lines.extend(section_lines)

        train_config_path = output_dir.parent / "training_args.toml"
        if generate_training_args_only or not train_config_path.exists():
            train_config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
            logger(f"  training_args: {train_config_path}")
        else:
            logger(f"  training_args: {train_config_path} (using existing)")

        if generate_training_args_only:
            logger("  training args generated (no training launched)")
        else:
            launch_args = build_config_file_command(musubi_python, "sdxl_train_network.py", train_config_path)
            logger(f"  command: {format_command_for_log(launch_args)}")
            try:
                run_command(
                launch_args,
                    cwd=backend_dir,
                    cancel_requested=cancel_requested,
                    logger=logger,
                    stream_to_logger=stream_training_output,
                    stream_mode="train_progress",
                    inherit_io=not stream_training_output,
                )
            except TrainingCancelledError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Training launch failed for {model_label}.\n"
                    "Verify SDXL-family model paths in Settings.\n"
                    f"Details: {exc}"
                ) from exc

    logger("")


def run_job(
    runtime_config: RuntimeConfig,
    dataset_name: str,
    output_name: str,
    output_dir: Path,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    network_type: str,
    lokr_factor: int,
    optimizer_type: str,
    optimizer_args: str,
    learning_rate: str,
    train_steps: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
    gradient_accumulation_steps: int,
    unet_lr: str,
    text_encoder_lr: str,
    enable_compile_optimizations: bool,
    enable_cuda_allow_tf32: bool,
    enable_cuda_cudnn_benchmark: bool,
    enable_fp8_dit: bool,
    enable_gradient_checkpointing: bool,
    enable_gradient_checkpointing_cpu_offload: bool,
    enable_training_logging: bool,
    training_log_backend: str,
    training_log_tracker_name: str,
    stream_training_output: bool,
    auto_cleanup_states: bool,
    logger: Callable[[str], None],
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
    max_data_loader_n_workers: int | None = None,
    generate_training_args_only: bool = False,
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    cancel_requested: Callable[[], bool] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> int:
    if not dataset_name.strip():
        message = "No dataset selected for job."
        logger(message)
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED

    if not output_name.strip():
        message = "Output name is required for job."
        logger(message)
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED

    if not (do_prep_dataset or do_cache_latents or do_cache_text or do_train or generate_training_args_only):
        message = "No steps selected."
        logger(message)
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED

    output_name_resolved = output_name.strip()
    output_dir_resolved = output_dir.resolve()
    resume_checkpoint, resume_step = latest_checkpoint_for_output(output_dir_resolved, output_name_resolved)
    finished_checkpoint = finished_checkpoint_for_output(output_dir_resolved, output_name_resolved)
    resume_state_dir, resume_state_step = latest_resume_state_for_output(
        output_dir_resolved, output_name_resolved, resume_step
    )
    recorded_completed_step = read_recorded_completed_steps(output_dir_resolved, output_name_resolved)
    progress_step = max(resume_step, resume_state_step, recorded_completed_step)
    effective_resume_state: Path | None = None
    resume_step_offset = 0
    effective_warmstart_checkpoint: Path | None = None
    train_steps_override: int | None = None

    if progress_step >= train_steps and not generate_training_args_only:
        logger(f"Job already complete at step {progress_step}; nothing to run.")
        if auto_cleanup_states:
            cleanup_step_states_for_completed_output(output_dir_resolved, output_name_resolved, logger)
        write_recorded_completed_steps(output_dir_resolved, output_name_resolved, progress_step, train_steps)
        return JOB_EXIT_SUCCESS

    if resume_state_dir is not None and resume_state_step >= resume_step:
        effective_resume_state = resume_state_dir
        if finished_checkpoint is not None:
            effective_warmstart_checkpoint = finished_checkpoint
        known_step = max(progress_step, resume_step, resume_state_step)
        resume_step_offset = known_step
        if known_step > 0:
            train_steps_override = max(1, train_steps - known_step)
        logger(f"  resuming optimizer state from {resume_state_dir.name} (step {resume_state_step})")
    elif resume_checkpoint is not None and resume_step > 0:
        effective_warmstart_checkpoint = resume_checkpoint
        train_steps_override = max(1, train_steps - resume_step)
        logger(f"  warm-starting from {resume_checkpoint.name} (step {resume_step})")
    elif finished_checkpoint is not None and progress_step > 0:
        effective_warmstart_checkpoint = finished_checkpoint
        train_steps_override = max(1, train_steps - progress_step)
        logger(
            f"  warm-starting from finished checkpoint {finished_checkpoint.name} "
            f"(recorded step {progress_step})"
        )

    try:
        run_steps_for_model(
            runtime_config,
            dataset_name,
            default_caption_keyword=default_caption_keyword,
            resolution=resolution,
            network_dim=network_dim,
            network_alpha=network_alpha,
            network_type=network_type,
            lokr_factor=lokr_factor,
            optimizer_type=optimizer_type,
            optimizer_args=optimizer_args,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            unet_lr=unet_lr,
            text_encoder_lr=text_encoder_lr,
            train_steps=train_steps,
            save_every_n_steps=save_every_n_steps,
            enable_compile_optimizations=enable_compile_optimizations,
            enable_cuda_allow_tf32=enable_cuda_allow_tf32,
            enable_cuda_cudnn_benchmark=enable_cuda_cudnn_benchmark,
            enable_fp8_dit=enable_fp8_dit,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_gradient_checkpointing_cpu_offload=enable_gradient_checkpointing_cpu_offload,
            enable_training_logging=enable_training_logging,
            training_log_backend=training_log_backend,
            training_log_tracker_name=training_log_tracker_name,
            stream_training_output=stream_training_output,
            do_prep_dataset=do_prep_dataset,
            do_cache_latents=do_cache_latents,
            do_cache_text=do_cache_text,
            do_train=(do_train or generate_training_args_only),
            max_data_loader_n_workers=max_data_loader_n_workers,
            generate_training_args_only=generate_training_args_only,
            resume_state_dir=effective_resume_state,
            resume_step_offset=resume_step_offset,
            warmstart_checkpoint=effective_warmstart_checkpoint,
            train_steps_override=train_steps_override,
            output_name_override=output_name,
            output_dir_override=output_dir,
            logger=logger,
            cancel_requested=cancel_requested,
        )
        if generate_training_args_only:
            return JOB_EXIT_SUCCESS
        if effective_resume_state is not None and resume_step_offset > 0:
            remap_resume_artifacts_for_output(output_dir_resolved, output_name_resolved, resume_step_offset, logger)
        if auto_cleanup_states:
            cleanup_step_states_for_completed_output(output_dir_resolved, output_name_resolved, logger)
        write_recorded_completed_steps(output_dir_resolved, output_name_resolved, train_steps, train_steps)
        logger(f"Job completed: {output_name}")
        return JOB_EXIT_SUCCESS
    except TrainingCancelledError:
        if effective_resume_state is not None and resume_step_offset > 0:
            remap_resume_artifacts_for_output(output_dir_resolved, output_name_resolved, resume_step_offset, logger)
        if auto_cleanup_states:
            cleanup_step_states_for_cancel_output(output_dir_resolved, output_name_resolved, logger)
        logger("Job cancelled by user.")
        return JOB_EXIT_CANCELLED
    except Exception as exc:
        message = str(exc)
        logger(f"Job failed for '{output_name}': {message}")
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED
