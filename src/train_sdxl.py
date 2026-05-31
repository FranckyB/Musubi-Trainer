"""SDXL LoRA training module using sd-scripts backend.

Uses:
  - tools/cache_latents.py (--sdxl)
  - tools/cache_text_encoder_outputs.py (--sdxl)
  - sdxl_train_network.py
"""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path
from typing import Callable

from .runtime_config import RuntimeConfig
from .train_utils import (
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
    resume_state_dir: Path | None = None,
    resume_step_offset: int = 0,
    warmstart_checkpoint: Path | None = None,
    train_steps_override: int | None = None,
    output_name_override: str | None = None,
    output_dir_override: Path | None = None,
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
            "--skip_cache_check",
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
            "--skip_cache_check",
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

        logging_flags: list[str] = []
        if enable_training_logging:
            log_backend = training_log_backend.strip().lower() or "tensorboard"
            logging_dir, auto_tracker_name = next_dataset_log_run_dir(training_dir, model_name)
            logging_flags.extend(["--log_with", log_backend, "--logging_dir", str(logging_dir)])
            tracker_name = training_log_tracker_name.strip() or auto_tracker_name
            if tracker_name:
                logging_flags.extend(["--log_tracker_name", tracker_name])

        optimizer_key = (optimizer_type or "adamw8bit").strip().lower()
        optimizer_arg = "Prodigy" if optimizer_key == "prodigy" else (optimizer_type or "adamw8bit").strip()
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate
        scheduler_key = str(lr_scheduler or "constant").strip().lower()
        warmup_steps = max(0, int(lr_warmup_steps))
        if scheduler_key == "constant" and warmup_steps > 0:
            logger("  lr_warmup_steps is ignored for constant scheduler; forcing to 0")
            warmup_steps = 0

        train_args = [
            str(musubi_python),
            "sdxl_train_network.py",
            "--pretrained_model_name_or_path", str(sdxl_base),
            "--dataset_config", str(dataset_config_for_sd),
            "--output_dir", str(output_dir),
            "--output_name", output_name,
            "--network_module", "networks.lora",
            "--network_dim", str(network_dim),
            "--network_alpha", str(network_alpha),
            "--optimizer_type", optimizer_arg,
            "--learning_rate", learning_rate_for_run,
            "--lr_scheduler", scheduler_key,
            "--lr_warmup_steps", str(warmup_steps),
            "--gradient_accumulation_steps", str(max(1, int(gradient_accumulation_steps))),
            "--max_train_steps", str(train_steps_for_run),
            "--persistent_data_loader_workers",
            "--save_every_n_steps", str(save_every_n_steps),
            "--mixed_precision", "bf16",
            "--sdpa",
            "--cache_latents",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--save_state",
            "--save_state_on_train_end",
            "--seed", "42",
        ]
        if enable_gradient_checkpointing:
            train_args.append("--gradient_checkpointing")
        if vae_override is not None:
            train_args += ["--vae", str(vae_override)]

        if enable_compile_optimizations:
            train_args.append("--torch_compile")
        if enable_fp8_dit:
            train_args.append("--fp8_base_unet")
        if enable_gradient_checkpointing_cpu_offload:
            train_args.append("--cpu_offload_checkpointing")
        if enable_cuda_allow_tf32 or enable_cuda_cudnn_benchmark:
            logger("  cuda tuning note: sd-scripts SDXL path does not expose tf32/cudnn flags here; using script defaults")

        if optimizer_args.strip():
            parsed = shlex.split(optimizer_args)
            if parsed:
                train_args += ["--optimizer_args", *parsed]

        if unet_lr.strip():
            train_args += ["--unet_lr", unet_lr.strip()]
        if text_encoder_lr.strip():
            train_args += ["--text_encoder_lr", text_encoder_lr.strip()]

        if resume_state_dir is not None:
            train_args += ["--resume", str(resume_state_dir)]
        if warmstart_checkpoint is not None:
            train_args += ["--network_weights", str(warmstart_checkpoint)]

        train_args += logging_flags

        logger(f"  command: {format_command_for_log(train_args)}")
        try:
            run_command(
                train_args,
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

    if not (do_prep_dataset or do_cache_latents or do_cache_text or do_train):
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

    if progress_step >= train_steps:
        logger(f"Job already complete at step {progress_step}; nothing to run.")
        if auto_cleanup_states:
            cleanup_step_states_for_completed_output(output_dir_resolved, output_name_resolved, logger)
        write_recorded_completed_steps(output_dir_resolved, output_name_resolved, progress_step, train_steps)
        return JOB_EXIT_SUCCESS

    if resume_state_dir is not None and resume_state_step >= resume_step:
        effective_resume_state = resume_state_dir
        if finished_checkpoint is not None:
            effective_warmstart_checkpoint = finished_checkpoint
        known_step = max(resume_step, resume_state_step)
        resume_step_offset = known_step
        if known_step > 0:
            train_steps_override = max(1, train_steps - known_step)
        logger(f"  resuming optimizer state from {resume_state_dir.name} (step {resume_state_step})")
    elif resume_checkpoint is not None and resume_step > 0:
        effective_warmstart_checkpoint = resume_checkpoint
        train_steps_override = max(1, train_steps - resume_step)
        logger(f"  warm-starting from {resume_checkpoint.name} (step {resume_step})")

    try:
        run_steps_for_model(
            runtime_config,
            dataset_name,
            default_caption_keyword=default_caption_keyword,
            resolution=resolution,
            network_dim=network_dim,
            network_alpha=network_alpha,
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
            do_train=do_train,
            resume_state_dir=effective_resume_state,
            resume_step_offset=resume_step_offset,
            warmstart_checkpoint=effective_warmstart_checkpoint,
            train_steps_override=train_steps_override,
            output_name_override=output_name,
            output_dir_override=output_dir,
            logger=logger,
            cancel_requested=cancel_requested,
        )
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
