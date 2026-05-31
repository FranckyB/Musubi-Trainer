"""LTX-2.3 training module for Musubi-Trainer.

Uses:
  ltx2_cache_latents.py
  ltx2_cache_text_encoder_outputs.py
  ltx2_train_network.py

LTX-2.3 has no separate VAE — ``ltx2_checkpoint`` is the single model file.
Gemma text encoder can be provided as either ``--gemma_root`` (folder)
or ``--gemma_safetensors`` (single file).
"""

from __future__ import annotations

import shlex
import re
import json
from datetime import datetime
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
    module_available,
    require_model_file,
)

# LTX-2.3 latent cache suffix
LATENT_SUFFIX = "ltx23"
LTX_VERSION = "2.3"

# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────


def _output_name_default(dataset_name: str) -> str:
    return f"{dataset_name}_LTX"


def _normalize_ltx_mode(mode_value: str | None) -> str:
    value = (mode_value or "video").strip().lower()
    if value in {"video", "v", "image"}:
        return "video"
    if value in {"av", "va"}:
        return "av"
    if value in {"audio", "a"}:
        return "audio"
    return "video"


def _looks_like_sharded_safetensor(path_value: Path) -> bool:
    name = path_value.name.lower()
    if name.endswith(".safetensors.index.json"):
        return True
    if not name.endswith(".safetensors"):
        return False
    # HuggingFace shard naming: model-00001-of-00005.safetensors
    return re.search(r"-\d{5}-of-\d{5}\.safetensors$", name) is not None


def _resolve_ltx_text_encoder(path_value: Path | None) -> tuple[str, Path]:
    if path_value is None:
        raise RuntimeError(
            "LTX-2.3 Text Encoder is not configured. Open Settings and set Gemma root folder or Gemma safetensors."
        )
    candidate = path_value.expanduser()
    if not candidate.exists():
        raise RuntimeError(f"LTX-2.3 Text Encoder path does not exist: {candidate}")
    if candidate.is_dir():
        return "--gemma_root", candidate
    if _looks_like_sharded_safetensor(candidate):
        return "--gemma_root", candidate.parent
    normalized = require_model_file(candidate, "LTX-2.3 Text Encoder")
    return "--gemma_safetensors", normalized


def _latest_checkpoint(training_dir: Path, dataset_name: str) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    return latest_checkpoint_for_output(output_dir, _output_name_default(dataset_name))


def _latest_resume_state(training_dir: Path, dataset_name: str, checkpoint_step: int) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    return latest_resume_state_for_output(output_dir, _output_name_default(dataset_name), checkpoint_step)


def _finished_checkpoint(training_dir: Path, dataset_name: str) -> Path | None:
    output_dir = training_dir / dataset_name / "output"
    return finished_checkpoint_for_output(output_dir, _output_name_default(dataset_name))


# ────────────────────────────────────────────────────────────────────────────
# Core training pipeline
# ────────────────────────────────────────────────────────────────────────────


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
    lr_scheduler: str,
    lr_warmup_steps: int,
    gradient_accumulation_steps: int,
    blocks_to_swap: int,
    timestep_sampling: str,
    ltx_lora_target_preset: str,
    ltx_first_frame_conditioning_p: float,
    train_steps: int,
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    enable_compile_optimizations: bool = False,
    enable_cuda_allow_tf32: bool = False,
    enable_cuda_cudnn_benchmark: bool = False,
    enable_fp8_dit: bool = False,
    enable_gradient_checkpointing_cpu_offload: bool = False,
    enable_training_logging: bool = False,
    training_log_backend: str = "tensorboard",
    training_log_tracker_name: str = "",
    stream_training_output: bool = True,
    ltx_mode: str = "video",
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
    musubi_dir = Path(runtime_config.musubi_dir)
    training_dir = Path(runtime_config.training_dir)

    output_dir = (output_dir_override or training_dir / model_name / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = (output_name_override or "").strip() or _output_name_default(model_name)
    train_steps_for_run = train_steps_override if train_steps_override is not None else train_steps
    ltx_mode_value = _normalize_ltx_mode(ltx_mode)

    dataset_config = training_dir / model_name / "dataset.toml"

    # ── Step 1: prep dataset ────────────────────────────────────────────────
    if do_prep_dataset:
        logger(f"[1/4] Preparing dataset: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        result = prep_dataset_minimal(training_dir, model_name, default_caption_keyword, resolution)
        logger(f"  dataset.toml {'already existed' if result['had_dataset_toml'] else 'created'}")
        logger(f"  captions created: {result['created']}")

    if not dataset_config.is_file():
        raise RuntimeError(
            f"dataset.toml not found for dataset '{model_name}' at {dataset_config}. "
            "Run Step 1 (Prepare Dataset) first."
        )

    # ── Resolve model paths ─────────────────────────────────────────────────
    dit_path = require_model_file(runtime_config.dit, "LTX-2.3 Model")
    gemma_flag, gemma_path = _resolve_ltx_text_encoder(runtime_config.text_encoder)

    # ── Step 2: cache latents ───────────────────────────────────────────────
    if do_cache_latents:
        logger(f"[2/4] Caching latents: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        cache_latents_args = [
            str(musubi_python),
            "ltx2_cache_latents.py",
            "--dataset_config", str(dataset_config),
            "--ltx2_checkpoint", str(dit_path),
            "--skip_existing",
            "--ltx2_mode", ltx_mode_value,
        ]
        logger(f"  command: {format_command_for_log(cache_latents_args)}")
        try:
            run_command(
                cache_latents_args,
                cwd=musubi_dir,
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
                "Latent caching failed for LTX-2.3.\n"
                "Verify LTX-2.3 Model path in Settings.\n"
                f"Details: {exc}"
            ) from exc

    # ── Step 3: cache text encoder outputs ─────────────────────────────────
    if do_cache_text:
        logger(f"[3/4] Caching text encoder outputs: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        cache_text_args = [
            str(musubi_python),
            "ltx2_cache_text_encoder_outputs.py",
            "--dataset_config", str(dataset_config),
            "--ltx2_checkpoint", str(dit_path),
            gemma_flag, str(gemma_path),
            "--device", "cuda",
            "--mixed_precision", "bf16",
            "--batch_size", "1",
            "--skip_existing",
            "--ltx2_mode", ltx_mode_value,
        ]
        if gemma_flag == "--gemma_root":
            cache_text_args.append("--gemma_load_in_4bit")
        logger(f"  command: {format_command_for_log(cache_text_args)}")
        try:
            run_command(
                cache_text_args,
                cwd=musubi_dir,
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
                "Text encoder caching failed for LTX-2.3.\n"
                "Verify LTX-2.3 Text Encoder path in Settings.\n"
                "If using a Gemma safetensors file, try a Gemma root folder for 4-bit loading.\n"
                f"Details: {exc}"
            ) from exc

    # ── Step 4: train ───────────────────────────────────────────────────────
    if do_train:
        logger(f"[4/4] Training: {model_name}  output_name={output_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError

        logging_flags: list[str] = []
        if enable_training_logging:
            log_backend = training_log_backend.strip().lower() or "tensorboard"
            if log_backend == "all":
                if not module_available("tensorboard") and not module_available("wandb"):
                    raise RuntimeError(
                        "Logging backend 'all' requires tensorboard and/or wandb, but neither is installed."
                    )
            logging_dir, auto_tracker_name = next_dataset_log_run_dir(training_dir, model_name)
            logging_flags.extend(["--log_with", log_backend, "--logging_dir", str(logging_dir)])
            tracker_name = training_log_tracker_name.strip() or auto_tracker_name
            if tracker_name:
                logging_flags.extend(["--log_tracker_name", tracker_name])

        optimizer_key = (optimizer_type or "adamw8bit").strip().lower()
        optimizer_arg = "prodigyopt.Prodigy" if optimizer_key == "prodigy" else (optimizer_type or "adamw8bit").strip()
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate

        train_args = [
            str(musubi_python),
            "ltx2_train_network.py",
            "--ltx_version", LTX_VERSION,
            "--ltx2_checkpoint", str(dit_path),
            gemma_flag, str(gemma_path),
            "--ltx2_mode", ltx_mode_value,
            "--dataset_config", str(dataset_config),
            "--output_dir", str(output_dir),
            "--output_name", output_name,
            "--network_dim", str(network_dim),
            "--network_alpha", str(network_alpha),
            "--optimizer_type", optimizer_arg,
            "--learning_rate", learning_rate_for_run,
            "--lr_scheduler", str(lr_scheduler or "constant"),
            "--lr_warmup_steps", str(max(0, int(lr_warmup_steps))),
            "--gradient_accumulation_steps", str(max(1, int(gradient_accumulation_steps))),
            "--timestep_sampling", str(timestep_sampling or "sigma"),
            "--lora_target_preset", str(ltx_lora_target_preset or "t2v"),
            "--ltx2_first_frame_conditioning_p", str(float(ltx_first_frame_conditioning_p)),
            "--max_train_steps", str(train_steps_for_run),
            "--mixed_precision", "bf16",
            "--sdpa",
            "--gradient_checkpointing",
            "--save_every_n_steps", str(save_every_n_steps),
            "--save_state",
            "--save_state_on_train_end",
            "--seed", "42",
        ]
        if int(blocks_to_swap) > 0:
            train_args += ["--blocks_to_swap", str(int(blocks_to_swap))]
        if enable_fp8_dit:
            train_args += ["--fp8_base", "--fp8_scaled"]
        if enable_gradient_checkpointing_cpu_offload:
            train_args.append("--gradient_checkpointing_cpu_offload")
        if enable_compile_optimizations:
            train_args.append("--compile")
            if enable_cuda_allow_tf32:
                train_args.append("--cuda_allow_tf32")
            if enable_cuda_cudnn_benchmark:
                train_args.append("--cuda_cudnn_benchmark")
        optimizer_args_values: list[str] = []
        if optimizer_key == "prodigy":
            optimizer_args_values = [
                "safeguard_warmup=True",
                "use_bias_correction=True",
                "weight_decay=0.01",
                "betas=(0.9,0.99)",
            ]
            optimizer_args_raw = (optimizer_args or "").strip()
            if optimizer_args_raw:
                optimizer_args_values = [value for value in shlex.split(optimizer_args_raw) if value.strip()]
        if optimizer_args_values:
            train_args += ["--optimizer_args", *optimizer_args_values]
        if warmstart_checkpoint is not None:
            train_args += ["--network_weights", str(warmstart_checkpoint)]
        if resume_state_dir is not None:
            train_args += ["--resume", str(resume_state_dir)]
        train_args.extend(logging_flags)

        logger(f"  command: {format_command_for_log(train_args)}")
        try:
            run_command(
                train_args,
                cwd=musubi_dir,
                cancel_requested=cancel_requested,
                logger=logger,
                stream_to_logger=stream_training_output,
                stream_mode="plain",
                inherit_io=not stream_training_output,
            )
        except TrainingCancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Training launch failed for LTX-2.3.\n"
                "Verify Settings > LTX > Model and Text Encoder paths.\n"
                f"Details: {exc}"
            ) from exc

    logger("")


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


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
    lr_scheduler: str,
    lr_warmup_steps: int,
    gradient_accumulation_steps: int,
    blocks_to_swap: int,
    timestep_sampling: str,
    ltx_lora_target_preset: str,
    ltx_first_frame_conditioning_p: float,
    train_steps: int,
    enable_compile_optimizations: bool,
    enable_cuda_allow_tf32: bool,
    enable_cuda_cudnn_benchmark: bool,
    enable_fp8_dit: bool,
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
    ltx_mode: str = "video",
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
            blocks_to_swap=blocks_to_swap,
            timestep_sampling=timestep_sampling,
            ltx_lora_target_preset=ltx_lora_target_preset,
            ltx_first_frame_conditioning_p=ltx_first_frame_conditioning_p,
            train_steps=train_steps,
            save_every_n_steps=save_every_n_steps,
            enable_compile_optimizations=enable_compile_optimizations,
            enable_cuda_allow_tf32=enable_cuda_allow_tf32,
            enable_cuda_cudnn_benchmark=enable_cuda_cudnn_benchmark,
            enable_fp8_dit=enable_fp8_dit,
            enable_gradient_checkpointing_cpu_offload=enable_gradient_checkpointing_cpu_offload,
            enable_training_logging=enable_training_logging,
            training_log_backend=training_log_backend,
            training_log_tracker_name=training_log_tracker_name,
            stream_training_output=stream_training_output,
            ltx_mode=ltx_mode,
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
