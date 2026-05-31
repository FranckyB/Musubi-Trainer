"""FLUX.2 family training — dev, klein-base-9b, klein-9b, klein-base-4b, klein-4b."""
from __future__ import annotations

import re
import shlex
import shutil
import os
import subprocess
from pathlib import Path
from typing import Callable

from .runtime_config import RuntimeConfig
from .train_utils import (  # noqa: F401  (re-exported for backward compat)
    DEFAULT_LEARNING_RATE,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_NETWORK_DIM,
    DEFAULT_RESOLUTION,
    DEFAULT_SAVE_EVERY_N_STEPS,
    DEFAULT_TRAIN_STEPS,
    JOB_EXIT_CANCELLED,
    JOB_EXIT_FAILED,
    JOB_EXIT_SUCCESS,
    JOB_PROGRESS_FILE_NAME,
    SHARDED_SAFETENSORS_NAME_RE,
    VALID_IMAGE_EXTENSIONS,
    TrainingCancelledError,
    _step_state_dirs,
    centralized_logs_root,
    cleanup_step_states_for_cancel_output,
    cleanup_step_states_for_completed_output,
    dataset_image_directory_from_config,
    dataset_image_files,
    finished_checkpoint_for_output,
    format_command_for_log,
    latest_checkpoint_for_output,
    latest_resume_state_for_output,
    module_available,
    next_dataset_log_run_dir,
    normalize_model_checkpoint_path,
    prep_dataset_minimal,
    progress_metadata_path_for_output,
    read_recorded_completed_steps,
    remap_resume_artifacts_for_output,
    require_model_file,
    run_command,
    toml_quote,
    toml_string_list,
    write_recorded_completed_steps,
)

# ---------------------------------------------------------------------------
# FLUX.2-specific constants
# ---------------------------------------------------------------------------
LATENT_SUFFIX = "f2k9b"


def _dataset_output_dir(training_dir: Path, dataset_name: str) -> Path:
    return training_dir / dataset_name / "output"


def _dataset_output_name(dataset_name: str) -> str:
    return f"{dataset_name}_Flux2"


def is_step1_ready(training_dir: Path, dataset_name: str) -> bool:
    dataset_dir = training_dir / dataset_name
    dataset_toml_exists = (dataset_dir / "dataset.toml").exists()
    images = dataset_image_files(training_dir, dataset_name)
    if not dataset_toml_exists or not images:
        return False
    return all(image_path.with_suffix(".txt").exists() for image_path in images)


def count_latent_cache_ready(training_dir: Path, dataset_name: str) -> tuple[int, int]:
    images = dataset_image_files(training_dir, dataset_name)
    total = len(images)
    if total == 0:
        return 0, 0

    cache_dir = training_dir / dataset_name / "cache"
    if not cache_dir.exists() or not cache_dir.is_dir():
        return 0, total

    ready = 0
    for image_path in images:
        pattern = f"{image_path.stem}_*_{LATENT_SUFFIX}.safetensors"
        if any(cache_dir.glob(pattern)):
            ready += 1
    return ready, total


def count_text_cache_ready(training_dir: Path, dataset_name: str) -> tuple[int, int]:
    images = dataset_image_files(training_dir, dataset_name)
    total = len(images)
    if total == 0:
        return 0, 0

    cache_dir = training_dir / dataset_name / "cache"
    if not cache_dir.exists() or not cache_dir.is_dir():
        return 0, total

    ready = 0
    for image_path in images:
        expected = cache_dir / f"{image_path.stem}_{LATENT_SUFFIX}_te.safetensors"
        if expected.exists():
            ready += 1
    return ready, total


def latest_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> tuple[Path | None, int]:
    return latest_checkpoint_for_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
    )


def finished_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> Path | None:
    return finished_checkpoint_for_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
    )


def latest_resume_state_for_dataset(training_dir: Path, dataset_name: str, checkpoint_step: int) -> tuple[Path | None, int]:
    return latest_resume_state_for_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
        checkpoint_step,
    )


def remap_resume_artifacts_to_continued_steps(
    training_dir: Path,
    dataset_name: str,
    resume_step_offset: int,
    logger: Callable[[str], None],
) -> None:
    remap_resume_artifacts_for_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
        resume_step_offset,
        logger,
    )


def cleanup_step_states_for_completed_run(training_dir: Path, dataset_name: str, logger: Callable[[str], None]) -> None:
    cleanup_step_states_for_completed_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
        logger,
    )


def cleanup_step_states_for_cancel(training_dir: Path, dataset_name: str, logger: Callable[[str], None]) -> None:
    cleanup_step_states_for_cancel_output(
        _dataset_output_dir(training_dir, dataset_name),
        _dataset_output_name(dataset_name),
        logger,
    )





def run_steps_for_model(
    runtime_config: RuntimeConfig,
    model_name: str,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    optimizer_type: str,
    optimizer_args: str,
    learning_rate: str,
    train_steps: int,
    save_every_n_steps: int,
    enable_compile_optimizations: bool,
    enable_cuda_allow_tf32: bool,
    enable_cuda_cudnn_benchmark: bool,
    enable_fp8_dit: bool,
    enable_gradient_checkpointing_cpu_offload: bool,
    enable_training_logging: bool,
    training_log_backend: str,
    training_log_tracker_name: str,
    stream_training_output: bool,
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
    resume_state_dir: Path | None,
    resume_step_offset: int,
    warmstart_checkpoint: Path | None,
    train_steps_override: int | None,
    output_name_override: str | None,
    output_dir_override: Path | None,
    logger: Callable[[str], None],
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    def require_musubi_python() -> Path:
        python_path = runtime_config.musubi_python
        if python_path is None or not python_path.is_file():
            raise RuntimeError(
                "Musubi-Tuner Python was not found in its venv. "
                "Expected one of: .venv/Scripts/python.exe or venv/Scripts/python.exe inside Musubi-Tuner."
            )
        return python_path

    def require_model_file(path_value: Path | None, label: str) -> Path:
        if path_value is None:
            raise RuntimeError(f"{label} is not configured. Open Settings and select a file for {label}.")
        normalized = normalize_model_checkpoint_path(path_value, label)
        if not normalized.is_file():
            raise RuntimeError(f"{label} file does not exist: {normalized}")
        return normalized

    def resolve_text_encoder_file(path_value: Path | None) -> Path:
        selected = require_model_file(path_value, "FLUX.2 Text Encoder")
        return selected

    def check_cancel() -> None:
        if cancel_requested is not None and cancel_requested():
            raise TrainingCancelledError("Cancelled by user.")

    def module_available(module_name: str) -> bool:
        result = subprocess.run(
            [str(musubi_python), "-c", f"import {module_name}"],
            cwd=str(runtime_config.musubi_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return result.returncode == 0

    def wandb_key_available() -> bool:
        if os.environ.get("WANDB_API_KEY", "").strip():
            return True
        home_dir = Path.home()
        for netrc_name in (".netrc", "_netrc"):
            netrc_path = home_dir / netrc_name
            if not netrc_path.is_file():
                continue
            try:
                content = netrc_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "api.wandb.ai" in content:
                return True
        return False

    musubi_python = require_musubi_python()

    dataset_config = runtime_config.training_dir / model_name / "dataset.toml"
    output_dir = output_dir_override if output_dir_override is not None else (runtime_config.training_dir / model_name / "output")
    dataset_config = dataset_config.resolve()
    output_dir = output_dir.resolve()
    output_name = (output_name_override or "").strip() or f"{model_name}_Flux2"
    output_name_for_run = (
        f"{output_name}-resume" if (resume_state_dir is not None and resume_step_offset > 0) else output_name
    )

    divider = "=" * 58
    logger(divider)
    logger(f"MODEL_NAME: {model_name}")
    logger(divider)
    logger(f"Starting training for: {model_name}")
    logger(f"dataset_config: {dataset_config}")
    logger(f"output_dir:     {output_dir}")
    logger("")

    check_cancel()

    total_steps = int(do_prep_dataset) + int(do_cache_latents) + int(do_cache_text) + int(do_train)
    current_step = 0

    if do_prep_dataset:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Dataset Check:")
        if is_step1_ready(runtime_config.training_dir, model_name):
            logger("  prep: already ready, skipped")
        else:
            prep_result = prep_dataset_minimal(
                runtime_config.training_dir,
                model_name,
                default_caption_keyword,
                resolution,
            )
            toml_status = "existed" if bool(prep_result["had_dataset_toml"]) else "created"
            logger(f"  prep: dataset.toml {toml_status}, captions created {prep_result['created']}")
        logger("")

    if do_cache_latents:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Cache Latent:")
        vae_path = require_model_file(runtime_config.vae, "FLUX.2 VAE")
        vae_path = vae_path.resolve()
        before_ready, before_total = count_latent_cache_ready(runtime_config.training_dir, model_name)
        if before_total > 0 and before_ready == before_total:
            logger(f"  cache_latents: already ready ({before_ready}/{before_total}), skipped")
        else:
            try:
                run_command(
                    [
                        str(musubi_python),
                        "flux_2_cache_latents.py",
                        "--dataset_config",
                        str(dataset_config),
                        "--vae",
                        str(vae_path),
                        "--batch_size",
                        "16",
                        "--skip_existing",
                        "--model_version",
                        runtime_config.model_version,
                    ],
                    cwd=runtime_config.musubi_dir,
                    cancel_requested=cancel_requested,
                    logger=logger,
                    stream_to_logger=stream_training_output,
                    stream_mode="cache_progress",
                )
            except TrainingCancelledError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    "FLUX.2 VAE appears invalid or incompatible for Flux2 caching.\n"
                    "Open Settings and verify Settings > FLUX.2 > VAE points to the correct VAE checkpoint file.\n"
                    f"Details: {exc}"
                ) from exc

            after_ready, after_total = count_latent_cache_ready(runtime_config.training_dir, model_name)
            generated = max(0, after_ready - before_ready)
            logger(f"  cache_latents: done ({after_ready}/{after_total} ready, +{generated} generated)")
        logger("")

    if do_cache_text:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Cache Text Encoder:")
        text_encoder_path = resolve_text_encoder_file(runtime_config.text_encoder)
        text_encoder_path = text_encoder_path.resolve()
        logger(f"  text_encoder: {text_encoder_path}")
        before_ready, before_total = count_text_cache_ready(runtime_config.training_dir, model_name)
        if before_total > 0 and before_ready == before_total:
            logger(f"  cache_text: already ready ({before_ready}/{before_total}), skipped")
        else:
            try:
                run_command(
                    [
                        str(musubi_python),
                        "flux_2_cache_text_encoder_outputs.py",
                        "--dataset_config",
                        str(dataset_config),
                        "--text_encoder",
                        str(text_encoder_path),
                        "--batch_size",
                        "16",
                        "--skip_existing",
                        "--model_version",
                        runtime_config.model_version,
                    ],
                    cwd=runtime_config.musubi_dir,
                    cancel_requested=cancel_requested,
                    logger=logger,
                    stream_to_logger=stream_training_output,
                    stream_mode="cache_progress",
                )
            except TrainingCancelledError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    "FLUX.2 Text Encoder appears invalid or incompatible for Flux2 text caching.\n"
                    "Open Settings and verify Settings > FLUX.2 > Text Encoder points to the correct checkpoint.\n"
                    f"Details: {exc}"
                ) from exc

            after_ready, after_total = count_text_cache_ready(runtime_config.training_dir, model_name)
            generated = max(0, after_ready - before_ready)
            logger(f"  cache_text: done ({after_ready}/{after_total} ready, +{generated} generated)")
        logger("")

    if do_train:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Train:")
        dit_path = require_model_file(runtime_config.dit, "FLUX.2 Model")
        vae_path = require_model_file(runtime_config.vae, "FLUX.2 VAE")
        text_encoder_path = resolve_text_encoder_file(runtime_config.text_encoder)
        dit_path = dit_path.resolve()
        vae_path = vae_path.resolve()
        text_encoder_path = text_encoder_path.resolve()
        logger(f"  text_encoder: {text_encoder_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_steps_for_run = train_steps_override if train_steps_override is not None else train_steps
        compile_enabled = bool(enable_compile_optimizations and not enable_fp8_dit)
        if enable_compile_optimizations and enable_fp8_dit:
            logger("  compile note: ignoring --compile because FP8 is enabled")
        if compile_enabled and not module_available("triton"):
            logger("  compile note: ignoring --compile because triton is not installed/available")
            compile_enabled = False
        compile_flags: list[str] = []
        if compile_enabled:
            compile_flags.extend(["--compile"])
        if compile_enabled and enable_cuda_allow_tf32:
            compile_flags.extend(["--cuda_allow_tf32"])
        if compile_enabled and enable_cuda_cudnn_benchmark:
            compile_flags.extend(["--cuda_cudnn_benchmark"])
        if compile_enabled:
            compile_flags.extend(["--compile_cache_size_limit", "32"])
        fp8_flags = ["--fp8_base", "--fp8_scaled"] if enable_fp8_dit else []
        gc_offload_flags = ["--gradient_checkpointing_cpu_offload"] if enable_gradient_checkpointing_cpu_offload else []
        logging_flags: list[str] = []
        if enable_training_logging:
            log_backend = (training_log_backend or "tensorboard").strip().lower()
            if log_backend not in {"tensorboard", "wandb", "all"}:
                log_backend = "tensorboard"
            has_tensorboard = module_available("tensorboard")
            has_wandb = module_available("wandb")
            has_wandb_key = wandb_key_available()
            requirements_hint = (
                "Run Setup.bat (or install from Musubi-Trainer requirements.txt) "
                "to ensure logging dependencies are available in the shared app venv."
            )
            if log_backend == "tensorboard":
                if not has_tensorboard:
                    raise RuntimeError(
                        "Logging backend 'tensorboard' requires the tensorboard package. "
                        f"{requirements_hint}"
                    )
            elif log_backend == "wandb":
                if not has_wandb:
                    raise RuntimeError(
                        "Logging backend 'wandb' requires the wandb package. "
                        f"{requirements_hint}"
                    )
                if not has_wandb_key:
                    raise RuntimeError(
                        "Logging backend 'wandb' requires a configured W&B API key "
                        "(WANDB_API_KEY env var or wandb login)."
                    )
            else:
                if has_tensorboard and has_wandb:
                    if not has_wandb_key:
                        logger("  logging note: W&B API key not configured; using tensorboard only")
                        log_backend = "tensorboard"
                elif has_tensorboard:
                    logger("  logging note: wandb is not installed in Musubi venv; using tensorboard only")
                    log_backend = "tensorboard"
                elif has_wandb:
                    if not has_wandb_key:
                        raise RuntimeError(
                            "Logging backend 'all' could only use wandb, but W&B API key is not configured "
                            "(WANDB_API_KEY env var or wandb login)."
                        )
                    logger("  logging note: tensorboard is not installed in Musubi venv; using wandb only")
                    log_backend = "wandb"
                else:
                    raise RuntimeError(
                        "Logging backend 'all' requires tensorboard and wandb, but neither package is installed. "
                        f"{requirements_hint}"
                    )
            logging_dir, auto_tracker_name = next_dataset_log_run_dir(runtime_config.training_dir, model_name)
            tracker_name = training_log_tracker_name.strip() or auto_tracker_name
            logger(f"  logging_dir: {logging_dir}")
            logger(f"  log_tracker_name: {tracker_name}")
        optimizer_key = (optimizer_type or "adamw8bit").strip().lower()
        if optimizer_key == "prodigy" and not module_available("prodigyopt"):
            raise RuntimeError(
                "Optimizer 'prodigy' requires the prodigyopt package. "
                "Run Setup.bat (or install from Musubi-Trainer requirements.txt) "
                "to ensure prodigyopt is available in the shared app venv."
            )
        optimizer_arg = "prodigyopt.Prodigy" if optimizer_key == "prodigy" else (optimizer_type or "adamw8bit").strip()
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate
        optimizer_args_values: list[str] = []
        gradient_checkpointing_enabled = True
        gradient_checkpointing_cpu_offload_enabled = bool(enable_gradient_checkpointing_cpu_offload)
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
        train_config_path = output_dir.parent / "train_config.toml"
        config_lines: list[str] = [
            f"dit = {toml_quote(str(dit_path))}",
            f"vae = {toml_quote(str(vae_path))}",
            'vae_dtype = "bf16"',
            f"text_encoder = {toml_quote(str(text_encoder_path))}",
            f"model_version = {toml_quote(runtime_config.model_version)}",
            f"optimizer_type = {toml_quote(optimizer_arg)}",
            'timestep_sampling = "flux2_shift"',
            f"dataset_config = {toml_quote(str(dataset_config))}",
            f"output_dir = {toml_quote(str(output_dir))}",
            f"output_name = {toml_quote(output_name_for_run)}",
            'network_module = "networks.lora_flux_2"',
            f"network_dim = {network_dim}",
            f"network_alpha = {network_alpha}",
            f"learning_rate = {learning_rate_for_run}",
            f"max_train_steps = {train_steps_for_run}",
            'mixed_precision = "bf16"',
            "sdpa = true",
            f"gradient_checkpointing = {'true' if gradient_checkpointing_enabled else 'false'}",
            f"gradient_checkpointing_cpu_offload = {'true' if gradient_checkpointing_cpu_offload_enabled else 'false'}",
            "persistent_data_loader_workers = true",
            "max_data_loader_n_workers = 2",
            f"save_every_n_steps = {save_every_n_steps}",
            "save_state = true",
            "save_state_on_train_end = true",
            "seed = 42",
            f"fp8_base = {'true' if enable_fp8_dit else 'false'}",
            f"fp8_scaled = {'true' if enable_fp8_dit else 'false'}",
            f"compile = {'true' if compile_enabled else 'false'}",
            f"cuda_allow_tf32 = {'true' if (compile_enabled and enable_cuda_allow_tf32) else 'false'}",
            f"cuda_cudnn_benchmark = {'true' if (compile_enabled and enable_cuda_cudnn_benchmark) else 'false'}",
            f"compile_cache_size_limit = {32 if compile_enabled else 0}",
        ]
        if optimizer_args_values:
            config_lines.append(f"optimizer_args = {toml_string_list(optimizer_args_values)}")
        if enable_training_logging:
            config_lines.extend(
                [
                    f"log_with = {toml_quote(log_backend)}",
                    f"logging_dir = {toml_quote(str(logging_dir))}",
                ]
            )
            if tracker_name:
                config_lines.append(f"log_tracker_name = {toml_quote(tracker_name)}")
        if warmstart_checkpoint is not None:
            config_lines.append(f"network_weights = {toml_quote(str(warmstart_checkpoint))}")
        if resume_state_dir is not None:
            config_lines.append(f"resume = {toml_quote(str(resume_state_dir))}")

        train_config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

        train_args = [
            str(musubi_python),
            "flux_2_train_network.py",
            "--config_file",
            str(train_config_path),
        ]
        logger(f"  train_config: {train_config_path}")
        logger(f"  train_command: {format_command_for_log(train_args)}")
        try:
            run_command(
                train_args,
                cwd=runtime_config.musubi_dir,
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
                "Training launch failed due to invalid or incompatible FLUX.2 model files.\n"
                "Open Settings and verify Settings > FLUX.2 > Model, VAE, and Text Encoder paths.\n"
                f"Details: {exc}"
            ) from exc

        logger("")


def train_models(
    runtime_config: RuntimeConfig,
    model_names: list[str],
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    optimizer_type: str,
    optimizer_args: str,
    learning_rate: str,
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
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    cancel_requested: Callable[[], bool] | None = None,
) -> int:
    if not model_names:
        logger("No valid model names entered. Exiting.")
        return 1

    if not (do_prep_dataset or do_cache_latents or do_cache_text or do_train):
        logger("No steps selected. Select at least one step.")
        return 1

    logger(f"Queued models: {', '.join(model_names)}")
    failed_models: list[str] = []

    for index, model_name in enumerate(model_names, start=1):
        logger("")

        if cancel_requested is not None and cancel_requested():
            logger("Training cancelled by user. Stopping remaining models.")
            return 1

        effective_do_prep_dataset = do_prep_dataset
        effective_do_cache_latents = do_cache_latents
        effective_do_cache_text = do_cache_text
        effective_do_train = do_train
        resume_checkpoint, resume_step = latest_checkpoint_for_dataset(runtime_config.training_dir, model_name)
        finished_checkpoint = finished_checkpoint_for_dataset(runtime_config.training_dir, model_name)
        resume_state_dir, resume_state_step = latest_resume_state_for_dataset(
            runtime_config.training_dir,
            model_name,
            resume_step,
        )
        progress_step = max(resume_step, resume_state_step)
        effective_resume_state: Path | None = None
        resume_step_offset = 0
        effective_warmstart_checkpoint: Path | None = None
        train_steps_override: int | None = None

        if progress_step >= train_steps:
            logger(f"  checkpoint already complete at step {progress_step}: skipping")
            continue

        if resume_state_dir is not None and resume_state_step >= resume_step:
            effective_resume_state = resume_state_dir
            if finished_checkpoint is not None:
                effective_warmstart_checkpoint = finished_checkpoint
            known_progress_step = max(resume_step, resume_state_step)
            resume_step_offset = known_progress_step
            if known_progress_step > 0:
                train_steps_override = max(1, train_steps - known_progress_step)
            if resume_state_step > 0:
                logger(
                    f"  resuming optimizer state from {resume_state_dir.name} (step {resume_state_step}), "
                    f"remaining steps {train_steps_override if train_steps_override is not None else train_steps}"
                )
            else:
                logger(f"  resuming optimizer state from {resume_state_dir.name}")
            if finished_checkpoint is not None:
                logger(f"  using finished checkpoint weights: {finished_checkpoint.name}")
        elif resume_checkpoint is not None and resume_step > 0:
            effective_warmstart_checkpoint = resume_checkpoint
            train_steps_override = max(1, train_steps - resume_step)
            logger(
                f"  warm-starting from {resume_checkpoint.name} (step {resume_step}) via --network_weights, "
                f"remaining steps {train_steps_override}"
            )

        try:
            run_steps_for_model(
                runtime_config,
                model_name,
                default_caption_keyword=default_caption_keyword,
                resolution=resolution,
                network_dim=network_dim,
                network_alpha=network_alpha,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args,
                learning_rate=learning_rate,
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
                do_prep_dataset=effective_do_prep_dataset,
                do_cache_latents=effective_do_cache_latents,
                do_cache_text=effective_do_cache_text,
                do_train=effective_do_train,
                resume_state_dir=effective_resume_state,
                resume_step_offset=resume_step_offset,
                warmstart_checkpoint=effective_warmstart_checkpoint,
                train_steps_override=train_steps_override,
                output_name_override=None,
                output_dir_override=None,
                logger=logger,
                cancel_requested=cancel_requested,
            )
            if effective_resume_state is not None and resume_step_offset > 0:
                remap_resume_artifacts_to_continued_steps(
                    runtime_config.training_dir,
                    model_name,
                    resume_step_offset,
                    logger,
                )
            if auto_cleanup_states:
                cleanup_step_states_for_completed_run(runtime_config.training_dir, model_name, logger)
            logger(f"[{index}/{len(model_names)}] Completed: {model_name}")
        except TrainingCancelledError:
            if effective_resume_state is not None and resume_step_offset > 0:
                remap_resume_artifacts_to_continued_steps(
                    runtime_config.training_dir,
                    model_name,
                    resume_step_offset,
                    logger,
                )
            if auto_cleanup_states:
                cleanup_step_states_for_cancel(runtime_config.training_dir, model_name, logger)
            logger("Training cancelled by user. Stopping remaining models.")
            return 1
        except Exception as exc:
            logger(f"Training failed for '{model_name}': {exc}")
            failed_models.append(model_name)
            logger(f"[{index}/{len(model_names)}] Failed: {model_name} (continuing with next model)")
            continue

    logger("")
    if failed_models:
        logger("All model runs completed with failures.")
        logger(f"Failed models ({len(failed_models)}): {', '.join(failed_models)}")
        return 1

    logger("All model runs completed.")
    return 0


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
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    cancel_requested: Callable[[], bool] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> int:
    if not dataset_name.strip():
        message = "No dataset selected for job."
        logger(message)
        if on_error is not None:
            on_error(message)
        return 1

    if not output_name.strip():
        message = "Output name is required for job."
        logger(message)
        if on_error is not None:
            on_error(message)
        return 1

    if not (do_prep_dataset or do_cache_latents or do_cache_text or do_train):
        message = "No steps selected. Select at least one step."
        logger(message)
        if on_error is not None:
            on_error(message)
        return 1

    output_name_resolved = output_name.strip()
    output_dir_resolved = output_dir.resolve()
    resume_checkpoint, resume_step = latest_checkpoint_for_output(output_dir_resolved, output_name_resolved)
    finished_checkpoint = finished_checkpoint_for_output(output_dir_resolved, output_name_resolved)
    resume_state_dir, resume_state_step = latest_resume_state_for_output(
        output_dir_resolved,
        output_name_resolved,
        resume_step,
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
        known_progress_step = max(resume_step, resume_state_step)
        resume_step_offset = known_progress_step
        if known_progress_step > 0:
            train_steps_override = max(1, train_steps - known_progress_step)
        if resume_state_step > 0:
            logger(
                f"  resuming optimizer state from {resume_state_dir.name} (step {resume_state_step}), "
                f"remaining steps {train_steps_override if train_steps_override is not None else train_steps}"
            )
        else:
            logger(f"  resuming optimizer state from {resume_state_dir.name}")
        if finished_checkpoint is not None:
            logger(f"  using finished checkpoint weights: {finished_checkpoint.name}")
    elif resume_checkpoint is not None and resume_step > 0:
        effective_warmstart_checkpoint = resume_checkpoint
        train_steps_override = max(1, train_steps - resume_step)
        logger(
            f"  warm-starting from {resume_checkpoint.name} (step {resume_step}) via --network_weights, "
            f"remaining steps {train_steps_override}"
        )

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
        if on_error is not None:
            on_error(message)
        return JOB_EXIT_FAILED
