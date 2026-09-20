"""MiniMax H3 training module for Musubi-Trainer.

The launcher currently maps MiniMax H3 jobs to the documented one-frame image
LoRA recipe:
  - latent cache: --task t2va --one_frame
  - text cache:   --task t2va --one_frame
  - training:     --task t2va --one_frame --video_only

This matches the launcher's existing image-dataset job creation flow while
still accepting BF16, pruned, ConvRot INT8, and NVFP4 text-encoder artifacts.
"""

from __future__ import annotations

import json
import os
import shlex
import tomllib
from pathlib import Path
from typing import Callable

from .launcher_shared import VALID_IMAGE_EXTENSIONS, VALID_VIDEO_EXTENSIONS
from .runtime_config import RuntimeConfig
from .train_utils import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_NETWORK_ALPHA,
    DEFAULT_NETWORK_DIM,
    DEFAULT_SAVE_EVERY_N_STEPS,
    DEFAULT_TRAIN_STEPS,
    JOB_EXIT_CANCELLED,
    JOB_EXIT_FAILED,
    JOB_EXIT_SUCCESS,
    TrainingCancelledError,
    build_config_file_command,
    cleanup_step_states_for_cancel_output,
    cleanup_step_states_for_completed_output,
    clear_dataset_cache_directories,
    finished_checkpoint_for_output,
    format_command_for_log,
    latest_checkpoint_for_output,
    latest_resume_state_for_output,
    next_dataset_log_run_dir,
    read_recorded_completed_steps,
    remap_resume_artifacts_for_output,
    require_model_file,
    run_command,
    toml_quote,
    toml_string_list,
    write_recorded_completed_steps,
)


def _output_name_default(dataset_name: str) -> str:
    return f"{dataset_name}_MiniMax"


def _recommended_minimax_dataloader_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return max(2, min(8, cpu_count // 2))


def _network_settings(network_type: str, lora_module: str) -> tuple[str, bool]:
    selected = (network_type or "lora").strip().lower()
    use_lokr = selected == "lokr"
    return ("networks.lokr" if use_lokr else lora_module, use_lokr)


def _load_minimax_dataset_entries(dataset_config: Path) -> list[dict[str, object]]:
    try:
        config_data = tomllib.loads(dataset_config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read MiniMax dataset config at {dataset_config}: {exc}") from exc

    datasets = config_data.get("datasets")
    if not isinstance(datasets, list):
        raise RuntimeError(f"MiniMax dataset config at {dataset_config} does not define any [[datasets]] entries.")

    return [entry for entry in datasets if isinstance(entry, dict)]


def _inspect_minimax_dataset_config(dataset_config: Path) -> tuple[bool, bool]:
    datasets = _load_minimax_dataset_entries(dataset_config)

    has_video = any(str(entry.get("video_directory", "")).strip() for entry in datasets)
    has_image = any(str(entry.get("image_directory", "")).strip() for entry in datasets)
    return has_video, has_image


def _resolve_dataset_source_dir(dataset_config: Path, raw_path: str) -> Path:
    source_dir = Path(raw_path).expanduser()
    if source_dir.is_absolute():
        return source_dir
    return (dataset_config.parent / source_dir).resolve()


def _minimax_cache_manifest_path(dataset_config: Path) -> Path:
    return dataset_config.parent / ".minimax_cache_manifest.json"


def _build_minimax_cache_manifest(dataset_config: Path) -> dict[str, object]:
    file_records: list[dict[str, int | str]] = []
    datasets = _load_minimax_dataset_entries(dataset_config)
    for entry in datasets:
        for directory_key, valid_extensions in (
            ("image_directory", VALID_IMAGE_EXTENSIONS),
            ("video_directory", VALID_VIDEO_EXTENSIONS),
        ):
            raw_dir = str(entry.get(directory_key, "") or "").strip()
            if not raw_dir:
                continue
            source_dir = _resolve_dataset_source_dir(dataset_config, raw_dir)
            if not source_dir.exists() or not source_dir.is_dir():
                continue
            for source_path in sorted(source_dir.iterdir()):
                if not source_path.is_file() or source_path.suffix.lower() not in valid_extensions:
                    continue
                try:
                    source_stat = source_path.stat()
                except OSError:
                    continue
                file_records.append(
                    {
                        "path": str(source_path.resolve()),
                        "size": int(source_stat.st_size),
                        "mtime_ns": int(source_stat.st_mtime_ns),
                    }
                )
                caption_path = source_path.with_suffix(".txt")
                if caption_path.exists() and caption_path.is_file():
                    try:
                        caption_stat = caption_path.stat()
                    except OSError:
                        continue
                    file_records.append(
                        {
                            "path": str(caption_path.resolve()),
                            "size": int(caption_stat.st_size),
                            "mtime_ns": int(caption_stat.st_mtime_ns),
                        }
                    )

    dataset_config_stat = dataset_config.stat()
    return {
        "dataset_config": str(dataset_config.resolve()),
        "dataset_config_mtime_ns": int(dataset_config_stat.st_mtime_ns),
        "files": file_records,
    }


def _load_minimax_cache_manifest(dataset_config: Path) -> dict[str, object] | None:
    manifest_path = _minimax_cache_manifest_path(dataset_config)
    if not manifest_path.exists() or not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_minimax_cache_manifest(dataset_config: Path) -> None:
    manifest_path = _minimax_cache_manifest_path(dataset_config)
    manifest = _build_minimax_cache_manifest(dataset_config)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimax_cache_reset_reason(dataset_config: Path) -> str | None:
    previous_manifest = _load_minimax_cache_manifest(dataset_config)
    if previous_manifest is None:
        return None
    current_manifest = _build_minimax_cache_manifest(dataset_config)
    if current_manifest == previous_manifest:
        return None
    previous_files = previous_manifest.get("files")
    current_files = current_manifest.get("files")
    previous_count = len(previous_files) if isinstance(previous_files, list) else 0
    current_count = len(current_files) if isinstance(current_files, list) else 0
    if current_count != previous_count:
        return f"dataset file count changed ({previous_count} -> {current_count})"
    return "dataset inputs changed"


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "cuda out of memory" in message or "outofmemoryerror" in message


def run_steps_for_model(
    runtime_config: RuntimeConfig,
    model_name: str,
    *,
    network_dim: int,
    network_alpha: int,
    network_type: str,
    lokr_factor: int,
    optimizer_type: str,
    optimizer_args: str,
    learning_rate: str,
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
    blocks_to_swap: int = 0,
    video_vae_path: Path | None = None,
    audio_vae_path: Path | None = None,
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

    dataset_config = training_dir / model_name / "dataset.toml"
    if not dataset_config.is_file():
        raise RuntimeError(
            f"dataset.toml not found for '{model_name}' at {dataset_config}. "
            "Create the job first or provide a MiniMax-compatible dataset config."
        )
    has_video_datasets, has_image_datasets = _inspect_minimax_dataset_config(dataset_config)
    if not has_video_datasets and not has_image_datasets:
        raise RuntimeError(
            f"MiniMax dataset config at {dataset_config} must contain at least one image_directory or video_directory entry."
        )
    use_one_frame = has_image_datasets
    use_video_only = has_image_datasets and not has_video_datasets

    if do_cache_latents or do_cache_text:
        cache_reset_reason = _minimax_cache_reset_reason(dataset_config)
        if cache_reset_reason is not None:
            cleared_cache_dirs = clear_dataset_cache_directories(dataset_config, logger)
            if cleared_cache_dirs > 0:
                plural = "y" if cleared_cache_dirs == 1 else "ies"
                logger(
                    f"  cache reset: cleared {cleared_cache_dirs} cache director{plural} "
                    f"because {cache_reset_reason}"
                )
            else:
                logger(f"  cache reset: dataset changed ({cache_reset_reason})")
        else:
            logger("  cache reuse: MiniMax dataset unchanged; keeping existing cache directories")

    dit_path = require_model_file(runtime_config.dit, "MiniMax-H3 DiT")
    text_encoder_path = require_model_file(runtime_config.text_encoder, "MiniMax-H3 Text Encoder")
    video_vae_resolved = require_model_file(video_vae_path, "MiniMax-H3 Video VAE")
    audio_vae_resolved = require_model_file(audio_vae_path, "MiniMax-H3 Audio VAE")

    if do_cache_latents:
        logger(f"[1/3] Caching latents: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        cache_latents_args = [
            str(musubi_python),
            "minimax_h3_cache_latents.py",
            "--dataset_config", str(dataset_config),
            "--task", "t2va",
            "--video_vae", str(video_vae_resolved),
            "--audio_vae", str(audio_vae_resolved),
            "--cache_seed", "42",
            "--skip_existing",
        ]
        if use_one_frame:
            cache_latents_args.insert(5, "--one_frame")
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
                f"Latent caching failed for MiniMax H3 ({model_name}).\n"
                "Verify MiniMax H3 Video VAE and Audio VAE paths in Settings.\n"
                f"Details: {exc}"
            ) from exc

    if do_cache_text:
        logger(f"[2/3] Caching text encoder outputs: {model_name}")
        if cancel_requested and cancel_requested():
            raise TrainingCancelledError
        cache_text_args = [
            str(musubi_python),
            "minimax_h3_cache_text_encoder_outputs.py",
            "--dataset_config", str(dataset_config),
            "--task", "t2va",
            "--text_encoder", str(text_encoder_path),
            "--text_cache_dtype", "bf16",
            "--skip_existing",
        ]
        if use_one_frame:
            cache_text_args.insert(5, "--one_frame")
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
                f"Text caching failed for MiniMax H3 ({model_name}).\n"
                "Verify MiniMax H3 Text Encoder path in Settings.\n"
                f"Details: {exc}"
            ) from exc

    if do_train:
        logger(f"[3/3] Training: {model_name}  output_name={output_name}")
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
        optimizer_arg = "prodigyopt.Prodigy" if optimizer_key == "prodigy" else (optimizer_type or "adamw8bit").strip()
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate
        resolved_workers = max_data_loader_n_workers
        if resolved_workers is None:
            resolved_workers = _recommended_minimax_dataloader_workers()
        else:
            resolved_workers = max(1, int(resolved_workers))

        resolved_blocks_to_swap = max(0, int(blocks_to_swap))
        if resolved_blocks_to_swap <= 0:
            resolved_blocks_to_swap = 32
            logger(
                "  memory note: MiniMax H3 now defaults to blocks_to_swap=32 as a balanced 32 GB-friendly path"
            )
        elif resolved_blocks_to_swap > 48:
            logger("  memory note: clamping MiniMax H3 blocks_to_swap to 48 (trainer maximum is 48)")
            resolved_blocks_to_swap = 48

        if enable_fp8_dit:
            logger("  fp8 note: MiniMax H3 does not support FP8 DiT loading; ignoring the FP8 toggle")

        compile_enabled = bool(enable_compile_optimizations)
        if compile_enabled:
            logger(
                "  compile note: disabling torch compile for MiniMax H3 because Inductor can spike VRAM during graph compilation"
            )
            compile_enabled = False

        model_lines = [
            f"task = {toml_quote('t2va')}",
            f"one_frame = {'true' if use_one_frame else 'false'}",
            f"video_only = {'true' if use_video_only else 'false'}",
            f"dit = {toml_quote(str(dit_path))}",
            f"video_vae = {toml_quote(str(video_vae_resolved))}",
            f"audio_vae = {toml_quote(str(audio_vae_resolved))}",
            f"text_encoder = {toml_quote(str(text_encoder_path))}",
        ]

        data_output_lines = [
            f"dataset_config = {toml_quote(str(dataset_config))}",
            f"output_dir = {toml_quote(str(output_dir))}",
            f"output_name = {toml_quote(output_name)}",
        ]
        selected_network_module, is_lokr = _network_settings(network_type, "networks.lora_minimax_h3")
        network_lines = [
            f"network_module = {toml_quote(selected_network_module)}",
            f"network_dim = {network_dim}",
            f"network_alpha = {network_alpha}",
        ]
        if is_lokr and lokr_factor != -1:
            network_lines.append(f"network_args = {toml_string_list([f'factor={lokr_factor}'])}")

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

        optimization_lines = [
            f"optimizer_type = {toml_quote(optimizer_arg)}",
            f"learning_rate = {learning_rate_for_run}",
            f"max_train_steps = {train_steps_for_run}",
            f"timestep_sampling = {toml_quote('uniform')}",
            f"weighting_scheme = {toml_quote('none')}",
            "discrete_flow_shift = 1.0",
        ]
        if optimizer_args_values:
            optimization_lines.append(f"optimizer_args = {toml_string_list(optimizer_args_values)}")

        runtime_lines = [
            'mixed_precision = "bf16"',
            "sdpa = true",
            "gradient_checkpointing = true",
            f"gradient_checkpointing_cpu_offload = {'true' if enable_gradient_checkpointing_cpu_offload else 'false'}",
            "persistent_data_loader_workers = true",
            f"max_data_loader_n_workers = {resolved_workers}",
            f"blocks_to_swap = {resolved_blocks_to_swap}",
            "block_swap_h2d_only = true",
            f"compile = {'true' if compile_enabled else 'false'}",
            f"cuda_allow_tf32 = {'true' if enable_cuda_allow_tf32 else 'false'}",
            f"cuda_cudnn_benchmark = {'true' if enable_cuda_cudnn_benchmark else 'false'}",
        ]
        checkpoint_lines = [
            f"save_every_n_steps = {save_every_n_steps}",
            "save_state = true",
            "save_state_on_train_end = true",
            "seed = 42",
        ]

        restore_lines: list[str] = []
        if warmstart_checkpoint is not None:
            restore_lines.append(f"network_weights = {toml_quote(str(warmstart_checkpoint))}")
        if resume_state_dir is not None:
            restore_lines.append(f"resume = {toml_quote(str(resume_state_dir))}")

        logging_lines: list[str] = []
        if enable_training_logging and logging_dir is not None:
            logging_lines.extend(
                [
                    f"log_with = {toml_quote(log_backend)}",
                    f"logging_dir = {toml_quote(str(logging_dir))}",
                ]
            )
            if tracker_name:
                logging_lines.append(f"log_tracker_name = {toml_quote(tracker_name)}")

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
        train_config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
        logger(f"  training_args: {train_config_path}")

        if generate_training_args_only:
            logger("  training args generated (no training launched)")
        else:
            launch_args = build_config_file_command(musubi_python, "minimax_h3_train_network.py", train_config_path)
            logger(f"  command: {format_command_for_log(launch_args)}")
            try:
                run_command(
                    launch_args,
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
                if _is_cuda_oom_error(exc):
                    raise RuntimeError(
                        f"MiniMax H3 training ran out of VRAM ({model_name}).\n"
                        "This is not a model-path problem. MiniMax H3 is using the configured crop/window and still exceeds GPU memory.\n"
                        "Try lowering resolution, lowering network dim/alpha, or switching to a lighter model family for audio-heavy LoRA work.\n"
                        f"Details: {exc}"
                    ) from exc
                raise RuntimeError(
                    f"Training launch failed for MiniMax H3 ({model_name}).\n"
                    "Verify Settings > MiniMax > DiT, Video VAE, Audio VAE, and Text Encoder paths.\n"
                    f"Details: {exc}"
                ) from exc

    if do_cache_latents or do_cache_text:
        _write_minimax_cache_manifest(dataset_config)

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
    max_data_loader_n_workers: int | None = None,
    generate_training_args_only: bool = False,
    save_every_n_steps: int = DEFAULT_SAVE_EVERY_N_STEPS,
    cancel_requested: Callable[[], bool] | None = None,
    on_error: Callable[[str], None] | None = None,
    blocks_to_swap: int = 0,
    video_vae_path: Path | None = None,
    audio_vae_path: Path | None = None,
) -> int:
    del default_caption_keyword, resolution, do_prep_dataset

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

    if not (do_cache_latents or do_cache_text or do_train or generate_training_args_only):
        message = "No steps selected."
        logger(message)
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED

    if video_vae_path is None or audio_vae_path is None:
        message = "MiniMax H3 requires both Video VAE and Audio VAE paths in Settings."
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
    elif progress_step > 0 and not generate_training_args_only:
        message = (
            f"Progress metadata reports step {progress_step}, but no resume state/checkpoint was found "
            f"for '{output_name_resolved}'. Refusing to start from step 0. "
            "Reset the job for a fresh run or restore resume artifacts."
        )
        logger(message)
        if on_error:
            on_error(message)
        return JOB_EXIT_FAILED

    try:
        run_steps_for_model(
            runtime_config,
            dataset_name,
            network_dim=network_dim,
            network_alpha=network_alpha,
            network_type=network_type,
            lokr_factor=lokr_factor,
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
            blocks_to_swap=blocks_to_swap,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
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