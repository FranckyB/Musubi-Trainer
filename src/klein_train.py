import subprocess
import re
import time
import tempfile
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

from .klein_runtime_config import KleinRuntimeConfig

DEFAULT_RESOLUTION = 1024
DEFAULT_NETWORK_DIM = 32
DEFAULT_NETWORK_ALPHA = 32
DEFAULT_LEARNING_RATE = "1e-4"
DEFAULT_TRAIN_STEPS = 3000

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"


class TrainingCancelledError(RuntimeError):
    pass


def latest_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    if not output_dir.exists():
        return None, 0

    pattern = re.compile(rf"^{re.escape(dataset_name)}_Klein-step(\d+)\.safetensors$", re.IGNORECASE)
    latest_step = 0
    latest_path: Path | None = None

    for checkpoint_path in output_dir.glob("*.safetensors"):
        match = pattern.match(checkpoint_path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = checkpoint_path

    return latest_path, latest_step


def latest_resume_state_for_dataset(training_dir: Path, dataset_name: str, checkpoint_step: int) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    if not output_dir.exists():
        return None, 0

    output_name = f"{dataset_name}_Klein"

    # Preferred: step-aligned state dir for the latest known checkpoint step.
    if checkpoint_step > 0:
        step_state_dir = output_dir / f"{output_name}-step{checkpoint_step:08d}-state"
        if step_state_dir.is_dir() and (step_state_dir / "scheduler.bin").exists():
            return step_state_dir, checkpoint_step

    # Fallback: final train-end state dir (no step in folder name).
    last_state_dir = output_dir / f"{output_name}-state"
    if last_state_dir.is_dir() and (last_state_dir / "scheduler.bin").exists():
        return last_state_dir, checkpoint_step

    return None, 0


def dataset_image_files(training_dir: Path, dataset_name: str) -> list[Path]:
    images_dir = training_dir / dataset_name / "images"
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])


def is_step1_ready(training_dir: Path, dataset_name: str) -> bool:
    dataset_dir = training_dir / dataset_name
    dataset_toml_exists = (dataset_dir / "dataset.toml").exists()
    image_files = dataset_image_files(training_dir, dataset_name)

    if not dataset_toml_exists or not image_files:
        return False

    return all(image_path.with_suffix(".txt").exists() for image_path in image_files)


def is_step2_ready(training_dir: Path, dataset_name: str) -> bool:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        pattern = f"{image_path.stem}_*_{LATENT_SUFFIX}.safetensors"
        if not any(cache_dir.glob(pattern)):
            return False

    return True


def is_step3_ready(training_dir: Path, dataset_name: str) -> bool:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"

    if not image_files or not cache_dir.exists():
        return False

    for image_path in image_files:
        expected = cache_dir / f"{image_path.stem}_{LATENT_SUFFIX}_te.safetensors"
        if not expected.exists():
            return False

    return True


def dataset_status(training_dir: Path, dataset_name: str) -> dict[str, bool]:
    step1 = is_step1_ready(training_dir, dataset_name)
    step2 = is_step2_ready(training_dir, dataset_name)
    step3 = is_step3_ready(training_dir, dataset_name)
    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "ready_to_train": step1 and step2 and step3,
    }


def prep_dataset_minimal(
    training_dir: Path,
    dataset_name: str,
    default_caption_keyword: str,
    resolution: int,
) -> dict[str, int | bool]:
    dataset_dir = training_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_toml = dataset_dir / "dataset.toml"
    image_dir_abs = (dataset_dir / "images").resolve().as_posix()
    cache_dir_abs = (dataset_dir / "cache").resolve().as_posix()

    had_dataset_toml = dataset_toml.exists()
    if not had_dataset_toml:
        dataset_toml.write_text(
            "\n".join(
                [
                    "[general]",
                    f"resolution = [{resolution}, {resolution}]",
                    'caption_extension = ".txt"',
                    "batch_size = 1",
                    "enable_bucket = true",
                    "bucket_no_upscale = false",
                    "",
                    "[[datasets]]",
                    f'image_directory = "{image_dir_abs}"',
                    f'cache_directory = "{cache_dir_abs}"',
                    "num_repeats = 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    created = 0
    caption_text = default_caption_keyword.strip()
    for image_path in dataset_image_files(training_dir, dataset_name):
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            continue
        caption_path.write_text(caption_text, encoding="utf-8")
        created += 1

    return {"had_dataset_toml": had_dataset_toml, "created": created}


def count_latent_cache_ready(training_dir: Path, dataset_name: str) -> tuple[int, int]:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"
    if not image_files or not cache_dir.exists():
        return 0, len(image_files)

    ready = 0
    for image_path in image_files:
        pattern = f"{image_path.stem}_*_{LATENT_SUFFIX}.safetensors"
        if any(cache_dir.glob(pattern)):
            ready += 1
    return ready, len(image_files)


def count_text_cache_ready(training_dir: Path, dataset_name: str) -> tuple[int, int]:
    image_files = dataset_image_files(training_dir, dataset_name)
    cache_dir = training_dir / dataset_name / "cache"
    if not image_files or not cache_dir.exists():
        return 0, len(image_files)

    ready = 0
    for image_path in image_files:
        expected = cache_dir / f"{image_path.stem}_{LATENT_SUFFIX}_te.safetensors"
        if expected.exists():
            ready += 1
    return ready, len(image_files)


def backup_existing_step_checkpoints(output_dir: Path, output_name: str) -> tuple[Path | None, int]:
    if not output_dir.exists() or not output_dir.is_dir():
        return None, 0

    pattern = re.compile(rf"^{re.escape(output_name)}-step\d{{8}}\.safetensors$", re.IGNORECASE)
    step_files = sorted([p for p in output_dir.glob("*.safetensors") if pattern.match(p.name)])
    if not step_files:
        return None, 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = output_dir / "backup" / f"resume-pre-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source in step_files:
        shutil.copy2(source, backup_dir / source.name)
        copied += 1

    return backup_dir, copied


def run_command(
    args: Iterable[str],
    cwd: Path,
    cancel_requested: Callable[[], bool] | None = None,
    logger: Callable[[str], None] | None = None,
    stream_to_logger: bool = False,
) -> None:
    if not stream_to_logger:
        process = subprocess.Popen(list(args), cwd=str(cwd), env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
        while True:
            if cancel_requested is not None and cancel_requested():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise TrainingCancelledError("Cancelled by user.")

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(args)}")
                return
            time.sleep(0.2)

    process: subprocess.Popen | None = None
    log_path: Path | None = None
    read_offset = 0
    partial_line = ""
    recent_lines: deque[str] = deque(maxlen=40)

    def flush_new_output() -> None:
        nonlocal read_offset, partial_line
        if logger is None or log_path is None or not log_path.exists():
            return
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                reader.seek(read_offset)
                chunk = reader.read()
                read_offset = reader.tell()
        except OSError:
            return

        if not chunk:
            return

        chunk = partial_line + chunk
        lines = chunk.splitlines()
        if chunk and not chunk.endswith(("\n", "\r")):
            partial_line = lines.pop() if lines else chunk
        else:
            partial_line = ""

        for line in lines:
            cleaned = line.rstrip("\r\n")
            recent_lines.append(cleaned)
            logger(cleaned)

    def flush_partial_line() -> None:
        nonlocal partial_line
        if logger is None:
            return
        if not partial_line:
            return
        cleaned = partial_line.rstrip("\r\n")
        recent_lines.append(cleaned)
        logger(cleaned)
        partial_line = ""

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", errors="replace", delete=False, suffix=".log") as log_file:
            log_path = Path(log_file.name)
            process = subprocess.Popen(list(args), cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT, env=env)
            while True:
                flush_new_output()
                if cancel_requested is not None and cancel_requested():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise TrainingCancelledError("Cancelled by user.")

                return_code = process.poll()
                if return_code is not None:
                    flush_new_output()
                    flush_partial_line()
                    if return_code != 0:
                        output_tail = "\n".join(recent_lines)
                        if not output_tail and log_path.exists():
                            try:
                                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                if lines:
                                    output_tail = "\n".join(lines[-40:])
                            except OSError:
                                output_tail = ""
                        if output_tail:
                            raise RuntimeError(
                                f"Command failed with exit code {return_code}: {' '.join(args)}\n"
                                f"--- command output (tail) ---\n{output_tail}"
                            )
                        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(args)}")
                    return
                time.sleep(0.2)
    finally:
        if log_path is not None and log_path.exists():
            try:
                log_path.unlink()
            except OSError:
                pass


def format_command_for_log(args: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(arg) for arg in args])


def run_steps_for_model(
    runtime_config: KleinRuntimeConfig,
    model_name: str,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    learning_rate: str,
    train_steps: int,
    enable_compile_optimizations: bool,
    enable_cuda_allow_tf32: bool,
    enable_cuda_cudnn_benchmark: bool,
    enable_fp8_dit: bool,
    enable_gradient_checkpointing_cpu_offload: bool,
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
    resume_state_dir: Path | None,
    warmstart_checkpoint: Path | None,
    train_steps_override: int | None,
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
        if not path_value.is_file():
            raise RuntimeError(f"{label} file does not exist: {path_value}")
        return path_value

    def check_cancel() -> None:
        if cancel_requested is not None and cancel_requested():
            raise TrainingCancelledError("Cancelled by user.")

    musubi_python = require_musubi_python()

    dataset_config = runtime_config.training_dir / model_name / "dataset.toml"
    output_dir = runtime_config.training_dir / model_name / "output"
    dataset_config = dataset_config.resolve()
    output_dir = output_dir.resolve()
    output_name = f"{model_name}_Klein"

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
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
        vae_path = vae_path.resolve()
        before_ready, before_total = count_latent_cache_ready(runtime_config.training_dir, model_name)
        if before_total > 0 and before_ready == before_total:
            logger(f"  cache_latents: already ready ({before_ready}/{before_total}), skipped")
        else:
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
                    "--model_version",
                    runtime_config.model_version,
                ],
                cwd=runtime_config.musubi_dir,
                cancel_requested=cancel_requested,
                logger=logger,
                stream_to_logger=False,
            )
            after_ready, after_total = count_latent_cache_ready(runtime_config.training_dir, model_name)
            generated = max(0, after_ready - before_ready)
            logger(f"  cache_latents: done ({after_ready}/{after_total} ready, +{generated} generated)")
        logger("")

    if do_cache_text:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Cache Text Encoder:")
        text_encoder_path = require_model_file(runtime_config.text_encoder, "Klein Text Encoder")
        text_encoder_path = text_encoder_path.resolve()
        before_ready, before_total = count_text_cache_ready(runtime_config.training_dir, model_name)
        if before_total > 0 and before_ready == before_total:
            logger(f"  cache_text: already ready ({before_ready}/{before_total}), skipped")
        else:
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
                    "--model_version",
                    runtime_config.model_version,
                ],
                cwd=runtime_config.musubi_dir,
                cancel_requested=cancel_requested,
                logger=logger,
                stream_to_logger=False,
            )
            after_ready, after_total = count_text_cache_ready(runtime_config.training_dir, model_name)
            generated = max(0, after_ready - before_ready)
            logger(f"  cache_text: done ({after_ready}/{after_total} ready, +{generated} generated)")
        logger("")

    if do_train:
        check_cancel()
        current_step += 1
        logger(f"[{current_step}/{total_steps}]   Train:")
        dit_path = require_model_file(runtime_config.dit, "Klein Model")
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
        text_encoder_path = require_model_file(runtime_config.text_encoder, "Klein Text Encoder")
        dit_path = dit_path.resolve()
        vae_path = vae_path.resolve()
        text_encoder_path = text_encoder_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if resume_state_dir is not None:
            backup_dir, copied = backup_existing_step_checkpoints(output_dir, output_name)
            if copied > 0 and backup_dir is not None:
                logger(f"  resume safety: backed up {copied} step checkpoint(s) to {backup_dir}")

        train_steps_for_run = train_steps_override if train_steps_override is not None else train_steps
        compile_flags: list[str] = []
        if enable_compile_optimizations:
            compile_flags.extend(["--compile"])
        if enable_compile_optimizations and enable_cuda_allow_tf32:
            compile_flags.extend(["--cuda_allow_tf32"])
        if enable_compile_optimizations and enable_cuda_cudnn_benchmark:
            compile_flags.extend(["--cuda_cudnn_benchmark"])
        if enable_compile_optimizations:
            compile_flags.extend(["--compile_cache_size_limit", "32"])
        fp8_flags = ["--fp8_base", "--fp8_scaled"] if enable_fp8_dit else []
        gc_offload_flags = ["--gradient_checkpointing_cpu_offload"] if enable_gradient_checkpointing_cpu_offload else []
        train_args = [
            str(musubi_python),
            "flux_2_train_network.py",
            "--dit", str(dit_path),
            "--vae", str(vae_path),
            "--vae_dtype", "bf16",
            "--text_encoder", str(text_encoder_path),
            "--model_version", runtime_config.model_version,
            "--optimizer_type", "adamw8bit",
            "--timestep_sampling", "flux2_shift",
            "--dataset_config", str(dataset_config),
            "--output_dir", str(output_dir),
            "--output_name", output_name,
            "--network_module", "networks.lora_flux_2",
            "--network_dim", str(network_dim),
            "--network_alpha", str(network_alpha),
            "--learning_rate", learning_rate,
            "--max_train_steps", str(train_steps_for_run),
            "--mixed_precision", "bf16",
            "--sdpa",
            "--gradient_checkpointing",
            *gc_offload_flags,
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--save_every_n_steps", "250",
            "--save_state",
            "--save_state_on_train_end",
            "--seed", "42",
            *fp8_flags,
            *compile_flags,
            *(["--network_weights", str(warmstart_checkpoint)] if warmstart_checkpoint is not None else []),
            *(["--resume", str(resume_state_dir)] if resume_state_dir is not None else []),
        ]
        logger(f"  train_command: {format_command_for_log(train_args)}")
        run_command(
            train_args,
            cwd=runtime_config.musubi_dir,
            cancel_requested=cancel_requested,
            logger=logger,
            stream_to_logger=False,
        )
        logger("")


def train_models(
    runtime_config: KleinRuntimeConfig,
    model_names: list[str],
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    learning_rate: str,
    train_steps: int,
    enable_compile_optimizations: bool,
    enable_cuda_allow_tf32: bool,
    enable_cuda_cudnn_benchmark: bool,
    enable_fp8_dit: bool,
    enable_gradient_checkpointing_cpu_offload: bool,
    logger: Callable[[str], None],
    do_prep_dataset: bool,
    do_cache_latents: bool,
    do_cache_text: bool,
    do_train: bool,
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
        resume_state_dir, resume_state_step = latest_resume_state_for_dataset(
            runtime_config.training_dir,
            model_name,
            resume_step,
        )
        progress_step = max(resume_step, resume_state_step)
        effective_resume_state: Path | None = None
        effective_warmstart_checkpoint: Path | None = None
        train_steps_override: int | None = None

        if progress_step >= train_steps:
            logger(f"  checkpoint already complete at step {progress_step}: skipping")
            continue

        if resume_state_dir is not None and resume_state_step >= resume_step:
            effective_resume_state = resume_state_dir
            known_progress_step = max(resume_step, resume_state_step)
            if known_progress_step > 0:
                train_steps_override = max(1, train_steps - known_progress_step)
            if resume_state_step > 0:
                logger(
                    f"  resuming optimizer state from {resume_state_dir.name} (step {resume_state_step}), "
                    f"remaining steps {train_steps_override if train_steps_override is not None else train_steps}"
                )
            else:
                logger(f"  resuming optimizer state from {resume_state_dir.name}")
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
                learning_rate=learning_rate,
                train_steps=train_steps,
                enable_compile_optimizations=enable_compile_optimizations,
                enable_cuda_allow_tf32=enable_cuda_allow_tf32,
                enable_cuda_cudnn_benchmark=enable_cuda_cudnn_benchmark,
                enable_fp8_dit=enable_fp8_dit,
                enable_gradient_checkpointing_cpu_offload=enable_gradient_checkpointing_cpu_offload,
                do_prep_dataset=effective_do_prep_dataset,
                do_cache_latents=effective_do_cache_latents,
                do_cache_text=effective_do_cache_text,
                do_train=effective_do_train,
                resume_state_dir=effective_resume_state,
                warmstart_checkpoint=effective_warmstart_checkpoint,
                train_steps_override=train_steps_override,
                logger=logger,
                cancel_requested=cancel_requested,
            )
            logger(f"[{index}/{len(model_names)}] Completed: {model_name}")
        except TrainingCancelledError:
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
