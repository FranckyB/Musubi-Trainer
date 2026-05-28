import subprocess
import re
import time
import tempfile
import os
import shutil
import json
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

from .klein_runtime_config import KleinRuntimeConfig

DEFAULT_RESOLUTION = 1024
DEFAULT_NETWORK_DIM = 32
DEFAULT_NETWORK_ALPHA = 32
DEFAULT_LEARNING_RATE = "1e-4"
DEFAULT_TRAIN_STEPS = 3000
JOB_EXIT_SUCCESS = 0
JOB_EXIT_FAILED = 1
JOB_EXIT_CANCELLED = 2

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LATENT_SUFFIX = "f2k9b"
SHARDED_SAFETENSORS_NAME_RE = re.compile(r".+-\d{5}-of-\d{5}\.safetensors$", re.IGNORECASE)
JOB_PROGRESS_FILE_NAME = "progress.json"


def centralized_logs_root(training_dir: Path) -> Path:
    return training_dir


def next_dataset_log_run_dir(training_dir: Path, dataset_name: str) -> tuple[Path, str]:
    logs_root = centralized_logs_root(training_dir) / dataset_name / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    date_tag = datetime.now().strftime("%y%m%d")
    escaped_dataset = re.escape(dataset_name)
    run_pattern = re.compile(rf"^{escaped_dataset}_{date_tag}_(\d{{2}})$", re.IGNORECASE)

    max_index = 0
    for child in logs_root.iterdir():
        if not child.is_dir():
            continue
        match = run_pattern.match(child.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))

    while True:
        run_index = max_index + 1
        run_name = f"{dataset_name}_{date_tag}_{run_index:02d}"
        run_dir = logs_root / run_name
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, run_name
        max_index += 1


class TrainingCancelledError(RuntimeError):
    pass


def latest_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    output_name = f"{dataset_name}_Klein"
    return latest_checkpoint_for_output(output_dir, output_name)


def latest_checkpoint_for_output(output_dir: Path, output_name: str) -> tuple[Path | None, int]:
    if not output_dir.exists():
        return None, 0

    pattern = re.compile(rf"^{re.escape(output_name)}-step(\d+)\.safetensors$", re.IGNORECASE)
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


def progress_metadata_path_for_output(output_dir: Path) -> Path:
    return output_dir.parent / JOB_PROGRESS_FILE_NAME


def read_recorded_completed_steps(output_dir: Path, output_name: str) -> int:
    metadata_path = progress_metadata_path_for_output(output_dir)
    if not metadata_path.exists() or not metadata_path.is_file():
        return 0

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if not isinstance(payload, dict):
        return 0

    raw_value = payload.get("completed_step", 0)
    try:
        completed_step = int(raw_value)
    except (TypeError, ValueError):
        return 0

    return completed_step if completed_step > 0 else 0


def write_recorded_completed_steps(output_dir: Path, output_name: str, completed_step: int, target_steps: int) -> None:
    metadata_path = progress_metadata_path_for_output(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_name": output_name,
        "completed_step": int(max(0, completed_step)),
        "target_steps": int(max(0, target_steps)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finished_checkpoint_for_dataset(training_dir: Path, dataset_name: str) -> Path | None:
    output_dir = training_dir / dataset_name / "output"
    output_name = f"{dataset_name}_Klein"
    return finished_checkpoint_for_output(output_dir, output_name)


def finished_checkpoint_for_output(output_dir: Path, output_name: str) -> Path | None:
    if not output_dir.exists():
        return None

    finished_path = output_dir / f"{output_name}.safetensors"
    if finished_path.is_file():
        return finished_path

    return None


def latest_resume_state_for_dataset(training_dir: Path, dataset_name: str, checkpoint_step: int) -> tuple[Path | None, int]:
    output_dir = training_dir / dataset_name / "output"
    output_name = f"{dataset_name}_Klein"
    return latest_resume_state_for_output(output_dir, output_name, checkpoint_step)


def latest_resume_state_for_output(output_dir: Path, output_name: str, checkpoint_step: int) -> tuple[Path | None, int]:
    if not output_dir.exists():
        return None, 0

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


def dataset_image_directory_from_config(training_dir: Path, dataset_name: str) -> Path | None:
    dataset_toml = training_dir / dataset_name / "dataset.toml"
    if not dataset_toml.exists() or not dataset_toml.is_file():
        return None

    try:
        content = dataset_toml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    match = re.search(r'^\s*image_directory\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
    if match is None:
        return None

    raw_path = match.group(1).strip()
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (dataset_toml.parent / candidate).resolve()

    return candidate


def dataset_image_files(training_dir: Path, dataset_name: str) -> list[Path]:
    images_dir = training_dir / dataset_name / "images"
    if images_dir.exists() and images_dir.is_dir():
        return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])

    configured_images_dir = dataset_image_directory_from_config(training_dir, dataset_name)
    if configured_images_dir is None:
        return []
    if not configured_images_dir.exists() or not configured_images_dir.is_dir():
        return []

    return sorted([p for p in configured_images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])


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


def remap_resume_artifacts_to_continued_steps(
    training_dir: Path,
    dataset_name: str,
    resume_step_offset: int,
    logger: Callable[[str], None],
) -> None:
    output_dir = training_dir / dataset_name / "output"
    base_output_name = f"{dataset_name}_Klein"
    remap_resume_artifacts_for_output(output_dir, base_output_name, resume_step_offset, logger)


def remap_resume_artifacts_for_output(
    output_dir: Path,
    base_output_name: str,
    resume_step_offset: int,
    logger: Callable[[str], None],
) -> None:
    if resume_step_offset <= 0:
        return

    if not output_dir.exists() or not output_dir.is_dir():
        return

    resume_output_name = f"{base_output_name}-resume"

    ckpt_pattern = re.compile(rf"^{re.escape(resume_output_name)}-step(\d{{8}})\.safetensors$", re.IGNORECASE)
    state_pattern = re.compile(rf"^{re.escape(resume_output_name)}-step(\d{{8}})-state$", re.IGNORECASE)

    renamed_ckpts = 0
    renamed_states = 0

    resume_ckpts: list[tuple[Path, int]] = []
    for path in output_dir.glob("*.safetensors"):
        match = ckpt_pattern.match(path.name)
        if not match:
            continue
        resume_ckpts.append((path, int(match.group(1))))

    for source, source_step in sorted(resume_ckpts, key=lambda pair: pair[1]):
        target_step = source_step + resume_step_offset
        target = output_dir / f"{base_output_name}-step{target_step:08d}.safetensors"
        if target.exists():
            logger(f"  rename warning: target exists, keeping {source.name} (target: {target.name})")
            continue
        try:
            source.rename(target)
            renamed_ckpts += 1
        except OSError as exc:
            logger(f"  rename warning: could not rename {source.name} -> {target.name}: {exc}")

    resume_states: list[tuple[Path, int]] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        match = state_pattern.match(path.name)
        if not match:
            continue
        resume_states.append((path, int(match.group(1))))

    for source, source_step in sorted(resume_states, key=lambda pair: pair[1]):
        target_step = source_step + resume_step_offset
        target = output_dir / f"{base_output_name}-step{target_step:08d}-state"
        if target.exists():
            logger(f"  rename warning: target exists, keeping {source.name} (target: {target.name})")
            continue
        try:
            source.rename(target)
            renamed_states += 1
        except OSError as exc:
            logger(f"  rename warning: could not rename {source.name} -> {target.name}: {exc}")

    resume_last_checkpoint = output_dir / f"{resume_output_name}.safetensors"
    if resume_last_checkpoint.is_file():
        target_last_checkpoint = output_dir / f"{base_output_name}.safetensors"
        try:
            if target_last_checkpoint.exists():
                target_last_checkpoint.unlink()
            resume_last_checkpoint.rename(target_last_checkpoint)
            logger(f"  rename: {resume_last_checkpoint.name} -> {target_last_checkpoint.name}")
        except OSError as exc:
            logger(
                f"  rename warning: could not rename {resume_last_checkpoint.name} -> "
                f"{target_last_checkpoint.name}: {exc}"
            )

    resume_last_state = output_dir / f"{resume_output_name}-state"
    if resume_last_state.is_dir():
        target_last_state = output_dir / f"{base_output_name}-state"
        try:
            if target_last_state.exists():
                shutil.rmtree(target_last_state)
            resume_last_state.rename(target_last_state)
            logger(f"  rename: {resume_last_state.name} -> {target_last_state.name}")
        except OSError as exc:
            logger(
                f"  rename warning: could not rename {resume_last_state.name} -> {target_last_state.name}: {exc}"
            )

    if renamed_ckpts > 0 or renamed_states > 0:
        logger(
            "  rename: remapped resumed artifacts with continued step numbering "
            f"(+{resume_step_offset}, checkpoints={renamed_ckpts}, states={renamed_states})"
        )


def _step_state_dirs(output_dir: Path, output_name: str) -> list[tuple[Path, int]]:
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    pattern = re.compile(rf"^{re.escape(output_name)}-step(\d{{8}})-state$", re.IGNORECASE)
    result: list[tuple[Path, int]] = []
    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        result.append((item, int(match.group(1))))
    return sorted(result, key=lambda pair: pair[1])


def cleanup_step_states_for_cancel(training_dir: Path, dataset_name: str, logger: Callable[[str], None]) -> None:
    output_dir = training_dir / dataset_name / "output"
    output_name = f"{dataset_name}_Klein"
    cleanup_step_states_for_cancel_output(output_dir, output_name, logger)


def cleanup_step_states_for_cancel_output(output_dir: Path, output_name: str, logger: Callable[[str], None]) -> None:
    step_state_dirs = _step_state_dirs(output_dir, output_name)
    if not step_state_dirs:
        return

    _latest_ckpt, latest_step = latest_checkpoint_for_output(output_dir, output_name)
    keep_dir: Path | None = None

    if latest_step > 0:
        for state_dir, state_step in step_state_dirs:
            if state_step == latest_step:
                keep_dir = state_dir
                break

    if keep_dir is None:
        # Fallback to newest step-state directory if checkpoint and state names do not line up.
        keep_dir = step_state_dirs[-1][0]

    removed = 0
    for state_dir, _state_step in step_state_dirs:
        if state_dir == keep_dir:
            continue
        try:
            shutil.rmtree(state_dir)
            removed += 1
        except OSError as exc:
            logger(f"  cleanup warning: could not remove {state_dir.name}: {exc}")

    logger(f"  cleanup: kept latest resume state {keep_dir.name}, removed {removed} older step state folder(s)")


def cleanup_step_states_for_completed_run(training_dir: Path, dataset_name: str, logger: Callable[[str], None]) -> None:
    output_dir = training_dir / dataset_name / "output"
    output_name = f"{dataset_name}_Klein"
    cleanup_step_states_for_completed_output(output_dir, output_name, logger)


def cleanup_step_states_for_completed_output(output_dir: Path, output_name: str, logger: Callable[[str], None]) -> None:
    last_state_dir = output_dir / f"{output_name}-state"
    if not last_state_dir.is_dir():
        logger(f"  cleanup skipped: final state folder not found ({last_state_dir.name})")
        return

    step_state_dirs = _step_state_dirs(output_dir, output_name)
    if not step_state_dirs:
        return

    removed = 0
    for state_dir, _state_step in step_state_dirs:
        try:
            shutil.rmtree(state_dir)
            removed += 1
        except OSError as exc:
            logger(f"  cleanup warning: could not remove {state_dir.name}: {exc}")

    logger(f"  cleanup: kept final state {last_state_dir.name}, removed {removed} step state folder(s)")


def run_command(
    args: Iterable[str],
    cwd: Path,
    cancel_requested: Callable[[], bool] | None = None,
    logger: Callable[[str], None] | None = None,
    stream_to_logger: bool = False,
    stream_mode: str = "plain",
) -> None:
    process: subprocess.Popen | None = None
    log_path: Path | None = None
    read_offset = 0
    partial_line = ""
    recent_lines: deque[str] = deque(maxlen=40)
    cache_progress_re = re.compile(r"^\s*\d+it\s+\[")
    train_progress_re = re.compile(r"^steps:\s+")

    def flush_new_output() -> None:
        nonlocal read_offset, partial_line
        if log_path is None or not log_path.exists():
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
            if stream_to_logger and logger is not None:
                if stream_mode == "cache_progress":
                    if cache_progress_re.search(cleaned):
                        logger("\r" + cleaned)
                elif stream_mode == "train_progress":
                    if train_progress_re.search(cleaned):
                        logger("\r" + cleaned)
                else:
                    logger(cleaned)

    def flush_partial_line() -> None:
        nonlocal partial_line
        if not partial_line:
            return
        cleaned = partial_line.rstrip("\r\n")
        recent_lines.append(cleaned)
        if stream_to_logger and logger is not None:
            if stream_mode == "cache_progress":
                if cache_progress_re.search(cleaned):
                    logger("\r" + cleaned)
            elif stream_mode == "train_progress":
                if train_progress_re.search(cleaned):
                    logger("\r" + cleaned)
            else:
                logger(cleaned)
        partial_line = ""

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        popen_kwargs: dict[str, object] = {
            "cwd": str(cwd),
            "env": env,
        }
        if os.name == "nt" and stream_to_logger:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = getattr(subprocess, "SW_MINIMIZE", 6)
            popen_kwargs["startupinfo"] = startupinfo

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", errors="replace", delete=False, suffix=".log") as log_file:
            log_path = Path(log_file.name)
            process = subprocess.Popen(list(args), stdout=log_file, stderr=subprocess.STDOUT, **popen_kwargs)
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


def toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def toml_string_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_quote(value) for value in values) + "]"


def normalize_model_checkpoint_path(path_value: Path, label: str) -> Path:
    candidate = path_value.expanduser()
    if not candidate.exists():
        return candidate

    if candidate.is_file() and candidate.name.lower().endswith(".safetensors.index.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} index file is invalid JSON: {candidate}\nDetails: {exc}") from exc

        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"{label} index file has no weight_map entries: {candidate}")

        shard_names = sorted({str(v) for v in weight_map.values() if isinstance(v, str) and v.lower().endswith(".safetensors")})
        preferred = next((name for name in shard_names if re.search(r"-00001-of-\d+\.safetensors$", name, flags=re.IGNORECASE)), None)
        shard_name = preferred or (shard_names[0] if shard_names else None)
        if not shard_name:
            raise RuntimeError(f"{label} index file did not reference any .safetensors shards: {candidate}")

        shard_path = candidate.parent / shard_name
        if not shard_path.is_file():
            raise RuntimeError(f"{label} shard file referenced by index does not exist: {shard_path}")
        return shard_path

    return candidate


def run_steps_for_model(
    runtime_config: KleinRuntimeConfig,
    model_name: str,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    optimizer_type: str,
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
        selected = require_model_file(path_value, "Klein Text Encoder")
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
    output_name = (output_name_override or "").strip() or f"{model_name}_Klein"
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
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
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
                    "Klein VAE appears invalid or incompatible for Flux2 caching.\n"
                    "Open Settings and verify Klein > VAE points to the correct VAE checkpoint file.\n"
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
                    "Klein Text Encoder appears invalid or incompatible for Flux2 text caching.\n"
                    "Open Settings and verify Klein > Text Encoder points to the correct checkpoint.\n"
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
        dit_path = require_model_file(runtime_config.dit, "Klein Model")
        vae_path = require_model_file(runtime_config.vae, "Klein VAE")
        text_encoder_path = resolve_text_encoder_file(runtime_config.text_encoder)
        dit_path = dit_path.resolve()
        vae_path = vae_path.resolve()
        text_encoder_path = text_encoder_path.resolve()
        logger(f"  text_encoder: {text_encoder_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

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
        logging_flags: list[str] = []
        if enable_training_logging:
            log_backend = (training_log_backend or "tensorboard").strip().lower()
            if log_backend not in {"tensorboard", "wandb", "all"}:
                log_backend = "tensorboard"
            has_tensorboard = module_available("tensorboard")
            has_wandb = module_available("wandb")
            has_wandb_key = wandb_key_available()
            requirements_hint = (
                "Install dependencies in Musubi-Tuner venv using "
                "requirements-musubi-tuner.txt from Musubi-Trainer."
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
            logging_flags.extend([
                "--log_with",
                log_backend,
                "--logging_dir",
                str(logging_dir),
            ])
            tracker_name = training_log_tracker_name.strip() or auto_tracker_name
            if tracker_name:
                logging_flags.extend(["--log_tracker_name", tracker_name])
            logger(f"  logging_dir: {logging_dir}")
            logger(f"  log_tracker_name: {tracker_name}")
        optimizer_key = (optimizer_type or "adamw8bit").strip().lower()
        if optimizer_key == "prodigy" and not module_available("prodigyopt"):
            raise RuntimeError(
                "Optimizer 'prodigy' requires the prodigyopt package. "
                "Install dependencies in Musubi-Tuner venv using requirements-musubi-tuner.txt from Musubi-Trainer."
            )
        optimizer_arg = "prodigyopt.Prodigy" if optimizer_key == "prodigy" else "adamw8bit"
        learning_rate_for_run = "1" if optimizer_key == "prodigy" else learning_rate
        optimizer_args_values: list[str] = []
        if optimizer_key == "prodigy":
            optimizer_args_values = [
                "safeguard_warmup=True",
                "use_bias_correction=True",
                "weight_decay=0.01",
                "betas=(0.9,0.99)",
            ]
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
            "gradient_checkpointing = true",
            f"gradient_checkpointing_cpu_offload = {'true' if enable_gradient_checkpointing_cpu_offload else 'false'}",
            "persistent_data_loader_workers = true",
            "max_data_loader_n_workers = 2",
            "save_every_n_steps = 250",
            "save_state = true",
            "save_state_on_train_end = true",
            "seed = 42",
            f"fp8_base = {'true' if enable_fp8_dit else 'false'}",
            f"fp8_scaled = {'true' if enable_fp8_dit else 'false'}",
            f"compile = {'true' if enable_compile_optimizations else 'false'}",
            f"cuda_allow_tf32 = {'true' if (enable_compile_optimizations and enable_cuda_allow_tf32) else 'false'}",
            f"cuda_cudnn_benchmark = {'true' if (enable_compile_optimizations and enable_cuda_cudnn_benchmark) else 'false'}",
            f"compile_cache_size_limit = {32 if enable_compile_optimizations else 0}",
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
                stream_mode="train_progress",
            )
        except TrainingCancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Training launch failed due to invalid or incompatible Klein model files.\n"
                "Open Settings and verify Klein > Model, VAE, and Text Encoder paths.\n"
                f"Details: {exc}"
            ) from exc

        logger("")


def train_models(
    runtime_config: KleinRuntimeConfig,
    model_names: list[str],
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    optimizer_type: str,
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
                learning_rate=learning_rate,
                train_steps=train_steps,
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
    runtime_config: KleinRuntimeConfig,
    dataset_name: str,
    output_name: str,
    output_dir: Path,
    default_caption_keyword: str,
    resolution: int,
    network_dim: int,
    network_alpha: int,
    optimizer_type: str,
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
            learning_rate=learning_rate,
            train_steps=train_steps,
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
