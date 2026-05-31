"""Shared training utilities used by all model-specific train modules.

This module contains:
- Common constants (defaults, exit codes, file patterns)
- TrainingCancelledError
- File / checkpoint / state helpers (model-agnostic *_for_output variants)
- Dataset preparation helpers
- run_command, require_model_file, format_command_for_log
- TOML helpers, normalize_model_checkpoint_path
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Common defaults
# ---------------------------------------------------------------------------
DEFAULT_RESOLUTION = 1024
DEFAULT_NETWORK_DIM = 32
DEFAULT_NETWORK_ALPHA = 32
DEFAULT_LEARNING_RATE = "1e-4"
DEFAULT_TRAIN_STEPS = 3000
DEFAULT_SAVE_EVERY_N_STEPS = 250

# Job exit codes
JOB_EXIT_SUCCESS = 0
JOB_EXIT_FAILED = 1
JOB_EXIT_CANCELLED = 2

# File patterns
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SHARDED_SAFETENSORS_NAME_RE = re.compile(r".+-\d{5}-of-\d{5}\.safetensors$", re.IGNORECASE)
JOB_PROGRESS_FILE_NAME = "progress.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TrainingCancelledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Log directory helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Checkpoint / state helpers  (model-agnostic, *_for_output variants)
# ---------------------------------------------------------------------------
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


def write_recorded_completed_steps(
    output_dir: Path, output_name: str, completed_step: int, target_steps: int
) -> None:
    metadata_path = progress_metadata_path_for_output(output_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_name": output_name,
        "completed_step": int(max(0, completed_step)),
        "target_steps": int(max(0, target_steps)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finished_checkpoint_for_output(output_dir: Path, output_name: str) -> Path | None:
    if not output_dir.exists():
        return None
    finished_path = output_dir / f"{output_name}.safetensors"
    if finished_path.is_file():
        return finished_path
    return None


def latest_resume_state_for_output(
    output_dir: Path, output_name: str, checkpoint_step: int
) -> tuple[Path | None, int]:
    if not output_dir.exists():
        return None, 0

    if checkpoint_step > 0:
        step_state_dir = output_dir / f"{output_name}-step{checkpoint_step:08d}-state"
        if step_state_dir.is_dir() and (step_state_dir / "scheduler.bin").exists():
            return step_state_dir, checkpoint_step

    last_state_dir = output_dir / f"{output_name}-state"
    if last_state_dir.is_dir() and (last_state_dir / "scheduler.bin").exists():
        return last_state_dir, checkpoint_step

    return None, 0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
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
        return sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

    configured_images_dir = dataset_image_directory_from_config(training_dir, dataset_name)
    if configured_images_dir is None:
        return []
    if not configured_images_dir.exists() or not configured_images_dir.is_dir():
        return []

    return sorted(
        p for p in configured_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


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
            "\n".join([
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
            ]),
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


# ---------------------------------------------------------------------------
# Resume artifact remapping
# ---------------------------------------------------------------------------
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
                f"  rename warning: could not rename {resume_last_state.name} -> "
                f"{target_last_state.name}: {exc}"
            )

    if renamed_ckpts > 0 or renamed_states > 0:
        logger(
            "  rename: remapped resumed artifacts with continued step numbering "
            f"(+{resume_step_offset}, checkpoints={renamed_ckpts}, states={renamed_states})"
        )


# ---------------------------------------------------------------------------
# Step-state cleanup
# ---------------------------------------------------------------------------
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


def cleanup_step_states_for_cancel_output(
    output_dir: Path, output_name: str, logger: Callable[[str], None]
) -> None:
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


def cleanup_step_states_for_completed_output(
    output_dir: Path, output_name: str, logger: Callable[[str], None]
) -> None:
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


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------
def run_command(
    args: Iterable[str],
    cwd: Path,
    cancel_requested: Callable[[], bool] | None = None,
    logger: Callable[[str], None] | None = None,
    stream_to_logger: bool = False,
    stream_mode: str = "plain",
    inherit_io: bool = False,
) -> None:
    import sys as _sys
    process: subprocess.Popen | None = None
    log_path: Path | None = None
    read_offset = 0
    partial_line = ""
    recent_lines: deque[str] = deque(maxlen=40)
    cache_progress_re = re.compile(r"^\s*\d+it\s+\[")
    train_progress_re = re.compile(r"^steps:\s+")
    # Echo to the real console when one is attached (python, not pythonw)
    _console = _sys.stdout if (_sys.stdout is not None and hasattr(_sys.stdout, 'write')) else None

    def _echo_console(cleaned: str, prefix: str = "") -> None:
        if _console is None:
            return
        try:
            _console.write(prefix + cleaned + "\n")
            _console.flush()
        except Exception:
            pass

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
            if stream_mode == "cache_progress":
                if cache_progress_re.search(cleaned):
                    _echo_console(cleaned, "\r")
                    if stream_to_logger and logger is not None:
                        logger("\r" + cleaned)
                else:
                    _echo_console(cleaned)
            elif stream_mode == "train_progress":
                if train_progress_re.search(cleaned):
                    _echo_console(cleaned, "\r")
                    if stream_to_logger and logger is not None:
                        logger("\r" + cleaned)
                else:
                    _echo_console(cleaned)
            else:
                _echo_console(cleaned)
                if stream_to_logger and logger is not None:
                    logger(cleaned)

    def flush_partial_line() -> None:
        nonlocal partial_line
        if not partial_line:
            return
        cleaned = partial_line.rstrip("\r\n")
        recent_lines.append(cleaned)
        if stream_mode == "cache_progress":
            if cache_progress_re.search(cleaned):
                _echo_console(cleaned, "\r")
                if stream_to_logger and logger is not None:
                    logger("\r" + cleaned)
            else:
                _echo_console(cleaned)
        elif stream_mode == "train_progress":
            if train_progress_re.search(cleaned):
                _echo_console(cleaned, "\r")
                if stream_to_logger and logger is not None:
                    logger("\r" + cleaned)
            else:
                _echo_console(cleaned)
        else:
            _echo_console(cleaned)
            if stream_to_logger and logger is not None:
                logger(cleaned)
        partial_line = ""

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        musubi_src = cwd / "src"
        if musubi_src.exists() and musubi_src.is_dir():
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(musubi_src)
                if not existing_pythonpath
                else f"{musubi_src}{os.pathsep}{existing_pythonpath}"
            )
        popen_kwargs: dict[str, object] = {
            "cwd": str(cwd),
            "env": env,
        }
        if os.name == "nt" and stream_to_logger:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = getattr(subprocess, "SW_MINIMIZE", 6)
            popen_kwargs["startupinfo"] = startupinfo

        if inherit_io:
            if os.name == "nt":
                has_parent_console = bool(_console is not None and getattr(_console, "isatty", lambda: False)())
                if has_parent_console:
                    # Reuse the current console when one is already attached.
                    launch_args = list(args)
                else:
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
                    # Wrap with cmd so the window stays open on failure when detached.
                    cmd_str = subprocess.list2cmdline([str(a) for a in args])
                    launch_args = [
                        "cmd", "/c",
                        f"{cmd_str} || (echo. & echo --- Process failed. Press any key to close. --- & pause > nul)",
                    ]
            else:
                launch_args = list(args)
            process = subprocess.Popen(launch_args, **popen_kwargs)
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
                        raise RuntimeError(
                            f"Command failed with exit code {return_code}: {' '.join(str(a) for a in args)}"
                        )
                    return
                time.sleep(0.2)

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", errors="replace", delete=False, suffix=".log"
        ) as log_file:
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
                                f"Command failed with exit code {return_code}: {' '.join(str(a) for a in args)}\n"
                                f"--- command output (tail) ---\n{output_tail}"
                            )
                        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(str(a) for a in args)}")
                    return
                time.sleep(0.2)
    finally:
        if log_path is not None and log_path.exists():
            try:
                log_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def format_command_for_log(args: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(arg) for arg in args])


def require_model_file(path_value: Path | None, label: str) -> Path:
    """Validate and return a required model file path."""
    if path_value is None:
        raise RuntimeError(f"{label} is not configured. Open Settings and select a file for {label}.")
    normalized = normalize_model_checkpoint_path(path_value, label)
    if not normalized.is_file():
        raise RuntimeError(f"{label} file does not exist: {normalized}")
    return normalized


def module_available(module_name: str, python_path: Path | None = None) -> bool:
    """Check if a Python module is importable in the given Python."""
    import sys
    exe = str(python_path) if python_path else sys.executable
    result = subprocess.run(
        [exe, "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    return result.returncode == 0


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

        shard_names = sorted({
            str(v) for v in weight_map.values()
            if isinstance(v, str) and v.lower().endswith(".safetensors")
        })
        preferred = next(
            (name for name in shard_names if re.search(r"-00001-of-\d+\.safetensors$", name, flags=re.IGNORECASE)),
            None,
        )
        shard_name = preferred or (shard_names[0] if shard_names else None)
        if not shard_name:
            raise RuntimeError(f"{label} index file did not reference any .safetensors shards: {candidate}")

        shard_path = candidate.parent / shard_name
        if not shard_path.is_file():
            raise RuntimeError(f"{label} shard file referenced by index does not exist: {shard_path}")
        return shard_path

    return candidate
