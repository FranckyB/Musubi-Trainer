from __future__ import annotations

import re
from pathlib import Path


def compact_merge_selection_token(selected_files: list[str]) -> str:
    step_values: list[int] = []
    for raw_path in selected_files:
        match = re.search(r"step0*(\d+)", Path(raw_path).name, re.IGNORECASE)
        if match:
            step_values.append(int(match.group(1)))

    if not step_values:
        return f"n{len(selected_files)}"

    unique_steps = sorted(set(step_values))
    if len(unique_steps) <= 4:
        return "s" + "-".join(str(step) for step in unique_steps)

    return f"s{unique_steps[0]}-{unique_steps[-1]}x{len(unique_steps)}"


def merge_preset_file_token(preset_name: str) -> str:
    preset_key = preset_name.strip().lower()
    if preset_key == "smooth":
        return "Smooth"
    if preset_key == "anti-overfit":
        return "NoOverfit"
    return "Balance"


def merge_preset_tooltip_text() -> str:
    return (
        "Preset guide\n"
        "Balanced: General starting point for most training runs.\n"
        "Smooth: Prefer when training converged early and you want a smoother blend.\n"
        "Anti-overfit: Prefer when late checkpoints look overfit and too close to training images."
    )


def merge_mode_tooltip_text() -> str:
    return (
        "Merge mode guide\n"
        "BETA: Uses a single constant decay rate across all checkpoints.\n"
        "BETA + BETA2: Interpolates decay from beta to beta2 across the merge order.\n"
        "SIGMA_REL: Uses Power Function EMA to compute decay schedule and reduce first-checkpoint bias."
    )


def next_merged_output_path(
    dataset_name: str,
    output_dir: Path,
    merge_mode_suffix: str,
    preset_name: str,
    selected_files: list[str],
) -> Path:
    # Keep method name in the filename to make comparisons between merge modes easy.
    method_token = re.sub(r"[^A-Za-z0-9]+", "_", merge_mode_suffix.strip()).strip("_")
    preset_token = merge_preset_file_token(preset_name)
    selection_token = compact_merge_selection_token(selected_files)
    if method_token and preset_token:
        base_name = f"{dataset_name}_merged_{method_token}_{preset_token}_{selection_token}"
    elif method_token:
        base_name = f"{dataset_name}_merged_{method_token}_{selection_token}"
    else:
        base_name = f"{dataset_name}_merged_{selection_token}"
    candidate = output_dir / f"{base_name}.safetensors"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_dir / f"{base_name}_{index}.safetensors"
        if not candidate.exists():
            return candidate
        index += 1


def post_hoc_ema_mode_args_for_preset(preset_name: str) -> dict[str, list[str]]:
    preset_key = preset_name.strip().lower()
    if preset_key == "smooth":
        return {
            "beta": ["--beta", "0.95"],
            "beta2": ["--beta", "0.95", "--beta2", "0.98"],
            "sigma_rel": ["--sigma_rel", "0.25"],
        }
    if preset_key == "anti-overfit":
        return {
            "beta": ["--beta", "0.8"],
            "beta2": ["--beta", "0.80", "--beta2", "0.90"],
            "sigma_rel": ["--sigma_rel", "0.15"],
        }
    return {
        "beta": ["--beta", "0.9"],
        "beta2": ["--beta", "0.90", "--beta2", "0.95"],
        "sigma_rel": ["--sigma_rel", "0.2"],
    }
