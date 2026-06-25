from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import save_file


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Implementation of Algorithm 2 from https://arxiv.org/pdf/2312.02696."""
    t = sigma_rel**-2
    coeffs = [1, 7, 16 - t, 12 - t]
    roots = np.roots(coeffs)
    gamma = roots[np.isreal(roots) & (roots.real >= 0)].real.max()
    return float(gamma)


def merge_lora_weights_with_post_hoc_ema(
    path: list[str],
    no_sort: bool,
    beta1: float,
    beta2: float,
    sigma_rel: Optional[float],
    output_file: str,
) -> None:
    from musubi_tuner.utils import model_utils
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    if not no_sort:
        print("Sorting files by modification time...")
        path.sort(key=lambda x: os.path.getmtime(x))

    print(f"Loading metadata from {path[-1]}")
    with MemoryEfficientSafeOpen(path[-1]) as f:
        metadata = f.metadata()
    if metadata is None:
        print("No metadata found in the last file, proceeding without metadata.")
    else:
        print("Metadata found, using metadata from the last file.")

    print(f"Loading weights from {path[0]}")
    with MemoryEfficientSafeOpen(path[0]) as f:
        original_dtypes: dict[str, torch.dtype] = {}
        state_dict: dict[str, torch.Tensor] = {}
        for key in f.keys():
            value: torch.Tensor = f.get_tensor(key)

            if value.dtype.is_floating_point:
                original_dtypes[key] = value.dtype
                value = value.to(torch.float32)
            else:
                print(f"Skipping non-floating point tensor: {key}")

            state_dict[key] = value

    ema_count = len(path) - 1
    gamma = sigma_rel_to_gamma(sigma_rel) if sigma_rel is not None else None

    for i, file in enumerate(path[1:]):
        if sigma_rel is not None:
            t = i + 1
            beta = (1 - 1 / t) ** (gamma + 1)
        else:
            beta = beta1 + (beta2 - beta1) * (i / (ema_count - 1)) if ema_count > 1 else beta1

        print(f"Loading weights from {file} for merging with beta={beta:.4f}")
        with MemoryEfficientSafeOpen(file) as f:
            for key in f.keys():
                value = f.get_tensor(key)

                if key.endswith(".alpha"):
                    if key not in state_dict:
                        continue

                    base_value = state_dict[key]
                    if base_value.dtype.is_floating_point or value.dtype.is_floating_point:
                        alpha_matches = torch.allclose(base_value.to(torch.float32), value.to(torch.float32))
                    else:
                        alpha_matches = torch.equal(base_value, value.to(base_value.dtype))

                    if alpha_matches:
                        continue
                    raise ValueError(f"Alpha tensors for key {key} do not match across files.")

                if not value.dtype.is_floating_point:
                    print(f"Skipping non-floating point tensor: {key}")
                    continue

                if key in state_dict:
                    value = value.to(torch.float32)
                    state_dict[key] = state_dict[key] * beta + value * (1 - beta)
                else:
                    raise KeyError(f"Key {key} not found in the initial state_dict.")

    for key in state_dict:
        if key in original_dtypes:
            state_dict[key] = state_dict[key].to(original_dtypes[key])

    if metadata is not None:
        print("Updating metadata with new hashes.")
        model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    print(f"Saving merged weights to {output_file}")
    save_file(state_dict, output_file, metadata=metadata)
    print("Merging completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Musubi-Trainer Post-Hoc EMA compatibility runner.")
    parser.add_argument("--musubi_src", type=str, required=True, help="Path to Musubi-Tuner src directory.")
    parser.add_argument("path", nargs="+", help="List of paths to LoRA safetensors files.")
    parser.add_argument("--no_sort", action="store_true", help="Do not sort files by modification time.")
    parser.add_argument("--beta", type=float, default=0.95, help="Decay rate for merging weights.")
    parser.add_argument("--beta2", type=float, default=None, help="Decay rate for linear interpolation.")
    parser.add_argument("--sigma_rel", type=float, default=None, help="Relative sigma for Power Function EMA.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for merged weights.")
    args = parser.parse_args()

    musubi_src = Path(args.musubi_src).resolve()
    if not musubi_src.exists() or not musubi_src.is_dir():
        raise FileNotFoundError(f"Musubi src path not found: {musubi_src}")

    musubi_src_text = str(musubi_src)
    if musubi_src_text not in sys.path:
        sys.path.insert(0, musubi_src_text)

    beta2 = args.beta if args.beta2 is None else args.beta2
    merge_lora_weights_with_post_hoc_ema(args.path, args.no_sort, args.beta, beta2, args.sigma_rel, args.output_file)


if __name__ == "__main__":
    main()
