"""Musubi-Trainer model download and detection utilities.

Covers all FLUX.2 / Klein model variants. Used by the Settings dialog for
auto-discovery and downloading of model files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each component entry contains:
#   repo_id    – HuggingFace repository
#   filename   – primary (or first-shard) filename
#   shards     – True when the component is split across multiple shard files
#   shard_pattern – Python format string, e.g. "model-{i:05d}-of-00004.safetensors"
#   shard_count   – total number of shards (only used when shards=True)

MODELS: dict[str, dict[str, dict]] = {
    # ── FLUX.2 ─────────────────────────────────────────────────────────
    # VAE is shared across all FLUX.2 variants → folder "flux2-vae"
    # Qwen3-8B TE is shared by all klein 9B variants → folder "qwen3-8b"
    # Qwen3-4B TE is shared by all klein 4B variants → folder "qwen3-4b"
    # Mistral3 TE is for flux2-dev → folder "mistral3"
    "klein-base-9b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
            "filename": "flux-2-klein-base-9b.safetensors",
            "shards": False,
            "folder_name": "klein-base-9b",
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
            "folder_name": "flux2-vae",
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "model-00001-of-00004.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00004.safetensors",
            "shard_count": 4,
            "folder_name": "qwen3-8b",
        },
    },
    "klein-9b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "flux-2-klein-9b.safetensors",
            "shards": False,
            "folder_name": "klein-9b",
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
            "folder_name": "flux2-vae",
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "model-00001-of-00004.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00004.safetensors",
            "shard_count": 4,
            "folder_name": "qwen3-8b",
        },
    },
    "klein-base-4b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
            "filename": "flux-2-klein-base-4b.safetensors",
            "shards": False,
            "folder_name": "klein-base-4b",
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
            "folder_name": "flux2-vae",
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "model-00001-of-00002.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00002.safetensors",
            "shard_count": 2,
            "folder_name": "qwen3-4b",
        },
    },
    "klein-4b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "flux-2-klein-4b.safetensors",
            "shards": False,
            "folder_name": "klein-4b",
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
            "folder_name": "flux2-vae",
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "model-00001-of-00002.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00002.safetensors",
            "shard_count": 2,
            "folder_name": "qwen3-4b",
        },
    },
    "flux2-dev": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "flux2-dev.safetensors",
            "shards": False,
            "folder_name": "flux2-dev",
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
            "folder_name": "flux2-vae",
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "model-00001-of-00010.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00010.safetensors",
            "shard_count": 10,
            "folder_name": "mistral3",
        },
    },

    # ── Wan 2.1 ────────────────────────────────────────────────────────
    # T5 / CLIP / VAE filenames from wan.md docs (explicit).
    # DiT filenames follow Comfy-Org repackaged naming convention.
    "wan2.1-t2v-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
            "shards": False,
            "folder_name": "wan2.1-t2v-14b",
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
            "folder_name": "wan-vae",
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
            "folder_name": "wan-t5",
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
            "folder_name": "wan-clip",
        },
    },
    "wan2.1-i2v-720p-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors",
            "shards": False,
            "folder_name": "wan2.1-i2v-720p-14b",
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
            "folder_name": "wan-vae",
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
            "folder_name": "wan-t5",
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
            "folder_name": "wan-clip",
        },
    },
    "wan2.1-i2v-480p-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors",
            "shards": False,
            "folder_name": "wan2.1-i2v-480p-14b",
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
            "folder_name": "wan-vae",
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
            "folder_name": "wan-t5",
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
            "folder_name": "wan-clip",
        },
    },

    # ── Wan 2.2 ────────────────────────────────────────────────────────
    # Wan2.2 uses two DiT models (high-noise + low-noise). CLIP not required.
    # VAE is the same Wan2.1 VAE (Wan2.2_VAE.pth is 5B only, not 14B).
    # DiT filenames follow Comfy-Org repackaged naming convention.
    "wan2.2-t2v-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_14B_high_noise_fp16.safetensors",
            "shards": False,
            "folder_name": "wan2.2-t2v-14b",
        },
        "dit_low": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_14B_low_noise_fp16.safetensors",
            "shards": False,
            "folder_name": "wan2.2-t2v-14b",
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
            "folder_name": "wan-vae",
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
            "folder_name": "wan-t5",
        },
    },

    # ── Z-Image ────────────────────────────────────────────────────────
    # Base model: Tongyi-MAI/Z-Image — split safetensors, filenames not
    # documented explicitly; use the De-Turbo single-file entry below where
    # an explicit filename is available.
    # De-Turbo DiT (single file, explicitly named in zimage.md):
    "zimage-de-turbo": {
        "dit": {
            "repo_id": "ostris/Z-Image-De-Turbo",
            "filename": "z_image_de_turbo_v1_bf16.safetensors",
            "shards": False,
            "folder_name": "zimage-de-turbo",
        },
    },

    # ── Qwen-Image ─────────────────────────────────────────────────────
    # All filenames from qwen_image.md (Comfy-Org repackaged, explicit).
    # Qwen 2.5 VL 7B TE is shared across all Qwen-Image variants → "qwen25-vl-7b"
    # Qwen-Image VAE is shared by standard variants → "qwen-image-vae"
    "qwen-image": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_bf16.safetensors",
            "shards": False,
            "folder_name": "qwen-image",
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
            "folder_name": "qwen-image-vae",
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
            "folder_name": "qwen25-vl-7b",
        },
    },
    "qwen-image-edit": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_bf16.safetensors",
            "shards": False,
            "folder_name": "qwen-image-edit",
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
            "folder_name": "qwen-image-vae",
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
            "folder_name": "qwen25-vl-7b",
        },
    },
    "qwen-image-edit-2509": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors",
            "shards": False,
            "folder_name": "qwen-image-edit-2509",
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
            "folder_name": "qwen-image-vae",
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
            "folder_name": "qwen25-vl-7b",
        },
    },
    "qwen-image-edit-2511": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "shards": False,
            "folder_name": "qwen-image-edit-2511",
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
            "folder_name": "qwen-image-vae",
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
            "folder_name": "qwen25-vl-7b",
        },
    },
    "qwen-image-layered": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_layered_bf16.safetensors",
            "shards": False,
            "folder_name": "qwen-image-layered",
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
            "filename": "split_files/vae/qwen_image_layered_vae.safetensors",
            "shards": False,
            "folder_name": "qwen-image-layered-vae",
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
            "folder_name": "qwen25-vl-7b",
        },
    },

    # ── LTX-2.3 ────────────────────────────────────────────────────────
    # Single combined checkpoint (no separate VAE file).
    # Text encoder: using the FP8 single-file Gemma variant (ltx_2.md).
    "ltx-2.3": {
        "dit": {
            "repo_id": "Lightricks/LTX-2.3",
            "filename": "ltx-2.3-22b-dev.safetensors",
            "shards": False,
            "folder_name": "ltx-2.3",
        },
        "text_encoder": {
            "repo_id": "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn",
            "filename": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "shards": False,
            "folder_name": "gemma3-12b",
        },
    },
}

DOWNLOAD_LOCATION_MODELS_FOLDER = "Models Folder"
DOWNLOAD_LOCATION_HUGGINGFACE = "HuggingFace Cache"
DOWNLOAD_LOCATIONS: tuple[str, ...] = (
    DOWNLOAD_LOCATION_MODELS_FOLDER,
    DOWNLOAD_LOCATION_HUGGINGFACE,
)

# ---------------------------------------------------------------------------
# Model family metadata (for UI grouping)
# ---------------------------------------------------------------------------

MODEL_FAMILIES: dict[str, list[str]] = {
    "FLUX.2": ["flux2-dev", "klein-base-9b", "klein-9b", "klein-base-4b", "klein-4b"],
    "LTX": ["ltx-2.3"],
    "Wan": ["wan2.1-t2v-14b", "wan2.1-i2v-720p-14b", "wan2.1-i2v-480p-14b", "wan2.2-t2v-14b"],
    "Z-Image": ["zimage-de-turbo"],
    "Qwen-Image": ["qwen-image", "qwen-image-edit", "qwen-image-edit-2509", "qwen-image-edit-2511", "qwen-image-layered"],
}

# Maps folder_name → human-readable identity shown next to component labels in the UI.
# Shared components (VAE, TE) appear under multiple models, so this tells the user
# exactly what file they are expected to provide.
COMPONENT_FRIENDLY_NAMES: dict[str, str] = {
    # FLUX.2
    "flux2-dev":    "FLUX.2-dev DiT",
    "flux2-vae":    "FLUX.2 AE (ae.safetensors)",
    "qwen3-8b":     "Qwen3 8B Text Encoder",
    "qwen3-4b":     "Qwen3 4B Text Encoder",
    "mistral3":     "Mistral3 Text Encoder",
    "klein-base-9b": "Klein Base 9B DiT",
    "klein-9b":     "Klein 9B DiT",
    "klein-base-4b": "Klein Base 4B DiT",
    "klein-4b":     "Klein 4B DiT",
    # LTX
    "ltx-2.3":      "LTX-2.3 Checkpoint",
    "gemma3-12b":   "Gemma 3 12B Text Encoder (FP8)",
    # Wan
    "wan2.1-t2v-14b":      "Wan 2.1 T2V DiT",
    "wan2.1-i2v-720p-14b": "Wan 2.1 I2V 720p DiT",
    "wan2.1-i2v-480p-14b": "Wan 2.1 I2V 480p DiT",
    "wan2.2-t2v-14b":      "Wan 2.2 T2V DiT",
    "wan-vae":      "Wan VAE",
    "wan-t5":       "UMT5-XXL Text Encoder",
    "wan-clip":     "CLIP ViT-H",
    # Z-Image
    "zimage-de-turbo": "Z-Image De-Turbo DiT",
    "zimage-vae":   "Z-Image VAE",
    "zimage-te":    "Z-Image Text Encoder",
    # Qwen-Image
    "qwen-image":          "Qwen-Image DiT",
    "qwen-image-edit":     "Qwen-Image Edit DiT",
    "qwen-image-edit-2509": "Qwen-Image Edit 25.09 DiT",
    "qwen-image-edit-2511": "Qwen-Image Edit 25.11 DiT",
    "qwen-image-layered":  "Qwen-Image Layered DiT",
    "qwen-image-vae":      "Qwen-Image VAE",
    "qwen25-vl-7b":        "Qwen2.5 VL 7B Text Encoder",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "flux2-dev": "FLUX.2-dev",
    "klein-base-9b": "Klein base 9B",
    "klein-9b": "Klein 9B",
    "klein-base-4b": "Klein base 4B",
    "klein-4b": "Klein 4B",
    "ltx-2.3": "LTX2.3",
    "wan2.1-t2v-14b": "Wan 2.1 T2V 14B",
    "wan2.1-i2v-720p-14b": "Wan 2.1 I2V 720p 14B",
    "wan2.1-i2v-480p-14b": "Wan 2.1 I2V 480p 14B",
    "wan2.2-t2v-14b": "Wan 2.2 T2V 14B",
    "zimage-de-turbo": "Z-Image De-Turbo",
    "qwen-image": "Qwen-Image",
    "qwen-image-edit": "Qwen-Image Edit",
    "qwen-image-edit-2509": "Qwen-Image Edit 25.09",
    "qwen-image-edit-2511": "Qwen-Image Edit 25.11",
    "qwen-image-layered": "Qwen-Image Layered",
}

# Canonical model version strings used as --model_version CLI argument
# (keyed by model_name, value is the string passed to Musubi-Tuner training scripts)
MODEL_VERSION_ARGS: dict[str, str] = {
    "flux2-dev": "dev",
    "klein-base-9b": "klein-base-9b",
    "klein-9b": "klein-9b",
    "klein-base-4b": "klein-base-4b",
    "klein-4b": "klein-4b",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def workspace_root() -> Path:
    """Root of the Musubi-Trainer workspace (parent directory of src/)."""
    return Path(__file__).resolve().parent.parent


def models_folder(ws_root: Path | None = None) -> Path:
    """Return the Models/ folder path inside the workspace root."""
    return (ws_root or workspace_root()) / "Models"


def _hf_try_load(repo_id: str, filename: str) -> Path | None:
    """Check the HuggingFace local cache without downloading anything."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        # _CACHED_NO_EXIST is a sentinel returned when the file is known absent.
        # We must import it to avoid treating the sentinel as a valid path.
        try:
            from huggingface_hub import _CACHED_NO_EXIST as _sentinel  # type: ignore
        except ImportError:
            _sentinel = None  # older versions don't export it; None check is sufficient

        result = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if result is not None and result is not _sentinel:
            p = Path(str(result))
            if p.is_file():
                return p
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_in_models_folder(
    model_name: str,
    component: str,
    ws_root: Path | None = None,
) -> Path | None:
    """Look for a component file inside Models/<folder_name>/ where folder_name
    comes from the component registry (shared folders for shared weights)."""
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        return None
    folder_name = info.get("folder_name", model_name)
    folder = models_folder(ws_root) / folder_name
    # The stored filename may contain subdirectory separators (e.g. Wan Comfy-Org
    # repackaged files have "split_files/..." paths). Only keep the basename.
    bare_name = Path(info["filename"]).name
    candidate = folder / bare_name
    return candidate if candidate.is_file() else None


def find_in_hf_cache(model_name: str, component: str) -> Path | None:
    """Look for a component file in the HuggingFace local cache.

    First tries the exact registered repo_id + filename via try_to_load_from_cache.
    Falls back to scanning all accessible cached snapshots by basename — this
    catches files that were cached under a different repo (e.g. mirrors or
    repackaged repos) as long as the snapshot links are intact.
    """
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        return None

    # 1. Exact repo match (fast path)
    found = _hf_try_load(info["repo_id"], info["filename"])
    if found:
        return found

    # 2. Scan all cached repos for a file with the same basename.
    #    This handles cases where the file was cached from a mirror / repackaged repo.
    target_name = Path(info["filename"]).name
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            for rev in repo.revisions:
                for cached_file in rev.files:
                    if Path(cached_file.file_path).name == target_name:
                        p = Path(str(cached_file.file_path))
                        if p.is_file():
                            return p
    except Exception:
        pass

    return None


def find_in_extra_paths(
    model_name: str,
    component: str,
    extra_paths: list[str | Path],
) -> Path | None:
    """Recursively search a list of extra directories for the component file by basename.

    Useful for locating files downloaded by ComfyUI or other tools into non-standard locations.
    """
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        return None
    target_name = Path(info["filename"]).name
    for root_dir in extra_paths:
        root = Path(root_dir)
        if not root.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(root):
            if target_name in files:
                candidate = Path(dirpath) / target_name
                if candidate.is_file():
                    return candidate
    return None


def find_component(
    model_name: str,
    component: str,
    ws_root: Path | None = None,
    extra_paths: list[str | Path] | None = None,
) -> Path | None:
    """Check all known locations and return the first found path, or None.

    Search order: Models folder → HF cache → extra_paths (e.g. ComfyUI models dir).
    """
    found = find_in_models_folder(model_name, component, ws_root)
    if found:
        return found
    found = find_in_hf_cache(model_name, component)
    if found:
        return found
    if extra_paths:
        return find_in_extra_paths(model_name, component, extra_paths)
    return None


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_component(
    model_name: str,
    component: str,
    location: str = DOWNLOAD_LOCATION_MODELS_FOLDER,
    ws_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
    token: str | None = None,
) -> Path:
    """Download a single component (dit / vae / text_encoder) for the given model.

    Returns the path to the primary file (first shard for sharded text encoders).
    Raises RuntimeError on failure.
    """
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        raise ValueError(f"Unknown model/component: {model_name}/{component}")

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        # Disable tqdm only when running in-process without a real stdout (pythonw.exe).
        # When called from the subprocess (download_cli.py), stdout is a real pipe so
        # we leave tqdm enabled so progress is visible in the console.
        import sys as _sys
        if _sys.stdout is None:
            try:
                import huggingface_hub.utils as _hf_utils
                if hasattr(_hf_utils, "disable_progress_bars"):
                    _hf_utils.disable_progress_bars()
            except Exception:
                pass
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    repo_id: str = info["repo_id"]
    filename: str = info["filename"]

    local_dir_str: str | None = None
    if location == DOWNLOAD_LOCATION_MODELS_FOLDER:
        folder_name = info.get("folder_name", model_name)
        dest = models_folder(ws_root) / folder_name
        dest.mkdir(parents=True, exist_ok=True)
        local_dir_str = str(dest)

    def _run_download(**kwargs: object) -> Path:
        def _attempt() -> Path:
            if token:
                kwargs["token"] = token
            return Path(hf_hub_download(**kwargs))  # type: ignore[arg-type]

        try:
            return _attempt()
        except Exception as exc:
            msg = str(exc)
            # httpx "client has been closed" — reset the backend and retry once
            if "client has been closed" in msg.lower():
                try:
                    import huggingface_hub
                    if hasattr(huggingface_hub, "configure_http_backend"):
                        huggingface_hub.configure_http_backend()
                except Exception:
                    pass
                try:
                    return _attempt()
                except Exception as retry_exc:
                    msg = str(retry_exc)
                    exc = retry_exc
            if any(word in msg.lower() for word in ("401", "403", "gated", "restricted", "credentials", "unauthorized", "access")):
                raise RuntimeError(
                    f"Access denied for '{kwargs.get('repo_id')}'.\n\n"
                    "This model requires you to accept its license on HuggingFace "
                    "and provide a valid HuggingFace token in Settings."
                ) from exc
            raise RuntimeError(f"Download failed: {exc}") from exc

    if info.get("shards"):
        shard_pattern: str = info["shard_pattern"]
        shard_count: int = info["shard_count"]
        first_path: Path | None = None
        for i in range(1, shard_count + 1):
            shard_name = shard_pattern.format(i=i)
            if progress:
                progress(f"Downloading {shard_name}  ({i}/{shard_count})…")
            kwargs: dict = {"repo_id": repo_id, "filename": shard_name}
            if local_dir_str:
                kwargs["local_dir"] = local_dir_str
            path = _run_download(**kwargs)
            if i == 1:
                first_path = path
        if first_path is None:
            raise RuntimeError(f"Download produced no output for {model_name}/{component}")
        return first_path
    else:
        if progress:
            progress(f"Downloading {filename}…")
        kwargs = {"repo_id": repo_id, "filename": filename}
        if local_dir_str:
            kwargs["local_dir"] = local_dir_str
        return _run_download(**kwargs)


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

def auto_resolve_klein(
    model_name: str = "klein-base-9b",
    location: str = DOWNLOAD_LOCATION_MODELS_FOLDER,
    ws_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
    download_if_missing: bool = True,
    token: str | None = None,
) -> dict[str, Path | None]:
    """Find (and optionally download) all three components for a Klein model.

    Backward-compatible alias for ``auto_resolve_model``.
    """
    return auto_resolve_model(
        model_name=model_name,
        location=location,
        ws_root=ws_root,
        progress=progress,
        download_if_missing=download_if_missing,
        token=token,
    )


def auto_resolve_model(
    model_name: str,
    location: str = DOWNLOAD_LOCATION_MODELS_FOLDER,
    ws_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
    download_if_missing: bool = True,
    token: str | None = None,
) -> dict[str, Path | None]:
    """Find (and optionally download) all components for any registered model.

    Returns a dict mapping each component key (``dit``, ``vae``,
    ``text_encoder``, ``clip``, etc.) to a resolved Path (or None).
    """
    result: dict[str, Path | None] = {}
    components = list(MODELS.get(model_name, {}).keys())
    for component in components:
        found = find_component(model_name, component, ws_root)
        if found:
            if progress:
                progress(f"Found {component}: {found.name}")
            result[component] = found
        elif download_if_missing:
            if progress:
                progress(f"Downloading {component}…")
            result[component] = download_component(
                model_name, component, location, ws_root, progress, token
            )
        else:
            result[component] = None
    return result


def model_status(
    model_name: str,
    ws_root: Path | None = None,
) -> str:
    """Return a short human-readable status string for a model.

    Returns one of: "✓ Ready", "Partial (n/total)", "Not configured".
    """
    components = list(MODELS.get(model_name, {}).keys())
    if not components:
        return "Unknown model"
    found = sum(1 for c in components if find_component(model_name, c, ws_root) is not None)
    if found == 0:
        return "Not configured"
    if found < len(components):
        return f"Partial ({found}/{len(components)})"
    return "✓ Ready"
