"""Musubi-Trainer model download and detection utilities.

Covers all FLUX.2 / Klein model variants. Used by the Settings dialog for
auto-discovery and downloading of model files.
"""
from __future__ import annotations

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
    "klein-base-9b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
            "filename": "flux-2-klein-base-9b.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "model-00001-of-00004.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00004.safetensors",
            "shard_count": 4,
        },
    },
    "klein-9b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "flux-2-klein-9b.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-9B",
            "filename": "model-00001-of-00004.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00004.safetensors",
            "shard_count": 4,
        },
    },
    "klein-base-4b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
            "filename": "flux-2-klein-base-4b.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "model-00001-of-00002.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00002.safetensors",
            "shard_count": 2,
        },
    },
    "klein-4b": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "flux-2-klein-4b.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "filename": "model-00001-of-00002.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00002.safetensors",
            "shard_count": 2,
        },
    },
    "flux2-dev": {
        "dit": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "flux2-dev.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "ae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "filename": "model-00001-of-00010.safetensors",
            "shards": True,
            "shard_pattern": "model-{i:05d}-of-00010.safetensors",
            "shard_count": 10,
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
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
        },
    },
    "wan2.1-i2v-720p-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
        },
    },
    "wan2.1-i2v-480p-14b": {
        "dit": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
        },
        "clip": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "shards": False,
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
        },
        "dit_low": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_14B_low_noise_fp16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "shards": False,
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
        },
    },

    # ── Qwen-Image ─────────────────────────────────────────────────────
    # All filenames from qwen_image.md (Comfy-Org repackaged, explicit).
    "qwen-image": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
        },
    },
    "qwen-image-edit": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
        },
    },
    "qwen-image-edit-2509": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
        },
    },
    "qwen-image-edit-2511": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "filename": "split_files/vae/qwen_image_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
        },
    },
    "qwen-image-layered": {
        "dit": {
            "repo_id": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
            "filename": "split_files/diffusion_models/qwen_image_layered_bf16.safetensors",
            "shards": False,
        },
        "vae": {
            "repo_id": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
            "filename": "split_files/vae/qwen_image_layered_vae.safetensors",
            "shards": False,
        },
        "text_encoder": {
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
            "shards": False,
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
        },
        "text_encoder": {
            "repo_id": "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn",
            "filename": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "shards": False,
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
        result = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if result is not None and result != "no-cache":
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
    """Look for a component file inside Models/<model_name>/."""
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        return None
    folder = models_folder(ws_root) / model_name
    candidate = folder / info["filename"]
    return candidate if candidate.is_file() else None


def find_in_hf_cache(model_name: str, component: str) -> Path | None:
    """Look for a component file in the HuggingFace local cache."""
    info = MODELS.get(model_name, {}).get(component)
    if not info:
        return None
    return _hf_try_load(info["repo_id"], info["filename"])


def find_component(
    model_name: str,
    component: str,
    ws_root: Path | None = None,
) -> Path | None:
    """Check both locations and return the first found path, or None."""
    found = find_in_models_folder(model_name, component, ws_root)
    if found:
        return found
    return find_in_hf_cache(model_name, component)


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
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    repo_id: str = info["repo_id"]
    filename: str = info["filename"]

    local_dir_str: str | None = None
    if location == DOWNLOAD_LOCATION_MODELS_FOLDER:
        dest = models_folder(ws_root) / model_name
        dest.mkdir(parents=True, exist_ok=True)
        local_dir_str = str(dest)

    def _run_download(**kwargs: object) -> Path:
        try:
            if token:
                kwargs["token"] = token
            return Path(hf_hub_download(**kwargs))  # type: ignore[arg-type]
        except Exception as exc:
            msg = str(exc)
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

    Returns a dict with keys ``dit``, ``vae``, ``text_encoder`` mapping to
    resolved Path objects (or None if not found and download was skipped).
    """
    result: dict[str, Path | None] = {}
    for component in ("dit", "vae", "text_encoder"):
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
