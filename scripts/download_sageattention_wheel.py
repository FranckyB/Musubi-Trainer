"""Download a SageAttention-for-windows wheel that matches the current torch runtime.

This script targets GitHub releases from sdbds/SageAttention-for-windows and picks
the best wheel match for:
- torch version (major.minor.patch preferred)
- CUDA tag (cu124/cu128/cu130)
- Python ABI tag (cp311, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve


def detect_runtime_tokens() -> tuple[str, str, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(f"Unable to import torch from current interpreter: {exc}") from exc

    torch_version_raw = str(getattr(torch, "__version__", "")).strip()
    torch_base = torch_version_raw.split("+", 1)[0]

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    cuda_tag = ""
    plus_match = re.search(r"\+(cu\d+)", torch_version_raw.lower())
    if plus_match:
        cuda_tag = plus_match.group(1)
    else:
        cuda_raw = str(getattr(torch.version, "cuda", "") or "").strip()
        if cuda_raw:
            parts = cuda_raw.split(".")
            major = parts[0]
            minor = parts[1] if len(parts) > 1 else "0"
            cuda_tag = f"cu{major}{minor}"

    if not torch_base:
        raise RuntimeError("Could not detect torch version.")
    if not cuda_tag:
        raise RuntimeError("Could not detect CUDA tag from torch runtime.")

    return torch_base, cuda_tag.lower(), py_tag.lower()


def parse_torch_version(torch_base: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_base)
    if not match:
        raise RuntimeError(f"Unsupported torch version format: {torch_base}")
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return major, minor, patch


def parse_asset_torch_version(name: str) -> tuple[int, int, int] | None:
    name_l = name.lower()

    dotted = re.search(r"torch(\d+)\.(\d+)(?:\.(\d+))?", name_l)
    if dotted:
        major = int(dotted.group(1))
        minor = int(dotted.group(2))
        patch = int(dotted.group(3) or 0)
        return major, minor, patch

    compact = re.search(r"torch(\d{2,5})(?=[^0-9]|$)", name_l)
    if not compact:
        return None

    digits = compact.group(1)
    if len(digits) == 2:
        return int(digits[0]), int(digits[1]), 0
    if len(digits) == 3:
        return int(digits[0]), int(digits[1]), int(digits[2])
    if len(digits) == 4:
        return int(digits[0]), int(digits[1:3]), int(digits[3])
    if len(digits) == 5:
        return int(digits[0]), int(digits[1:3]), int(digits[3:5])
    return None


def github_releases(repo: str) -> list[dict]:
    releases: list[dict] = []
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    page = 1
    while page <= 10:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "musubi-trainer-setup"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            detail = f"GitHub API request failed ({exc.code} {exc.reason}) for {url}."
            if exc.code == 403:
                detail += " You may have hit GitHub API rate limits. Set GITHUB_TOKEN and retry."
            raise RuntimeError(detail) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach GitHub API for {url}: {exc}") from exc
        if not data:
            break
        if not isinstance(data, list):
            break
        releases.extend(data)
        page += 1
    return releases


def score_asset(name: str, torch_base: str, cuda_tag: str, py_tag: str) -> int:
    name_l = name.lower()
    if not name_l.endswith(".whl"):
        return -1
    if name_l.endswith(".whl.whl"):
        return -1
    if "win_amd64" not in name_l:
        return -1
    if py_tag not in name_l:
        return -1
    if cuda_tag not in name_l:
        return -1

    target_major, target_minor, target_patch = parse_torch_version(torch_base)
    asset_version = parse_asset_torch_version(name)
    if asset_version is None:
        return -1

    asset_major, asset_minor, asset_patch = asset_version
    if asset_major != target_major:
        return -1

    # Prefer exact torch match, then closest lower/equal torch build in the same major.
    # This keeps selection deterministic when exact wheels are unavailable upstream.
    score = 10
    if asset_minor == target_minor and asset_patch == target_patch:
        score += 300
    elif asset_minor == target_minor and asset_patch < target_patch:
        score += 260 - (target_patch - asset_patch) * 5
    elif asset_minor < target_minor:
        score += 200 - (target_minor - asset_minor) * 20 - abs(target_patch - asset_patch)
    else:
        # Prefer lower/equal torch versions; higher versions are unlikely to be ABI-safe.
        return -1

    if "sageattention" in name_l:
        score += 10
    return score


def find_best_asset(releases: list[dict], torch_base: str, cuda_tag: str, py_tag: str) -> dict | None:
    best: tuple[int, dict] | None = None
    for rel in releases:
        assets = rel.get("assets") or []
        if not isinstance(assets, list):
            continue
        for asset in assets:
            name = str(asset.get("name") or "")
            s = score_asset(name, torch_base, cuda_tag, py_tag)
            if s < 0:
                continue
            if best is None or s > best[0]:
                best = (s, asset)
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="sdbds/SageAttention-for-windows")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    torch_base, cuda_tag, py_tag = detect_runtime_tokens()
    releases = github_releases(args.repo)
    if not releases:
        raise RuntimeError(f"No releases found for {args.repo}")

    best = find_best_asset(releases, torch_base, cuda_tag, py_tag)
    if best is None:
        raise RuntimeError(
            f"No matching SageAttention wheel found for torch={torch_base}, cuda={cuda_tag}, py={py_tag}."
        )

    url = str(best.get("browser_download_url") or "").strip()
    name = str(best.get("name") or "").strip()
    if not url or not name:
        raise RuntimeError("Matched SageAttention asset is missing download metadata.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    urlretrieve(url, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
