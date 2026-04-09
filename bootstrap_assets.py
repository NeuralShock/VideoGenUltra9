#!/usr/bin/env python3

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

OFFICIAL_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
OFFICIAL_REPO_DIR = Path("external/LTX-2")
MODEL_REPO_ID = "Lightricks/LTX-2.3"
DEFAULT_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
OFFICIAL_MODELS_DIR = Path("models/official/ltx-2.3")
GEMMA_DIR = Path("models/official/gemma-3-12b-it-qat-q4_0-unquantized")
DEV_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_NAME = "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_ATTEMPTS_PER_SOURCE = 3
DOWNLOAD_STALL_TIMEOUT_SECONDS = int(os.environ.get("LTX23_DOWNLOAD_STALL_TIMEOUT_SECONDS", "25"))
GEMMA_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.jinja",
    "*.safetensors",
]


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _comma_split_env(var_name: str) -> list[str]:
    raw = os.environ.get(var_name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_hf_resolve_url(repo_id: str, filename: str, host: str) -> str:
    normalized_host = host.rstrip("/")
    encoded_filename = urllib.parse.quote(filename)
    return f"{normalized_host}/{repo_id}/resolve/main/{encoded_filename}?download=true"


def _candidate_urls(repo_id: str, filename: str, env_key: str) -> list[str]:
    explicit_urls = _comma_split_env(env_key)
    if explicit_urls:
        return explicit_urls

    hosts = _comma_split_env("LTX23_MODEL_MIRROR_BASE_URLS")
    if not hosts:
        hosts = [
            "https://huggingface.co",
            "https://hf-mirror.com",
        ]
    return [_build_hf_resolve_url(repo_id, filename, host) for host in hosts]


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _gemma_repo_candidates() -> list[str]:
    ids = _comma_split_env("LTX23_GEMMA_REPO_IDS")
    return ids if ids else [DEFAULT_GEMMA_REPO_ID]


def _gemma_assets_present(gemma_dir: Path) -> bool:
    required_files = [
        gemma_dir / "tokenizer.model",
        gemma_dir / "preprocessor_config.json",
    ]
    return all(path.exists() for path in required_files) and any(gemma_dir.rglob("model*.safetensors"))


def _is_stall_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return isinstance(reason, (TimeoutError, socket.timeout))
    return False


def _stream_download_with_resume(url: str, destination: Path, *, attempts: int = DOWNLOAD_ATTEMPTS_PER_SOURCE) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.partial")
    headers = _auth_headers()

    for attempt in range(1, attempts + 1):
        start_byte = partial_path.stat().st_size if partial_path.exists() else 0
        request_headers = dict(headers)
        if start_byte > 0:
            request_headers["Range"] = f"bytes={start_byte}-"

        request = urllib.request.Request(url, headers=request_headers)
        mode = "ab" if start_byte > 0 else "wb"
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_STALL_TIMEOUT_SECONDS) as response:
                status_code = getattr(response, "status", None)
                if status_code == 200 and start_byte > 0:
                    # Source did not honor the range request; start from scratch.
                    start_byte = 0
                    mode = "wb"

                if status_code not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status_code} from {url}")

                content_length = response.headers.get("Content-Length")
                total_expected: int | None = None
                if content_length and content_length.isdigit():
                    current_payload = int(content_length)
                    total_expected = (
                        start_byte + current_payload if status_code == 206 else current_payload
                    )

                with partial_path.open(mode) as output:
                    with tqdm(
                        total=total_expected,
                        initial=start_byte,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=destination.name,
                        dynamic_ncols=True,
                    ) as progress:
                        while True:
                            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                            if not chunk:
                                break
                            output.write(chunk)
                            progress.update(len(chunk))
                partial_path.replace(destination)
                print(f"Downloaded: {destination}")
                return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            if _is_stall_error(exc):
                print(
                    f"Detected stalled download (no data for ~{DOWNLOAD_STALL_TIMEOUT_SECONDS}s). "
                    "Restarting from partial file..."
                )
            if attempt == attempts:
                raise RuntimeError(f"Failed downloading from {url}: {exc}") from exc
            delay = min(2**attempt, 20)
            print(f"Download attempt {attempt}/{attempts} failed ({exc}); retrying in {delay}s...")
            time.sleep(delay)


def ensure_gemma_assets() -> None:
    if _gemma_assets_present(GEMMA_DIR):
        print(f"Gemma assets already present: {GEMMA_DIR}")
        return

    local_gemma_dir = os.environ.get("LTX23_GEMMA_LOCAL_DIR", "").strip()
    if local_gemma_dir:
        source = Path(local_gemma_dir).expanduser()
        if source.exists():
            print(f"Using local Gemma source: {source}")
            run(["cp", "-a", f"{source}/.", str(GEMMA_DIR)])
            if _gemma_assets_present(GEMMA_DIR):
                print(f"Gemma assets copied from local source: {GEMMA_DIR}")
                return
            print("Local Gemma source copy completed but required files are still missing.")
        else:
            print(f"Configured local Gemma source does not exist: {source}")

    errors: list[str] = []
    for repo_id in _gemma_repo_candidates():
        try:
            print(f"Trying Gemma repo: {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(GEMMA_DIR),
                allow_patterns=GEMMA_ALLOW_PATTERNS,
            )
            if _gemma_assets_present(GEMMA_DIR):
                print(f"Gemma assets ready from repo: {repo_id}")
                return
            errors.append(f"- {repo_id}: download finished but required files were not found")
        except Exception as exc:
            errors.append(f"- {repo_id}: {exc}")
            print(f"Gemma source failed: {exc}")

    detail_block = "\n".join(errors) if errors else "- (no Gemma repositories configured)"
    raise RuntimeError(
        "Unable to prepare Gemma assets.\n"
        f"Tried these sources:\n{detail_block}\n\n"
        "Alternatives:\n"
        "1) Request/accept access for the Gemma repo on Hugging Face and set HF_TOKEN.\n"
        "2) Provide one or more alternate repository ids via LTX23_GEMMA_REPO_IDS.\n"
        "3) Point LTX23_GEMMA_LOCAL_DIR to a local directory containing Gemma files."
    )


def download_with_fallbacks(repo_id: str, filename: str, destination_dir: Path, env_key: str) -> Path:
    destination = destination_dir / filename
    if destination.exists():
        print(f"Already present: {destination}")
        return destination

    urls = _candidate_urls(repo_id, filename, env_key)
    errors: list[str] = []
    for url in urls:
        print(f"Trying source: {url}")
        try:
            _stream_download_with_resume(url, destination)
            return destination
        except Exception as exc:
            errors.append(f"- {url}: {exc}")
            print(f"Source failed: {exc}")

    print("Falling back to huggingface_hub download API...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(destination_dir),
        resume_download=True,
    )
    if destination.exists():
        return destination

    details = "\n".join(errors) if errors else "(no source URLs were available)"
    raise RuntimeError(f"All checkpoint download sources failed.\n{details}")


def ensure_official_repo() -> None:
    if OFFICIAL_REPO_DIR.exists():
        print(f"Official repo already present: {OFFICIAL_REPO_DIR}")
        return

    OFFICIAL_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_REPO_DIR)])


def ensure_model_assets() -> None:
    OFFICIAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = download_with_fallbacks(
        repo_id=MODEL_REPO_ID,
        filename=DEV_CHECKPOINT_NAME,
        destination_dir=OFFICIAL_MODELS_DIR,
        env_key="LTX23_DEV_CHECKPOINT_URLS",
    )
    print(f"Model checkpoint ready: {checkpoint_path}")

    distilled_lora_path = OFFICIAL_MODELS_DIR / DISTILLED_LORA_NAME
    if distilled_lora_path.exists():
        print(f"Distilled LoRA already present: {distilled_lora_path}")
    else:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=DISTILLED_LORA_NAME,
            local_dir=str(OFFICIAL_MODELS_DIR),
        )

    upscaler_path = OFFICIAL_MODELS_DIR / SPATIAL_UPSCALER_NAME
    if upscaler_path.exists():
        print(f"Spatial upscaler already present: {upscaler_path}")
    else:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=SPATIAL_UPSCALER_NAME,
            local_dir=str(OFFICIAL_MODELS_DIR),
        )

    ensure_gemma_assets()


def main() -> None:
    try:
        ensure_official_repo()
        ensure_model_assets()
    except Exception as exc:  # pragma: no cover - setup helper
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
