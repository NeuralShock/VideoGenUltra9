#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import sys
from importlib.util import find_spec
from pathlib import Path

PROMPT = "A sunrise over the ocean"
NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
OUTPUT_DIR = Path("output")
OFFICIAL_REPO_DIR = Path("external/LTX-2")
OFFICIAL_MODELS_DIR = Path("models/official/ltx-2.3")
GEMMA_DIR = Path("models/official/gemma-3-12b-it-qat-q4_0-unquantized")
DEV_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_NAME = "ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
TARGET_RAM_BUDGET_GB = 80
DEVICE_MODE = os.environ.get("VIDEOGEN_DEVICE", "auto").strip().lower()
WIDTH = 768
HEIGHT = 512
FPS = 24.0
NUM_FRAMES = 121
NUM_INFERENCE_STEPS = 30
VIDEO_GUIDER_CONFIG = {
    "cfg_scale": 3.0,
    "stg_scale": 1.0,
    "rescale_scale": 0.7,
    "modality_scale": 3.0,
    "skip_step": 0,
    "stg_blocks": [28],
}
AUDIO_GUIDER_CONFIG = {
    "cfg_scale": 7.0,
    "stg_scale": 1.0,
    "rescale_scale": 0.7,
    "modality_scale": 3.0,
    "skip_step": 0,
    "stg_blocks": [28],
}


def bootstrap_official_repo() -> None:
    core_src = OFFICIAL_REPO_DIR / "packages" / "ltx-core" / "src"
    pipelines_src = OFFICIAL_REPO_DIR / "packages" / "ltx-pipelines" / "src"

    if not core_src.exists() or not pipelines_src.exists():
        raise RuntimeError(
            "The official LTX-2 repository was not found under external/LTX-2.\n"
            "Clone it first with:\n"
            "git clone https://github.com/Lightricks/LTX-2.git external/LTX-2"
        )

    for path in (core_src, pipelines_src):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def validate_runtime_dependencies() -> None:
    missing = [name for name in ("av", "einops", "scipy") if find_spec(name) is None]
    if not missing:
        return

    missing_list = ", ".join(missing)
    raise RuntimeError(
        "The official LTX-2 runtime dependencies are incomplete.\n"
        f"Missing Python packages: {missing_list}\n\n"
        "Install them with:\n"
        "python -m pip install av einops scipy"
    )


def validate_local_assets() -> tuple[Path, Path, Path, Path]:
    required_paths = [
        OFFICIAL_REPO_DIR / "packages" / "ltx-core" / "src",
        OFFICIAL_REPO_DIR / "packages" / "ltx-pipelines" / "src",
        OFFICIAL_MODELS_DIR / DEV_CHECKPOINT_NAME,
        OFFICIAL_MODELS_DIR / DISTILLED_LORA_NAME,
        OFFICIAL_MODELS_DIR / SPATIAL_UPSCALER_NAME,
        GEMMA_DIR / "tokenizer.model",
        GEMMA_DIR / "preprocessor_config.json",
    ]

    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return (
            OFFICIAL_MODELS_DIR / DEV_CHECKPOINT_NAME,
            OFFICIAL_MODELS_DIR / DISTILLED_LORA_NAME,
            OFFICIAL_MODELS_DIR / SPATIAL_UPSCALER_NAME,
            GEMMA_DIR,
        )

    missing_list = "\n".join(f"- {path}" for path in missing)
    raise RuntimeError(
        "Local LTX-2 assets are not fully prepared.\n"
        f"Missing required files or directories:\n{missing_list}\n\n"
        "Run `source ./init.sh` first. That script installs Python dependencies and "
        "downloads the official LTX-2 assets before local execution."
    )


def next_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(OUTPUT_DIR.glob("ltx23_video_*.mp4"))
    if not existing:
        return OUTPUT_DIR / "ltx23_video_0001.mp4"

    last_index = max(int(path.stem.rsplit("_", 1)[-1]) for path in existing if path.stem.rsplit("_", 1)[-1].isdigit())
    return OUTPUT_DIR / f"ltx23_video_{last_index + 1:04d}.mp4"


def select_device(torch) -> torch.device:
    if DEVICE_MODE not in {"auto", "xpu", "cpu"}:
        raise ValueError("VIDEOGEN_DEVICE must be one of: auto, xpu, cpu")

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if DEVICE_MODE == "xpu":
        if not xpu_available:
            raise RuntimeError(
                "VIDEOGEN_DEVICE=xpu was requested, but torch.xpu is not available. "
                "Install Intel GPU drivers and XPU PyTorch wheels first."
            )
        return torch.device("xpu")

    if DEVICE_MODE == "cpu":
        return torch.device("cpu")

    return torch.device("xpu") if xpu_available else torch.device("cpu")


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    validate_runtime_dependencies()
    dev_checkpoint, distilled_lora, spatial_upsampler, gemma_root = validate_local_assets()
    bootstrap_official_repo()

    import torch
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.constants import DEFAULT_LORA_STRENGTH
    from ltx_pipelines.utils.media_io import encode_video

    device = select_device(torch)
    seed = random.SystemRandom().randrange(0, 2**63 - 1)
    output_path = next_output_path()
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(dev_checkpoint),
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                str(distilled_lora),
                DEFAULT_LORA_STRENGTH,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ],
        spatial_upsampler_path=str(spatial_upsampler),
        gemma_root=str(gemma_root),
        loras=[],
        device=device,
        quantization=None,
        torch_compile=False,
    )

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")
    print(f"Target RAM budget: ~{TARGET_RAM_BUDGET_GB} GB")

    video, audio = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=seed,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        frame_rate=FPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        video_guider_params=MultiModalGuiderParams(**VIDEO_GUIDER_CONFIG),
        audio_guider_params=MultiModalGuiderParams(**AUDIO_GUIDER_CONFIG),
        images=[],
        tiling_config=tiling_config,
        enhance_prompt=False,
        streaming_prefetch_count=None,
        max_batch_size=1,
    )

    encode_video(
        video=video,
        fps=int(FPS),
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )

    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
