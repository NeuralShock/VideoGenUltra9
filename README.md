# VideoGenUltra9

Python project scaffold for building LLM-assisted video generation workflows on Intel Core Ultra 9 systems.

## Project Scope

This repository is for developing Python tools and experiments for:

- local LLM-assisted video generation
- Intel Core Ultra 9 CPU / Intel integrated AI-capable hardware
- Intel-friendly inference and optimization paths
- OpenVINO-centered deployment and acceleration
- strictly non-CUDA development

## Hardware / Software Direction

We are intentionally targeting Intel Core Ultra 9 processors and Intel-oriented software stacks.

Primary stack:

- Python 3.12+
- PyTorch with Intel XPU support
- OpenVINO
- OpenVINO GenAI
- Hugging Face Transformers
- official Lightricks `LTX-2` repository
- official `ltx-core` / `ltx-pipelines` code

Explicit non-goals:

- no CUDA
- no NVIDIA-only dependencies
- no CUDA-specific code paths
- no `torch.cuda`
- no `cupy`
- no TensorRT-only workflows

## Why This Stack

- OpenVINO and OpenVINO GenAI remain useful Intel-focused runtime layers for future optimization work.
- The current generator uses the official Lightricks `LTX-2` repository instead of an unofficial converted Diffusers snapshot.
- PyTorch with Intel `xpu` support lets us use Intel Arc graphics on Core Ultra systems without introducing CUDA.
- Intel Extension for PyTorch is not the preferred foundation for new work here because Intel has published a retirement plan and recommends using PyTorch directly going forward.

## Environment Setup

Create or activate the local virtual environment:

```bash
source ./init.sh
```

Install or refresh dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Initial Layout

- `README.md` - project overview and setup guidance
- `requirements.txt` - Python dependencies for Intel-focused LLM/video work
- `PROJECT_SCOPE.md` - persistent instructions and guardrails for the project
- `init.sh` - virtual environment bootstrap / activation helper
- `generate_ltx23_video.py` - LTX-2.3 video generation script
- `output/` - generated MP4 files

## Working Rules

- Keep all development Intel-focused.
- Prefer Intel `xpu`, OpenVINO, and Intel-supported acceleration paths.
- Never add CUDA dependencies or CUDA-specific code.
- Validate any future package additions against Intel Core Ultra compatibility before adopting them.

## LTX-2.3 Generation

The current generation script targets a higher-capacity local configuration intended to stay within an 80 GB RAM budget:

- model: official `Lightricks/LTX-2.3` `ltx-2.3-22b-dev.safetensors`
- pipeline: official `ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline`
- runtime: Intel `xpu` preferred, CPU fallback, no CUDA
- checkpoint path: `ltx-2.3-22b-dev.safetensors`
- stage-2 refinement: `ltx-2.3-22b-distilled-lora-384.safetensors`
- upscaler path: `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- Gemma assets: tokenizer / processor files from `google/gemma-3-12b-it-qat-q4_0-unquantized`
- output target: 10 second MP4
- default prompt: `A sunrise over the ocean`

Install dependencies first:

```bash
source ./init.sh
```

Run the generator:

```bash
python generate_ltx23_video.py
```

Force Intel GPU execution:

```bash
VIDEOGEN_DEVICE=xpu python generate_ltx23_video.py
```

Force CPU execution:

```bash
VIDEOGEN_DEVICE=cpu python generate_ltx23_video.py
```

What the script does:

- uses only local assets that were already prepared by `init.sh`
- imports the official `ltx-core` and `ltx-pipelines` code from `external/LTX-2`
- generates a higher-quality 121-frame video from the built-in prompt using the full dev checkpoint
- writes the next numbered file into `output/`

Example outputs:

- `output/ltx23_video_0001.mp4`
- `output/ltx23_video_0002.mp4`

Notes:

- This is a best-effort 80 GB budget setup for LTX-2.3. Real memory usage depends on package versions, allocator behavior, and host CPU features.
- The script now uses the full `22b-dev` checkpoint plus the distilled LoRA refinement stage, which is a better fit for upgraded machines with substantially more RAM.
- We intentionally moved up from the distilled-only path because 80 GB of system RAM provides enough headroom for the full two-stage official pipeline while preserving the Intel-only runtime constraints.
- Intel `xpu` execution requires Intel GPU drivers and the XPU wheel build of PyTorch. On this machine, the hardware is present, but `torch.xpu.is_available()` was still `False` in the current environment before switching wheels, which usually means the Intel GPU runtime or driver stack still needs to be installed or activated.
- We intentionally moved away from the unofficial converted Diffusers snapshot because it was missing required model weights for `connectors` and `vae`. The current path uses the official Lightricks repository instead.
- `generate_ltx23_video.py` is now local-only. It does not clone repos or download model assets during execution. All networked setup belongs in `init.sh`.
