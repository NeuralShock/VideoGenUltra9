# Project Instructions

This file is a persistent reminder of the project scope and technical boundaries.

## Mission

Build Python-based LLM and video-generation tooling for Intel Core Ultra 9 processors.

## Hard Constraints

- Never use CUDA in this project.
- Never introduce NVIDIA-only packages, examples, or instructions.
- Never write code that depends on `torch.cuda`.
- Prefer Intel XPU, Intel CPU, Intel integrated graphics, OpenVINO, and Intel-supported inference paths.

## Preferred Stack

- Python 3.12+
- PyTorch XPU-capable builds
- OpenVINO
- OpenVINO GenAI
- Optimum Quanto
- Transformers
- Diffusers

## Development Priorities

- Keep code portable across Intel Core Ultra 9 developer machines.
- Treat 80 GB RAM as the working upper budget for the model/runtime on upgraded development machines.
- Prefer Intel `xpu` execution when available, with CPU fallback only when the Intel GPU runtime is unavailable.
- Favor inference quality within that budget while keeping local workflows practical.
- Build reusable Python modules first, then scripts, then notebooks.
- Default to clear setup instructions and reproducible environments.

## Package Review Rule

Before adding any new dependency:

1. Confirm it does not require CUDA.
2. Prefer a CPU/OpenVINO/Intel-friendly path.
3. Document why it is needed for local LLM-assisted video generation.

## Implementation Notes

- Use `source ./init.sh` to enter the project environment.
- Keep requirements in `requirements.txt`.
- Add future examples under a `src/` layout when implementation begins.
