#!/usr/bin/env python3
"""Print the recommended picoLLM local environment commands for this machine."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_commands(device: str) -> list[str]:
    extra = "gpu" if device == "cuda" else "cpu"
    commands = [
        f"uv sync --extra {extra}",
        "source .venv/bin/activate",
        "export PICOLLM_BASE_DIR=$PWD/artifacts/picollm",
    ]
    if device == "cuda":
        commands.append("export HF_HUB_VERBOSITY=warning")
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the recommended picoLLM environment setup commands for the current platform.")
    parser.add_argument("--format", choices=["text", "shell"], default="text")
    args = parser.parse_args()

    device = detect_device()
    commands = build_commands(device)
    if args.format == "shell":
        print("\n".join(commands))
        return 0

    print(f"Detected local device path: {device}")
    print("Recommended commands:")
    for command in commands:
        print(f"- {command}")
    if device == "mps":
        print("- macOS path: use CPU extra deps, then pass `--device-type mps` to chat/web smoke commands.")
    elif device == "cuda":
        print("- CUDA path: use `uv sync --extra gpu` and prefer the accelerated smoke test plus model smoke restore.")
    else:
        print("- CPU path: use `uv sync --extra cpu` and expect only lightweight smoke tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
