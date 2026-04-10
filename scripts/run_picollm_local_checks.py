#!/usr/bin/env python3
"""Run the common picoLLM local regression checks for CPU, MPS, or CUDA machines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard picoLLM local regression checks.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Target local device profile")
    parser.add_argument("--hf-repo-id", type=str, default="", help="Optional HF model repo id to smoke-test after the unit checks")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for subprocess checks")
    args = parser.parse_args()

    device = detect_device() if args.device == "auto" else args.device
    print(f"Local check device profile: {device}")

    run([args.python, "-m", "unittest", "tests.test_checkpoint_manager", "tests.test_identity_dataset", "tests.test_chat_interfaces", "-q"])

    if args.hf_repo_id:
        run(
            [
                args.python,
                str(REPO_ROOT / "scripts" / "smoke_picollm_model_repo.py"),
                args.hf_repo_id,
                "--device-type",
                device,
                "--python",
                args.python,
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
