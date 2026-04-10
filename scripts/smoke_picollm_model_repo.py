#!/usr/bin/env python3
"""Download the minimal picoLLM model bundle from Hugging Face and run one local chat smoke prompt."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from restore_picollm_from_hf import verify_layout


REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOW_PATTERNS = [
    "tokenizer/*",
    "base_checkpoints/*/model_*.pt",
    "base_checkpoints/*/meta_*.json",
    "chatsft_checkpoints/*/model_*.pt",
    "chatsft_checkpoints/*/meta_*.json",
    "identity_conversations.jsonl",
    "run_manifest.json",
    "picollm_model_metadata.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a published picoLLM model repo on the current machine.")
    parser.add_argument("repo_id", type=str, help="Hugging Face model repo id")
    parser.add_argument("--device-type", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Local device type used for the smoke prompt")
    parser.add_argument("--source", type=str, default="sft", choices=["base", "sft"], help="Checkpoint source for the smoke prompt")
    parser.add_argument("--prompt", type=str, default="Who are you?", help="Prompt used for the smoke test")
    parser.add_argument("--max-tokens", type=int, default=48, help="Maximum tokens for the smoke prompt")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for the CLI smoke command")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip() or None

    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            local_dir=str(base_dir),
            allow_patterns=ALLOW_PATTERNS,
            token=token,
        )
        verify_layout(base_dir)

        env = os.environ.copy()
        env["PICOLLM_BASE_DIR"] = str(base_dir)
        command = [
            args.python,
            "-m",
            "picollm.accelerated.chat.cli",
            "-i",
            args.source,
            "-p",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--device-type",
            args.device_type,
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
