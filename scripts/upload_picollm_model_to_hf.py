#!/usr/bin/env python3
"""Upload the runnable picoLLM artifact set to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from picollm_release_lib import (
    DEFAULT_BASE_DIR,
    MODEL_METADATA_FILENAME,
    create_hf_repo,
    maybe_load_run_manifest,
    require_hf_token,
    stage_inference_bundle,
    upload_path,
)
from restore_picollm_from_hf import verify_layout


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_post_upload_smoke(
    repo_id: str,
    prompt: str,
    max_tokens: int,
    device_type: str,
    python_executable: str,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        command = [
            python_executable,
            str(REPO_ROOT / "scripts" / "restore_picollm_from_hf.py"),
            repo_id,
            "--base-dir",
            temp_dir,
            "--source",
            "sft",
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
        ]
        if device_type:
            command.extend(["--device-type", device_type])
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the runnable picoLLM artifact set to a Hugging Face model repo.")
    parser.add_argument("repo_id", type=str, help="Destination Hugging Face model repo id")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Local PICOLLM_BASE_DIR to publish")
    parser.add_argument("--public", action="store_true", help="Create or update the destination repo as public instead of private")
    parser.add_argument("--dry-run", action="store_true", help="Verify layout and print the upload plan without calling the Hugging Face CLI")
    parser.add_argument("--post-upload-smoke", action="store_true", help="After upload, restore the repo into a temp dir and run one CLI smoke prompt")
    parser.add_argument("--post-upload-smoke-prompt", type=str, default="Who are you?", help="Prompt used for the optional post-upload smoke test")
    parser.add_argument("--post-upload-smoke-max-tokens", type=int, default=48, help="Maximum tokens for the optional post-upload smoke test")
    parser.add_argument("--post-upload-smoke-device-type", type=str, default="", choices=["", "cpu", "mps", "cuda"], help="Device type for the optional post-upload smoke test")
    parser.add_argument("--python", type=str, default="python", help="Python executable used for the optional post-upload smoke test")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    verify_layout(base_dir)
    token = require_hf_token(dry_run=args.dry_run, purpose="model repo upload")
    create_hf_repo(
        repo_id=args.repo_id,
        repo_type="model",
        public=args.public,
        token=token,
        dry_run=args.dry_run,
    )

    with tempfile.TemporaryDirectory() as staging_dir_name:
        staging_dir = Path(staging_dir_name)
        metadata = stage_inference_bundle(base_dir=base_dir, repo_id=args.repo_id, staging_dir=staging_dir)
        upload_path(args.repo_id, "model", staging_dir / "tokenizer", "tokenizer", "Upload picoLLM tokenizer", token, args.dry_run)
        upload_path(
            args.repo_id,
            "model",
            staging_dir / "base_checkpoints",
            "base_checkpoints",
            "Upload picoLLM base inference checkpoints",
            token,
            args.dry_run,
        )
        upload_path(
            args.repo_id,
            "model",
            staging_dir / "chatsft_checkpoints",
            "chatsft_checkpoints",
            "Upload picoLLM SFT inference checkpoints",
            token,
            args.dry_run,
        )
        upload_path(
            args.repo_id,
            "model",
            staging_dir / "identity_conversations.jsonl",
            "identity_conversations.jsonl",
            "Upload picoLLM identity dataset",
            token,
            args.dry_run,
        )
        upload_path(args.repo_id, "model", staging_dir / "report", "report", "Upload picoLLM reports", token, args.dry_run)
        upload_path(
            args.repo_id,
            "model",
            staging_dir / "run_manifest.json",
            "run_manifest.json",
            "Upload picoLLM run manifest",
            token,
            args.dry_run,
        )
        upload_path(
            args.repo_id,
            "model",
            staging_dir / MODEL_METADATA_FILENAME,
            MODEL_METADATA_FILENAME,
            "Upload picoLLM model metadata",
            token,
            args.dry_run,
        )
        upload_path(
            args.repo_id,
            "model",
            staging_dir / "README.md",
            "README.md",
            "Upload picoLLM model card",
            token,
            args.dry_run,
        )

        if args.post_upload_smoke and not args.dry_run:
            run_post_upload_smoke(
                repo_id=args.repo_id,
                prompt=args.post_upload_smoke_prompt,
                max_tokens=args.post_upload_smoke_max_tokens,
                device_type=args.post_upload_smoke_device_type,
                python_executable=args.python,
            )

        run_manifest = maybe_load_run_manifest(base_dir)
        if run_manifest is not None:
            print(f"Release provenance commit: {run_manifest.get('repo_commit', '')}")
        print(f"Preferred chat source: {metadata['preferred_chat_source']}")

    print(f"Model repo ready: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
