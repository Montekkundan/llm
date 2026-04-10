#!/usr/bin/env python3
"""Upload the fuller picoLLM run archive to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from picollm_release_lib import (
    DEFAULT_BASE_DIR,
    create_hf_repo,
    render_archive_readme,
    require_hf_token,
    upload_path,
    write_temp_text,
)
from restore_picollm_from_hf import verify_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the fuller picoLLM run archive to a Hugging Face dataset repo.")
    parser.add_argument("repo_id", type=str, help="Destination Hugging Face dataset repo id")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Local PICOLLM_BASE_DIR to publish")
    parser.add_argument("--public", action="store_true", help="Create or update the destination repo as public instead of private")
    parser.add_argument("--dry-run", action="store_true", help="Verify layout and print the upload plan without calling the Hugging Face CLI")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    verify_layout(base_dir)
    token = require_hf_token(dry_run=args.dry_run, purpose="archive dataset upload")
    create_hf_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        public=args.public,
        token=token,
        dry_run=args.dry_run,
    )

    upload_path(args.repo_id, "dataset", base_dir / "tokenizer", "tokenizer", "Upload picoLLM tokenizer archive", token, args.dry_run)
    upload_path(
        args.repo_id,
        "dataset",
        base_dir / "base_checkpoints",
        "base_checkpoints",
        "Upload picoLLM base checkpoint archive",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        "dataset",
        base_dir / "chatsft_checkpoints",
        "chatsft_checkpoints",
        "Upload picoLLM SFT checkpoint archive",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        "dataset",
        base_dir / "identity_conversations.jsonl",
        "identity_conversations.jsonl",
        "Upload picoLLM identity dataset archive",
        token,
        args.dry_run,
    )
    upload_path(args.repo_id, "dataset", base_dir / "report", "report", "Upload picoLLM report archive", token, args.dry_run)
    upload_path(args.repo_id, "dataset", base_dir / "run_manifest.json", "run_manifest.json", "Upload picoLLM run manifest", token, args.dry_run)

    readme_path = write_temp_text(render_archive_readme(args.repo_id))
    try:
        upload_path(args.repo_id, "dataset", Path(readme_path), "README.md", "Upload picoLLM archive README", token, args.dry_run)
    finally:
        os.unlink(readme_path)

    print(f"Archive dataset ready: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
