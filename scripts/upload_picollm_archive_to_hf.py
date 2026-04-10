#!/usr/bin/env python3
"""Upload the fuller picoLLM run archive to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from restore_picollm_from_hf import DEFAULT_BASE_DIR, verify_layout


def hf_run(command: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def require_hf_token(dry_run: bool) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token and not dry_run:
        raise SystemExit("Set HF_TOKEN before uploading picoLLM archive artifacts to the Hugging Face Hub.")
    return token


def upload_path(
    repo_id: str,
    local_path: Path,
    remote_path: str,
    commit_message: str,
    token: str,
    dry_run: bool,
) -> None:
    if not local_path.exists():
        print(f"Skipping missing path: {local_path}")
        return
    command = [
        "hf",
        "upload",
        repo_id,
        str(local_path),
        remote_path,
        "--repo-type",
        "dataset",
        "--commit-message",
        commit_message,
    ]
    if token:
        command.extend(["--token", token])
    hf_run(command, dry_run=dry_run)


def render_archive_readme(repo_id: str) -> str:
    return f"""# picoLLM Run Archive

This dataset repo stores the fuller artifact archive from a picoLLM accelerated run.

Intended contents:

- `tokenizer/`
- `base_checkpoints/`
- `chatsft_checkpoints/`
- `report/`
- `identity_conversations.jsonl`
- `run_manifest.json` if present

This repo is for preservation, inspection, teaching artifacts, and exact resume-oriented archives.

For the runnable local restore path, prefer the paired Hugging Face model repo and:

```bash
python scripts/restore_picollm_from_hf.py {repo_id}
```
"""


def write_temp_readme(repo_id: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    handle.write(render_archive_readme(repo_id))
    handle.flush()
    handle.close()
    return handle.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the fuller picoLLM run archive to a Hugging Face dataset repo.")
    parser.add_argument("repo_id", type=str, help="Destination Hugging Face dataset repo id")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Local PICOLLM_BASE_DIR to publish")
    parser.add_argument("--public", action="store_true", help="Create or update the destination repo as public instead of private")
    parser.add_argument("--dry-run", action="store_true", help="Verify layout and print the upload plan without calling the Hugging Face CLI")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    verify_layout(base_dir)
    token = require_hf_token(dry_run=args.dry_run)

    visibility_flag = "--public" if args.public else "--private"
    create_command = [
        "hf",
        "repos",
        "create",
        args.repo_id,
        "--type",
        "dataset",
        visibility_flag,
        "--exist-ok",
    ]
    if token:
        create_command.extend(["--token", token])
    hf_run(create_command, dry_run=args.dry_run)

    upload_path(args.repo_id, base_dir / "tokenizer", "tokenizer", "Upload picoLLM tokenizer archive", token, args.dry_run)
    upload_path(
        args.repo_id,
        base_dir / "base_checkpoints",
        "base_checkpoints",
        "Upload picoLLM base checkpoint archive",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        base_dir / "chatsft_checkpoints",
        "chatsft_checkpoints",
        "Upload picoLLM SFT checkpoint archive",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        base_dir / "identity_conversations.jsonl",
        "identity_conversations.jsonl",
        "Upload picoLLM identity dataset archive",
        token,
        args.dry_run,
    )
    upload_path(args.repo_id, base_dir / "report", "report", "Upload picoLLM report archive", token, args.dry_run)
    upload_path(args.repo_id, base_dir / "run_manifest.json", "run_manifest.json", "Upload picoLLM run manifest", token, args.dry_run)

    readme_path = write_temp_readme(args.repo_id)
    try:
        upload_path(args.repo_id, Path(readme_path), "README.md", "Upload picoLLM archive README", token, args.dry_run)
    finally:
        os.unlink(readme_path)

    print(f"Archive dataset ready: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
