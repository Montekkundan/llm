#!/usr/bin/env python3
"""Upload the runnable picoLLM artifact set to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from restore_picollm_from_hf import DEFAULT_BASE_DIR, verify_layout


REPO_ROOT = Path(__file__).resolve().parents[1]


def hf_run(command: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def require_hf_token(dry_run: bool) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token and not dry_run:
        raise SystemExit("Set HF_TOKEN before uploading picoLLM model artifacts to the Hugging Face Hub.")
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
        "model",
        "--commit-message",
        commit_message,
    ]
    if token:
        command.extend(["--token", token])
    hf_run(command, dry_run=dry_run)


def render_model_readme(repo_id: str) -> str:
    return f"""# picoLLM Runtime Artifacts

This repo contains the runnable artifact set produced by a `picollm/accelerated/speedrun.sh` run.

Included:

- `tokenizer/`
- `base_checkpoints/`
- `chatsft_checkpoints/`
- `report/` if present
- `identity_conversations.jsonl` if present

Restore locally:

```bash
git clone https://github.com/Montekkundan/llm
cd llm
uv sync --extra gpu
python scripts/restore_picollm_from_hf.py {repo_id}
```

If you already downloaded the repo contents into a custom artifact directory:

```bash
export PICOLLM_BASE_DIR=$PWD/artifacts/picollm
python -m picollm.accelerated.chat.cli -i sft
```

These files are native to picoLLM and are meant to be loaded through the repo's checkpoint manager rather than standard Transformers loaders.
"""


def write_temp_readme(repo_id: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    handle.write(render_model_readme(repo_id))
    handle.flush()
    handle.close()
    return handle.name


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
    token = require_hf_token(dry_run=args.dry_run)

    visibility_flag = "--public" if args.public else "--private"
    create_command = [
        "hf",
        "repos",
        "create",
        args.repo_id,
        "--type",
        "model",
        visibility_flag,
        "--exist-ok",
    ]
    if token:
        create_command.extend(["--token", token])
    hf_run(create_command, dry_run=args.dry_run)

    upload_path(args.repo_id, base_dir / "tokenizer", "tokenizer", "Upload picoLLM tokenizer", token, args.dry_run)
    upload_path(
        args.repo_id,
        base_dir / "base_checkpoints",
        "base_checkpoints",
        "Upload picoLLM base checkpoints",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        base_dir / "chatsft_checkpoints",
        "chatsft_checkpoints",
        "Upload picoLLM SFT checkpoints",
        token,
        args.dry_run,
    )
    upload_path(
        args.repo_id,
        base_dir / "identity_conversations.jsonl",
        "identity_conversations.jsonl",
        "Upload picoLLM identity dataset",
        token,
        args.dry_run,
    )
    upload_path(args.repo_id, base_dir / "report", "report", "Upload picoLLM reports", token, args.dry_run)

    readme_path = write_temp_readme(args.repo_id)
    try:
        upload_path(args.repo_id, Path(readme_path), "README.md", "Upload picoLLM restore instructions", token, args.dry_run)
    finally:
        os.unlink(readme_path)

    if args.post_upload_smoke and not args.dry_run:
        run_post_upload_smoke(
            repo_id=args.repo_id,
            prompt=args.post_upload_smoke_prompt,
            max_tokens=args.post_upload_smoke_max_tokens,
            device_type=args.post_upload_smoke_device_type,
            python_executable=args.python,
        )

    print(f"Model repo ready: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
