#!/usr/bin/env python3
"""Create a paired picoLLM model release and archive release on Hugging Face."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from picollm_release_lib import DEFAULT_BASE_DIR, summarize_checkpoint_root
from restore_picollm_from_hf import verify_layout


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_release_stem(base_dir: Path, release_name: str) -> str:
    checkpoint = summarize_checkpoint_root(base_dir / "chatsft_checkpoints")
    source = "sft"
    if checkpoint is None:
        checkpoint = summarize_checkpoint_root(base_dir / "base_checkpoints")
        source = "base"
    if checkpoint is None:
        raise SystemExit("Could not derive a release stem because no checkpoints were found.")
    release_label = release_name or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"picollm-{release_label}-{checkpoint['model_tag']}-{source}{checkpoint['step']:06d}"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish both the inference bundle and the fuller archive bundle for picoLLM.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Local PICOLLM_BASE_DIR to publish")
    parser.add_argument("--namespace", type=str, default="", help="HF namespace used to derive repo ids when explicit ids are not provided")
    parser.add_argument("--release-name", type=str, default="", help="Stable release label used in derived repo ids, for example april-h200-run")
    parser.add_argument("--model-repo-id", type=str, default="", help="Explicit HF model repo id for the inference bundle")
    parser.add_argument("--archive-repo-id", type=str, default="", help="Explicit HF dataset repo id for the fuller archive bundle")
    parser.add_argument("--latest-repo-id", type=str, default="", help="Optional stable model repo id that should mirror the latest inference bundle")
    parser.add_argument("--public", action="store_true", help="Create or update destination repos as public")
    parser.add_argument("--dry-run", action="store_true", help="Print the full release plan without calling the Hugging Face CLI")
    parser.add_argument("--post-upload-smoke", action="store_true", help="Run the model restore smoke test after uploading the versioned inference repo")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for nested helper calls")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    verify_layout(base_dir)

    stem = default_release_stem(base_dir=base_dir, release_name=args.release_name)
    namespace = args.namespace.strip()
    if not args.model_repo_id and not namespace:
        raise SystemExit("Provide either --model-repo-id or --namespace.")
    if not args.archive_repo_id and not namespace:
        raise SystemExit("Provide either --archive-repo-id or --namespace.")

    model_repo_id = args.model_repo_id or f"{namespace}/{stem}"
    archive_repo_id = args.archive_repo_id or f"{namespace}/{stem}-archive"
    latest_repo_id = args.latest_repo_id.strip()

    print(f"Release stem: {stem}")
    print(f"Inference repo: {model_repo_id}")
    print(f"Archive repo: {archive_repo_id}")
    if latest_repo_id:
        print(f"Latest alias repo: {latest_repo_id}")

    visibility_flag = ["--public"] if args.public else []
    dry_run_flag = ["--dry-run"] if args.dry_run else []
    smoke_flag = ["--post-upload-smoke"] if args.post_upload_smoke else []

    run_command(
        [
            args.python,
            str(REPO_ROOT / "scripts" / "upload_picollm_model_to_hf.py"),
            model_repo_id,
            "--base-dir",
            str(base_dir),
            *visibility_flag,
            *dry_run_flag,
            *smoke_flag,
        ]
    )
    run_command(
        [
            args.python,
            str(REPO_ROOT / "scripts" / "upload_picollm_archive_to_hf.py"),
            archive_repo_id,
            "--base-dir",
            str(base_dir),
            *visibility_flag,
            *dry_run_flag,
        ]
    )

    if latest_repo_id:
        run_command(
            [
                args.python,
                str(REPO_ROOT / "scripts" / "upload_picollm_model_to_hf.py"),
                latest_repo_id,
                "--base-dir",
                str(base_dir),
                *visibility_flag,
                *dry_run_flag,
            ]
        )

    print("Release helper finished.")


if __name__ == "__main__":
    main()
