#!/usr/bin/env python3
"""Periodically sync the current picoLLM archive bundle to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_relevant_files(base_dir: Path):
    roots = [
        base_dir / "base_checkpoints",
        base_dir / "chatsft_checkpoints",
        base_dir / "tokenizer",
        base_dir / "report",
    ]
    standalone = [
        base_dir / "identity_conversations.jsonl",
        base_dir / "run_manifest.json",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith((".tmp", ".lock")):
                continue
            yield path
    for path in standalone:
        if path.is_file():
            yield path


def artifact_signature(base_dir: Path) -> tuple[str | None, float | None]:
    files = list(iter_relevant_files(base_dir))
    if not files:
        return None, None
    lines = []
    newest_mtime = 0.0
    for path in files:
        stat = path.stat()
        newest_mtime = max(newest_mtime, stat.st_mtime)
        lines.append(
            f"{path.relative_to(base_dir)}\t{stat.st_size}\t{stat.st_mtime_ns}"
        )
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    newest_age_seconds = max(0.0, time.time() - newest_mtime)
    return digest, newest_age_seconds


def upload_archive(
    repo_id: str,
    base_dir: Path,
    public: bool,
    python_executable: str,
) -> None:
    command = [
        python_executable,
        str(REPO_ROOT / "scripts" / "upload_picollm_archive_to_hf.py"),
        repo_id,
        "--base-dir",
        str(base_dir),
    ]
    if public:
        command.append("--public")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodically sync picoLLM archive artifacts to a Hugging Face dataset repo.")
    parser.add_argument("repo_id", type=str, help="Destination Hugging Face dataset repo id")
    parser.add_argument("--base-dir", type=Path, required=True, help="PICOLLM_BASE_DIR to watch and sync")
    parser.add_argument("--interval-seconds", type=int, default=900, help="Polling interval between archive sync attempts")
    parser.add_argument("--min-file-age-seconds", type=int, default=180, help="Only sync when the newest relevant artifact is at least this old")
    parser.add_argument("--public", action="store_true", help="Create or update the destination repo as public instead of private")
    parser.add_argument("--once", action="store_true", help="Run at most one sync attempt and exit")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to call the archive upload helper")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    last_uploaded_signature: str | None = None

    while True:
        signature, newest_age_seconds = artifact_signature(base_dir)
        if signature is None:
            print(f"No archive artifacts found yet under {base_dir}")
        elif signature == last_uploaded_signature:
            print("Archive sync: no artifact changes detected")
        elif newest_age_seconds is not None and newest_age_seconds < args.min_file_age_seconds:
            print(
                "Archive sync: waiting for files to settle "
                f"(newest age {newest_age_seconds:.1f}s < {args.min_file_age_seconds}s)"
            )
        else:
            upload_archive(
                repo_id=args.repo_id,
                base_dir=base_dir,
                public=args.public,
                python_executable=args.python,
            )
            last_uploaded_signature = signature
            print("Archive sync: upload complete")

        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
