#!/usr/bin/env python3
"""Write a machine-readable run manifest for the current picoLLM artifact directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from restore_picollm_from_hf import choose_latest_step, choose_model_tag, DEFAULT_BASE_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def checkpoint_summary(checkpoints_root: Path) -> dict[str, object] | None:
    if not checkpoints_root.exists():
        return None
    model_tag_dir = choose_model_tag(checkpoints_root)
    model_path, meta_path, step = choose_latest_step(model_tag_dir)
    return {
        "model_tag": model_tag_dir.name,
        "step": step,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a machine-readable picoLLM run manifest.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="PICOLLM_BASE_DIR for the current run")
    parser.add_argument("--identity-source", type=str, required=True, help="Identity dataset source path used for SFT")
    parser.add_argument("--output", type=Path, default=None, help="Manifest output path. Defaults to <base-dir>/run_manifest.json")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output = args.output.resolve() if args.output is not None else base_dir / "run_manifest.json"

    manifest = {
        "repo_commit": git_commit(),
        "torch_version": torch.__version__,
        "base_dir": str(base_dir),
        "detected_hardware": os.environ.get("PICOLLM_HARDWARE_SUMMARY", ""),
        "speedrun_settings": os.environ.get("PICOLLM_SETTINGS_SUMMARY", ""),
        "identity_dataset_path": args.identity_source,
        "hf_model_repo_id": os.environ.get("HF_UPLOAD_REPO_ID", ""),
        "hf_archive_repo_id": os.environ.get("HF_ARCHIVE_REPO_ID", ""),
        "checkpoints": {
            "base": checkpoint_summary(base_dir / "base_checkpoints"),
            "sft": checkpoint_summary(base_dir / "chatsft_checkpoints"),
        },
    }

    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run manifest to {output}")


if __name__ == "__main__":
    main()
