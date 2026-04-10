#!/usr/bin/env python3
"""Download a picoLLM Hugging Face model repo, verify its layout, and optionally smoke-test it."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = REPO_ROOT / "artifacts" / "picollm"


def ensure_hf_cli() -> None:
    if shutil.which("hf") is None:
        raise SystemExit("Missing `hf` CLI. Install it first, then rerun this restore helper.")


def candidate_checkpoint_dirs(checkpoints_dir: Path) -> list[Path]:
    return [path for path in checkpoints_dir.iterdir() if path.is_dir()]


def choose_model_tag(checkpoints_dir: Path) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for path in candidate_checkpoint_dirs(checkpoints_dir):
        match = re.fullmatch(r"d(\d+)", path.name)
        depth = int(match.group(1)) if match else -1
        candidates.append((depth, path.stat().st_mtime, path))
    if not candidates:
        raise SystemExit(f"No checkpoint directories found in {checkpoints_dir}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def choose_latest_step(checkpoint_dir: Path) -> tuple[Path, Path, int]:
    model_paths = sorted(checkpoint_dir.glob("model_*.pt"))
    if not model_paths:
        raise SystemExit(f"No model checkpoints found in {checkpoint_dir}")
    latest_model = model_paths[-1]
    match = re.fullmatch(r"model_(\d+)\.pt", latest_model.name)
    if match is None:
        raise SystemExit(f"Unexpected checkpoint filename: {latest_model.name}")
    step = int(match.group(1))
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata file for checkpoint step {step}: {meta_path}")
    return latest_model, meta_path, step


def verify_layout(base_dir: Path) -> None:
    tokenizer_dir = base_dir / "tokenizer"
    required_tokenizer_files = [
        tokenizer_dir / "tokenizer.pkl",
        tokenizer_dir / "token_bytes.pt",
    ]
    missing = [path for path in required_tokenizer_files if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing tokenizer files: {formatted}")

    for checkpoint_root_name in ("base_checkpoints", "chatsft_checkpoints"):
        checkpoint_root = base_dir / checkpoint_root_name
        if not checkpoint_root.exists():
            raise SystemExit(f"Missing checkpoint root: {checkpoint_root}")
        model_tag_dir = choose_model_tag(checkpoint_root)
        latest_model, meta_path, step = choose_latest_step(model_tag_dir)
        print(
            f"{checkpoint_root_name}: ok (model_tag={model_tag_dir.name}, step={step}, "
            f"model={latest_model.name}, meta={meta_path.name})"
        )

    identity_path = base_dir / "identity_conversations.jsonl"
    if identity_path.exists():
        print(f"identity: present ({identity_path})")
    else:
        print("identity: not present in restored repo")


def run_hf_download(repo_id: str, base_dir: Path) -> None:
    ensure_hf_cli()
    command = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "model",
        "--local-dir",
        str(base_dir),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def run_cli_smoke(
    base_dir: Path,
    source: str,
    prompt: str,
    max_tokens: int,
    device_type: str,
    python_executable: str,
) -> None:
    env = os.environ.copy()
    env["PICOLLM_BASE_DIR"] = str(base_dir)
    command = [
        python_executable,
        "-m",
        "picollm.accelerated.chat.cli",
        "-i",
        source,
        "-p",
        prompt,
        "--max-tokens",
        str(max_tokens),
    ]
    if device_type:
        command.extend(["--device-type", device_type])
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a picoLLM model repo from Hugging Face and optionally smoke-test it.")
    parser.add_argument("repo_id", type=str, help="Hugging Face model repo id, for example user/picollm-run")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Local PICOLLM_BASE_DIR target")
    parser.add_argument("--skip-download", action="store_true", help="Skip hf download and only verify an existing local base dir")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the post-restore CLI smoke test")
    parser.add_argument("--source", type=str, default="sft", choices=["base", "sft"], help="Checkpoint source to smoke-test")
    parser.add_argument("--prompt", type=str, default="Who are you?", help="Prompt used for the optional smoke test")
    parser.add_argument("--max-tokens", type=int, default=48, help="Maximum tokens for the optional smoke test")
    parser.add_argument("--device-type", type=str, default="", choices=["", "cpu", "mps", "cuda"], help="Device type for the optional smoke test")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for the optional smoke test")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        run_hf_download(args.repo_id, base_dir)

    verify_layout(base_dir)

    if not args.skip_smoke:
        run_cli_smoke(
            base_dir=base_dir,
            source=args.source,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            device_type=args.device_type,
            python_executable=args.python,
        )

    print(f"PICOLLM_BASE_DIR={base_dir}")


if __name__ == "__main__":
    main()
