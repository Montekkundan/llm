#!/usr/bin/env python3
"""Shared helpers for picoLLM release and backup scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from restore_picollm_from_hf import (
    DEFAULT_BASE_DIR,
    choose_latest_step,
    choose_model_tag,
    verify_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_METADATA_FILENAME = "picollm_model_metadata.json"


def hf_run(command: list[str], dry_run: bool) -> None:
    print("Running:", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def require_hf_token(dry_run: bool, purpose: str) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if dry_run:
        return ""
    raise SystemExit(
        "Set HF_TOKEN before using the picoLLM Hugging Face release helpers "
        f"for {purpose}."
    )


def ensure_hf_cli(dry_run: bool) -> None:
    if dry_run:
        return
    if shutil.which("hf") is None:
        raise SystemExit("Missing `hf` CLI. Install it first, then rerun this release helper.")


def create_hf_repo(
    repo_id: str,
    repo_type: str,
    public: bool,
    token: str,
    dry_run: bool,
) -> None:
    ensure_hf_cli(dry_run=dry_run)
    visibility_flag = "--public" if public else "--private"
    command = [
        "hf",
        "repos",
        "create",
        repo_id,
        "--type",
        repo_type,
        visibility_flag,
        "--exist-ok",
    ]
    if token:
        command.extend(["--token", token])
    hf_run(command, dry_run=dry_run)


def upload_path(
    repo_id: str,
    repo_type: str,
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
        repo_type,
        "--commit-message",
        commit_message,
    ]
    if token:
        command.extend(["--token", token])
    hf_run(command, dry_run=dry_run)


def maybe_load_run_manifest(base_dir: Path) -> dict[str, object] | None:
    manifest_path = base_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def current_repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def summarize_checkpoint_root(checkpoints_root: Path) -> dict[str, object] | None:
    if not checkpoints_root.exists():
        return None
    model_tag_dir = choose_model_tag(checkpoints_root)
    model_path, meta_path, step = choose_latest_step(model_tag_dir)
    return {
        "root": checkpoints_root.name,
        "model_tag": model_tag_dir.name,
        "step": step,
        "latest_model": str(model_path.relative_to(checkpoints_root.parent)),
        "latest_meta": str(meta_path.relative_to(checkpoints_root.parent)),
    }


def preferred_chat_source(base_dir: Path) -> str:
    return "sft" if (base_dir / "chatsft_checkpoints").exists() else "base"


def build_model_metadata(base_dir: Path, repo_id: str) -> dict[str, object]:
    run_manifest = maybe_load_run_manifest(base_dir)
    repo_commit = (
        str(run_manifest.get("repo_commit", "")).strip()
        if run_manifest is not None
        else ""
    ) or current_repo_commit()
    base_summary = summarize_checkpoint_root(base_dir / "base_checkpoints")
    sft_summary = summarize_checkpoint_root(base_dir / "chatsft_checkpoints")
    identity_present = (base_dir / "identity_conversations.jsonl").exists()
    metadata: dict[str, object] = {
        "model_family_name": "picoLLM",
        "artifact_kind": "picoLLM native inference bundle",
        "repo_id": repo_id,
        "preferred_chat_source": preferred_chat_source(base_dir),
        "tokenizer_path": "tokenizer",
        "checkpoint_type": "picoLLM native checkpoints",
        "checkpoints": {
            "base": base_summary,
            "sft": sft_summary,
        },
        "compatible_picollm_commit_range": {
            "min_repo_commit": repo_commit,
            "max_repo_commit": repo_commit,
        },
        "restore_script": "scripts/restore_picollm_from_hf.py",
        "identity_dataset_path": "identity_conversations.jsonl" if identity_present else None,
        "run_manifest_path": "run_manifest.json" if run_manifest is not None else None,
        "known_limitations": [
            "This repo is for picoLLM-native restore and inference, not direct Transformers loading.",
            "Optimizer shards are intentionally excluded from the inference bundle.",
            "The model can hallucinate and should be treated as a small open model.",
        ],
    }
    if run_manifest is not None:
        metadata["run_manifest"] = {
            "identity_dataset_source": run_manifest.get("identity_dataset_path"),
            "detected_hardware": run_manifest.get("detected_hardware"),
            "speedrun_settings": run_manifest.get("speedrun_settings"),
            "hf_model_repo_id": run_manifest.get("hf_model_repo_id"),
            "hf_archive_repo_id": run_manifest.get("hf_archive_repo_id"),
        }
    return metadata


def render_model_readme(repo_id: str, metadata: dict[str, object]) -> str:
    checkpoints = metadata["checkpoints"]
    preferred_source = metadata["preferred_chat_source"]
    restore_command = (
        "git clone https://github.com/Montekkundan/llm\n"
        "cd llm\n"
        "uv sync --extra gpu\n"
        f"python scripts/restore_picollm_from_hf.py {repo_id}"
    )
    return f"""# picoLLM Runtime Artifacts

This repo contains the inference-focused artifact set produced by a `picollm/accelerated/speedrun.sh` run.

## Provenance

- picoLLM repo commit: `{metadata['compatible_picollm_commit_range']['min_repo_commit']}`
- Preferred chat source: `{preferred_source}`
- Base checkpoint: `{checkpoints['base']}`
- SFT checkpoint: `{checkpoints['sft']}`
- Run manifest: `{metadata['run_manifest_path']}`
- Identity dataset: `{metadata['identity_dataset_path']}`

## Intended Usage

- restore the bundle into `PICOLLM_BASE_DIR`
- run `python -m picollm.accelerated.chat.cli -i {preferred_source}`
- run `python -m picollm.accelerated.chat.web -i {preferred_source}`

## Known Limitations

- picoLLM-native checkpoints only; this repo is not Transformers-native
- optimizer shards are intentionally excluded here
- use the paired archive dataset repo if you need fuller resume-training artifacts
- the model can hallucinate and should be treated as a smaller open model

## Local Restore

```bash
{restore_command}
```

If you already downloaded the repo contents into a custom artifact directory:

```bash
export PICOLLM_BASE_DIR=$PWD/artifacts/picollm
python -m picollm.accelerated.chat.cli -i {preferred_source}
```
"""


def render_archive_readme(repo_id: str) -> str:
    return f"""# picoLLM Run Archive

This dataset repo stores the fuller artifact archive from a picoLLM accelerated run.

Intended contents:

- `tokenizer/`
- `base_checkpoints/`
- `chatsft_checkpoints/`
- `report/`
- `identity_conversations.jsonl`
- `run_manifest.json`

This repo is for preservation, inspection, teaching artifacts, and resume-oriented backups.
It may include optimizer shards and other training-only files that are intentionally excluded
from the inference-focused model repo.

For the runnable local restore path, prefer the paired Hugging Face model repo and:

```bash
python scripts/restore_picollm_from_hf.py {repo_id}
```
"""


def write_temp_text(contents: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    handle.write(contents)
    handle.flush()
    handle.close()
    return handle.name


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _stage_filtered_checkpoint_root(source_root: Path, target_root: Path) -> None:
    if not source_root.exists():
        return
    allowed_prefixes = ("model_", "meta_")
    for path in sorted(source_root.rglob("*")):
        if path.is_dir():
            continue
        if not path.name.startswith(allowed_prefixes):
            continue
        _copy_file(path, target_root / path.relative_to(source_root))


def stage_inference_bundle(base_dir: Path, repo_id: str, staging_dir: Path) -> dict[str, object]:
    _copy_tree(base_dir / "tokenizer", staging_dir / "tokenizer")
    _stage_filtered_checkpoint_root(
        base_dir / "base_checkpoints",
        staging_dir / "base_checkpoints",
    )
    _stage_filtered_checkpoint_root(
        base_dir / "chatsft_checkpoints",
        staging_dir / "chatsft_checkpoints",
    )
    if (base_dir / "identity_conversations.jsonl").exists():
        _copy_file(
            base_dir / "identity_conversations.jsonl",
            staging_dir / "identity_conversations.jsonl",
        )
    if (base_dir / "report").exists():
        _copy_tree(base_dir / "report", staging_dir / "report")
    if (base_dir / "run_manifest.json").exists():
        _copy_file(base_dir / "run_manifest.json", staging_dir / "run_manifest.json")

    metadata = build_model_metadata(base_dir, repo_id)
    metadata_path = staging_dir / MODEL_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme_path = staging_dir / "README.md"
    readme_path.write_text(render_model_readme(repo_id, metadata), encoding="utf-8")
    verify_layout(staging_dir)
    return metadata

