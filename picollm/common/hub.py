from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    commit_message: str = "Upload model artifacts",
    private: bool = False,
) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def download_snapshot(repo_id: str, revision: str | None = None, local_dir: str | Path | None = None) -> str:
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir) if local_dir else None)
