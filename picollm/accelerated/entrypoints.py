from __future__ import annotations

import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script(script_name: str) -> int:
    runpy.run_path(str(REPO_ROOT / "scripts" / script_name), run_name="__main__")
    return 0


def restore_from_hf() -> int:
    return _run_script("restore_picollm_from_hf.py")


def upload_model_to_hf() -> int:
    return _run_script("upload_picollm_model_to_hf.py")


def upload_archive_to_hf() -> int:
    return _run_script("upload_picollm_archive_to_hf.py")


def release_to_hf() -> int:
    return _run_script("release_picollm_to_hf.py")


def run_local_checks() -> int:
    return _run_script("run_picollm_local_checks.py")


def print_env() -> int:
    return _run_script("print_picollm_env.py")


def smoke_model_repo() -> int:
    return _run_script("smoke_picollm_model_repo.py")
