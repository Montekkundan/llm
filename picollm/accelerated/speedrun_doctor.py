import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from picollm.accelerated.common import get_base_dir
from picollm.accelerated.speedrun_config import _detect_config


def _parse_bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _required_free_vram_gib(device_batch_size: int, enable_fp8: bool) -> float:
    if enable_fp8:
        return {
            16: 70.0,
            8: 45.0,
            4: 30.0,
            2: 22.0,
        }.get(device_batch_size, 0.0)
    return {
        8: 70.0,
        4: 45.0,
        2: 30.0,
    }.get(device_batch_size, 0.0)


def _check_artifact_dir(base_dir: Path) -> dict[str, object]:
    base_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=base_dir, delete=True):
        pass
    total, used, free = shutil.disk_usage(base_dir)
    return {
        "base_dir": str(base_dir),
        "disk_free_gib": round(free / (1024 ** 3), 1),
    }


def _check_hf_upload_state() -> dict[str, object]:
    model_repo = os.environ.get("HF_UPLOAD_REPO_ID", "").strip()
    archive_repo = os.environ.get("HF_ARCHIVE_REPO_ID", "").strip()
    uploads_configured = bool(model_repo or archive_repo)
    hf_cli_present = shutil.which("hf") is not None
    hf_token_present = bool(os.environ.get("HF_TOKEN", "").strip())
    return {
        "uploads_configured": uploads_configured,
        "hf_cli_present": hf_cli_present,
        "hf_token_present": hf_token_present,
        "model_repo": model_repo,
        "archive_repo": archive_repo,
    }


def _check_vram(config: dict[str, object]) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "min_free_vram_gib": 0.0,
            "required_free_vram_gib": 0.0,
            "passes": False,
        }

    free_values = []
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_values.append(free_bytes / (1024 ** 3))

    min_free_vram_gib = round(min(free_values), 1)
    required_free_vram_gib = _required_free_vram_gib(
        int(config["PICOLLM_DEVICE_BATCH_SIZE"]),
        bool(int(config["PICOLLM_ENABLE_FP8"])),
    )
    return {
        "min_free_vram_gib": min_free_vram_gib,
        "required_free_vram_gib": required_free_vram_gib,
        "passes": min_free_vram_gib >= required_free_vram_gib,
    }


def run_doctor(min_free_disk_gib: float) -> tuple[dict[str, object], list[str]]:
    config = _detect_config()
    checks: dict[str, object] = {
        "speedrun_supported": bool(int(config["PICOLLM_SUPPORTED"])),
        "hardware_summary": config["PICOLLM_HARDWARE_SUMMARY"],
        "settings_summary": config.get("PICOLLM_SETTINGS_SUMMARY", ""),
    }
    failures: list[str] = []

    if not checks["speedrun_supported"]:
        failures.append(str(config["PICOLLM_UNSUPPORTED_REASON"]))

    artifact_check = _check_artifact_dir(Path(get_base_dir()))
    checks["artifact_dir"] = artifact_check
    if artifact_check["disk_free_gib"] < min_free_disk_gib:
        failures.append(
            f"Artifact dir only has {artifact_check['disk_free_gib']:.1f} GiB free; require at least {min_free_disk_gib:.1f} GiB"
        )

    hf_check = _check_hf_upload_state()
    checks["hf"] = hf_check
    if hf_check["uploads_configured"] and not hf_check["hf_cli_present"]:
        failures.append("HF uploads are configured but the `hf` CLI is not installed or not on PATH")
    if hf_check["uploads_configured"] and not hf_check["hf_token_present"]:
        failures.append("HF uploads are configured but HF_TOKEN is not set")

    vram_check = _check_vram(config)
    checks["vram"] = vram_check
    if checks["speedrun_supported"] and not vram_check["passes"]:
        failures.append(
            f"Visible free VRAM is {vram_check['min_free_vram_gib']:.1f} GiB but the chosen config expects at least "
            f"{vram_check['required_free_vram_gib']:.1f} GiB free per GPU"
        )

    return checks, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight doctor for the picoLLM accelerated speedrun path")
    parser.add_argument("--min-free-disk-gib", type=float, default=50.0, help="Minimum free disk space required in PICOLLM_BASE_DIR")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--allow-unsupported", action="store_true", help="Do not fail just because CUDA speedrun is unsupported on this machine")
    args = parser.parse_args()

    checks, failures = run_doctor(min_free_disk_gib=args.min_free_disk_gib)
    if args.allow_unsupported:
        failures = [failure for failure in failures if "speedrun.sh currently expects" not in failure]

    if args.format == "json":
        print(json.dumps({"checks": checks, "failures": failures}, indent=2))
    else:
        print(f"Doctor hardware: {checks['hardware_summary']}")
        if checks["settings_summary"]:
            print(f"Doctor settings: {checks['settings_summary']}")
        artifact_dir = checks["artifact_dir"]
        print(f"Doctor artifact dir: {artifact_dir['base_dir']} ({artifact_dir['disk_free_gib']:.1f} GiB free)")
        hf = checks["hf"]
        print(
            "Doctor HF: "
            f"uploads_configured={hf['uploads_configured']} hf_cli_present={hf['hf_cli_present']} hf_token_present={hf['hf_token_present']}"
        )
        vram = checks["vram"]
        if torch.cuda.is_available():
            print(
                "Doctor VRAM: "
                f"free={vram['min_free_vram_gib']:.1f} GiB required={vram['required_free_vram_gib']:.1f} GiB"
            )
        if failures:
            print("Doctor status: FAIL")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("Doctor status: OK")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
