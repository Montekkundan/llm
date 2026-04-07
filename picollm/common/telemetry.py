from __future__ import annotations

import netrc
import os
from pathlib import Path


def trainer_report_to(value: str) -> list[str]:
    if value == "none":
        return []
    return [value]


def has_wandb_auth() -> bool:
    if os.getenv("WANDB_API_KEY"):
        return True

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False

    try:
        credentials = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        return False

    return bool(credentials and credentials[2])


def ensure_reporter_ready(report_to: str) -> None:
    if report_to != "wandb":
        return

    if has_wandb_auth():
        return

    raise SystemExit(
        "You passed --report-to wandb, but no Weights & Biases login was found.\n"
        "Fix one of these before rerunning:\n"
        "  1. export WANDB_API_KEY=\"...\"\n"
        "  2. run: wandb login"
    )
