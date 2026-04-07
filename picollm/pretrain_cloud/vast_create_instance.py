from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vast_common import request


DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Vast.ai instance for picoLLM training.")
    parser.add_argument("--offer-id", type=int, required=True)
    parser.add_argument("--label", default="picollm-train")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--runtype", default="ssh_direct")
    parser.add_argument("--hf-token", default=None, help="Optional HF token to inject into the instance env.")
    parser.add_argument("--wandb-api-key", default=None, help="Optional W&B API key to inject into the instance env.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/workspace to inject into the instance env.")
    parser.add_argument("--repo-url", default="https://github.com/montekkundan/llm.git")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--onstart-file", default=None, help="Optional shell script to run on instance startup.")
    args = parser.parse_args()

    env = {}
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
    if args.wandb_api_key:
        env["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity

    onstart = None
    if args.onstart_file:
        onstart = Path(args.onstart_file).read_text(encoding="utf-8")

    payload = {
        "image": args.image,
        "label": args.label,
        "disk": args.disk_gb,
        "runtype": args.runtype,
        "env": env,
        "cancel_unavail": True,
    }
    if onstart:
        payload["onstart"] = onstart

    result = request("PUT", f"/asks/{args.offer_id}/", payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
