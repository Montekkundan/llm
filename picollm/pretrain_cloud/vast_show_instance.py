from __future__ import annotations

import argparse
import json

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a Vast.ai instance for picoLLM training.")
    parser.add_argument("--instance-id", type=int, required=True)
    args = parser.parse_args()

    result = request("GET", f"/instances/{args.instance_id}/")
    instance = result.get("instances", {})
    payload = {
        "id": instance.get("id"),
        "status": instance.get("actual_status"),
        "label": instance.get("label"),
        "gpu_name": instance.get("gpu_name"),
        "num_gpus": instance.get("num_gpus"),
        "ssh_host": instance.get("ssh_host"),
        "ssh_port": instance.get("ssh_port"),
        "public_ipaddr": instance.get("public_ipaddr"),
        "dph_total": instance.get("dph_total"),
        "ports": instance.get("ports"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
