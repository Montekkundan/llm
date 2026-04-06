from __future__ import annotations

import argparse
import json

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Vast.ai offers for picoLLM training.")
    parser.add_argument("--gpu-name", default="RTX 4090")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-ram-gb", type=int, default=24)
    parser.add_argument("--reliability", type=float, default=0.99)
    parser.add_argument("--type", default="ondemand", choices=["ondemand", "bid"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    payload = {
        "gpu_name": {"in": [args.gpu_name]},
        "num_gpus": {"gte": args.num_gpus},
        "gpu_ram": {"gte": args.gpu_ram_gb * 1000},
        "reliability": {"gte": args.reliability},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "type": args.type,
        "limit": args.limit,
    }
    result = request("POST", "/bundles/", payload)
    offers = result.get("offers", [])
    if not offers:
        raise SystemExit("No offers matched the search.")
    rows = [
        {
            "id": offer.get("id"),
            "gpu_name": offer.get("gpu_name"),
            "num_gpus": offer.get("num_gpus"),
            "gpu_ram": offer.get("gpu_ram"),
            "dph_total": offer.get("dph_total"),
            "reliability": offer.get("reliability2", offer.get("reliability")),
            "cpu_cores_effective": offer.get("cpu_cores_effective"),
            "dlperf": offer.get("dlperf"),
        }
        for offer in offers
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
