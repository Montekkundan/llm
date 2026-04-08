from __future__ import annotations

import argparse
import json

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Vast.ai offers for picoLLM training.")
    parser.add_argument("--gpu-name", default="H100 SXM")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-ram-gb", type=int, default=80)
    parser.add_argument("--min-disk-gb", type=int, default=200)
    parser.add_argument("--reliability", type=float, default=0.995)
    parser.add_argument("--type", default="ondemand", choices=["ondemand", "bid"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    payload = {
        "gpu_name": {"in": [args.gpu_name]},
        "num_gpus": {"gte": args.num_gpus},
        "gpu_ram": {"gte": args.gpu_ram_gb * 1000},
        "disk_space": {"gte": args.min_disk_gb},
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
            "price_per_hour": offer.get("dph_total"),
            "discounted_price_per_hour": offer.get("discounted_hourly"),
            "dph_total": offer.get("dph_total"),
            "reliability": offer.get("reliability2", offer.get("reliability")),
            "disk_space": offer.get("disk_space"),
            "disk_bw": offer.get("disk_bw"),
            "storage_cost_per_month": offer.get("storage_cost"),
            "cpu_cores_effective": offer.get("cpu_cores_effective"),
            "inet_down": offer.get("inet_down"),
            "dlperf": offer.get("dlperf"),
        }
        for offer in offers
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
