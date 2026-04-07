from __future__ import annotations

import argparse
import json

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Destroy a Vast.ai instance permanently.")
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.yes:
        answer = input(
            f"Destroy Vast.ai instance {args.instance_id}? This is irreversible and deletes remote data. [y/N] "
        ).strip()
        if answer.lower() not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    result = request("DELETE", f"/instances/{args.instance_id}/")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
