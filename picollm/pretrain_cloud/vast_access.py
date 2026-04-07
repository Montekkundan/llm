from __future__ import annotations

import argparse

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print SSH/SCP/rsync commands for a Vast.ai instance. This helper does not execute them."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new-contract", dest="contract_id", type=int, help="The new_contract value returned by Vast.")
    group.add_argument("--instance-id", dest="contract_id", type=int, help="Backward-compatible alias for --new-contract.")
    parser.add_argument("--local-dir", default="artifacts/picollm/pretrain-run")
    parser.add_argument("--remote-dir", default="/workspace/llm/artifacts/picollm/pretrain-run")
    parser.add_argument("--ssh-user", default="root")
    args = parser.parse_args()

    result = request("GET", f"/instances/{args.contract_id}/")
    instance = result.get("instances", {})
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if not host or not port:
        raise SystemExit("Instance does not expose SSH details yet.")

    ssh_target = f"{args.ssh_user}@{host}"
    print("This helper only prints the commands you should run next. It does not SSH or copy files by itself.")
    print()
    print("Run this SSH command if you want to connect:")
    print(f"ssh -p {port} {ssh_target}")
    print()
    print("Run this command on your local machine to copy the checkpoint from Vast to local:")
    print(f"scp -P {port} -r {ssh_target}:{args.remote_dir} {args.local_dir}")
    print()
    print("Or run this rsync command on your local machine:")
    print(f"rsync -avz -e 'ssh -p {port}' {ssh_target}:{args.remote_dir}/ {args.local_dir}/")


if __name__ == "__main__":
    main()
