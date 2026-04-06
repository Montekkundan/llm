from __future__ import annotations

import argparse

from .vast_common import request


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SSH/SCP/rsync commands for a Vast.ai instance.")
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--local-dir", default="artifacts/picollm/pretrain-run")
    parser.add_argument("--remote-dir", default="/workspace/llm/artifacts/picollm/pretrain-run")
    parser.add_argument("--ssh-user", default="root")
    args = parser.parse_args()

    result = request("GET", f"/instances/{args.instance_id}/")
    instance = result.get("instances", {})
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if not host or not port:
        raise SystemExit("Instance does not expose SSH details yet.")

    ssh_target = f"{args.ssh_user}@{host}"
    print("SSH:")
    print(f"ssh -p {port} {ssh_target}")
    print()
    print("Copy checkpoint from Vast to local:")
    print(f"scp -P {port} -r {ssh_target}:{args.remote_dir} {args.local_dir}")
    print()
    print("Rsync checkpoint from Vast to local:")
    print(f"rsync -avz -e 'ssh -p {port}' {ssh_target}:{args.remote_dir}/ {args.local_dir}/")


if __name__ == "__main__":
    main()
