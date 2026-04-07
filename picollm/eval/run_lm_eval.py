from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval-harness against a local or Hub checkpoint.")
    parser.add_argument("--model", required=True, help="Local checkpoint path or Hub id.")
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    lm_eval_path = shutil.which("lm_eval")
    if lm_eval_path is None:
        raise SystemExit(
            "lm_eval was not found on PATH.\n"
            "Install lm-eval-harness first, for example:\n"
            "  uv tool install lm-eval\n"
            "or add it to your environment before rerunning."
        )

    command = [
        lm_eval_path,
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model}",
        "--tasks",
        args.tasks,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_fewshot",
        str(args.num_fewshot),
    ]
    if args.output_path:
        command.extend(["--output_path", args.output_path])

    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

