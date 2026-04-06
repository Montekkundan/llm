from __future__ import annotations

import argparse
import json
from pathlib import Path

from picollm.common import generate_reply, load_generation_bundle


PROMPTS = [
    "Why is the sky blue?",
    "Explain self-attention in four sentences.",
    "Write a short poem about embeddings.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a base model and a LoRA adapter on a small prompt set.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = load_generation_bundle(args.base_model, device=args.device)
    adapted = load_generation_bundle(args.base_model, adapter_path=args.adapter, device=args.device) if args.adapter else None
    rows = []
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        row = {
            "prompt": prompt,
            "base": generate_reply(base.model, base.tokenizer, messages, temperature=0.0, max_new_tokens=128),
        }
        if adapted:
            row["adapter"] = generate_reply(adapted.model, adapted.tokenizer, messages, temperature=0.0, max_new_tokens=128)
        rows.append(row)
    payload = {"rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
