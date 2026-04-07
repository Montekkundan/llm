from __future__ import annotations

import argparse
import json
from pathlib import Path

from picollm.common import generate_reply, load_generation_bundle

from .compare_checkpoints import load_prompt_suite, normalize_messages


DEFAULT_SAFETY_PROMPT_SUITE = Path(__file__).with_name("safety_prompt_suite.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small safety and jailbreak-oriented prompt suite.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt-suite", default=str(DEFAULT_SAFETY_PROMPT_SUITE))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_generation_bundle(args.model, adapter_path=args.adapter, device=args.device)
    prompt_suite = load_prompt_suite(args.prompt_suite)
    rows = []
    for prompt_row in prompt_suite:
        if not isinstance(prompt_row, dict):
            raise SystemExit("Each safety prompt row must be a JSON object.")
        messages = normalize_messages(prompt_row)
        response = generate_reply(
            bundle.model,
            bundle.tokenizer,
            messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append(
            {
                "id": prompt_row.get("id"),
                "category": prompt_row.get("category"),
                "messages": messages,
                "response": response,
                "review_notes": "",
            }
        )
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "prompt_suite": str(args.prompt_suite),
        "rows": rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

