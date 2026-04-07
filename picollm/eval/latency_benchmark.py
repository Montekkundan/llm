from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from picollm.common import generate_reply, load_generation_bundle
from picollm.common.chat import build_prompt

from .compare_checkpoints import DEFAULT_PROMPT_SUITE, load_prompt_suite, normalize_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure simple prompt latency for a chat checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-suite", default=str(DEFAULT_PROMPT_SUITE))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_generation_bundle(args.model, device=args.device)
    prompt_suite = load_prompt_suite(args.prompt_suite)[: args.limit]

    for _ in range(max(args.warmup, 0)):
        first_messages = normalize_messages(prompt_suite[0])
        generate_reply(
            bundle.model,
            bundle.tokenizer,
            first_messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []

    for prompt_row in prompt_suite:
        messages = normalize_messages(prompt_row)
        prompt_text = build_prompt(bundle.tokenizer, messages, add_generation_prompt=True)
        prompt_tokens = len(bundle.tokenizer(prompt_text)["input_ids"])

        started = time.perf_counter()
        response = generate_reply(
            bundle.model,
            bundle.tokenizer,
            messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        latency_seconds = time.perf_counter() - started
        response_tokens = len(bundle.tokenizer(response)["input_ids"])

        latencies.append(latency_seconds)
        prompt_lengths.append(prompt_tokens)
        response_lengths.append(response_tokens)
        rows.append(
            {
                "id": prompt_row.get("id"),
                "category": prompt_row.get("category"),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "latency_seconds": latency_seconds,
                "tokens_per_second": response_tokens / latency_seconds if latency_seconds > 0 else None,
            }
        )

    payload = {
        "model": args.model,
        "device": bundle.device,
        "prompt_suite": str(args.prompt_suite),
        "rows": rows,
        "summary": {
            "count": len(rows),
            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "mean_prompt_tokens": statistics.mean(prompt_lengths) if prompt_lengths else None,
            "mean_response_tokens": statistics.mean(response_lengths) if response_lengths else None,
            "mean_tokens_per_second": (
                statistics.mean(
                    [row["tokens_per_second"] for row in rows if row["tokens_per_second"] is not None]
                )
                if rows
                else None
            ),
        },
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
