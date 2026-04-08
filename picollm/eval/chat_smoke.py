from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from picollm.common import generate_reply, load_generation_bundle

from .compare_checkpoints import DEFAULT_PROMPT_SUITE, load_prompt_suite, normalize_messages


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _max_ngram_repeat(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return max(counts.values(), default=0)


def analyze_response(text: str) -> dict[str, object]:
    stripped = text.strip()
    words = _tokenize_words(stripped)
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0
    max_bigram_repeat = _max_ngram_repeat(words, 2)
    max_trigram_repeat = _max_ngram_repeat(words, 3)
    flags: list[str] = []

    if not stripped:
        flags.append("empty")
    if words and len(words) < 3:
        flags.append("too_short")
    if len(words) >= 24 and unique_ratio < 0.32:
        flags.append("low_diversity")
    if max_bigram_repeat >= 6 or max_trigram_repeat >= 4:
        flags.append("looping_ngram")

    catastrophic = any(flag in {"empty", "looping_ngram"} for flag in flags)
    return {
        "chars": len(stripped),
        "word_count": len(words),
        "unique_ratio": round(unique_ratio, 4),
        "max_bigram_repeat": max_bigram_repeat,
        "max_trigram_repeat": max_trigram_repeat,
        "flags": flags,
        "catastrophic": catastrophic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick chat smoke test against a checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--prompt-suite", default=str(DEFAULT_PROMPT_SUITE))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-catastrophic", action="store_true")
    args = parser.parse_args()

    bundle = load_generation_bundle(args.model, device=args.device)
    prompt_suite = load_prompt_suite(args.prompt_suite)
    label = args.label or args.model

    rows: list[dict[str, object]] = []
    catastrophic_count = 0
    flagged_count = 0

    print(f"=== chat smoke: {label} ===", flush=True)
    for prompt_row in prompt_suite:
        if not isinstance(prompt_row, dict):
            raise SystemExit("Each prompt suite row must be a JSON object.")
        messages = normalize_messages(prompt_row)
        response = generate_reply(
            bundle.model,
            bundle.tokenizer,
            messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        analysis = analyze_response(response)
        flags = analysis["flags"]
        if flags:
            flagged_count += 1
        if analysis["catastrophic"]:
            catastrophic_count += 1

        row = {
            "id": prompt_row.get("id"),
            "category": prompt_row.get("category"),
            "messages": messages,
            "response": response,
            "analysis": analysis,
        }
        rows.append(row)

        prompt_preview = messages[-1]["content"] if messages else ""
        print(f"[{row['id']}] {prompt_preview}", flush=True)
        print(f"-> {response or '<empty>'}", flush=True)
        if flags:
            print(f"   flags: {', '.join(flags)}", flush=True)

    payload = {
        "model": args.model,
        "label": label,
        "prompt_suite": str(args.prompt_suite),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            "rows": len(rows),
            "flagged_rows": flagged_count,
            "catastrophic_rows": catastrophic_count,
        },
        "rows": rows,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)

    if args.fail_on_catastrophic and catastrophic_count > 0:
        raise SystemExit(f"Chat smoke test found {catastrophic_count} catastrophic responses.")


if __name__ == "__main__":
    main()
