from __future__ import annotations

import argparse
import json
from pathlib import Path

from picollm.common import generate_reply, load_generation_bundle


DEFAULT_PROMPT_SUITE = Path(__file__).with_name("prompt_suite.json")


def load_prompt_suite(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit("Prompt suite must be a JSON array.")
    return payload


def normalize_messages(prompt_row: dict[str, object]) -> list[dict[str, str]]:
    messages = prompt_row.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise SystemExit("Prompt suite 'messages' value must be a list.")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise SystemExit("Each message must be an object with role/content.")
            normalized.append(
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
            )
        return normalized

    prompt = prompt_row.get("prompt")
    if prompt is None:
        raise SystemExit("Each prompt row must provide either 'prompt' or 'messages'.")
    return [{"role": "user", "content": str(prompt)}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one or more checkpoints on a shared chat prompt suite.")
    parser.add_argument("--model", action="append", required=True, help="Model path or Hub id. Repeat for multiple checkpoints.")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Optional adapter path for each --model. Repeat in the same order as --model.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional human-readable label for each --model. Defaults to the model path.",
    )
    parser.add_argument("--prompt-suite", default=str(DEFAULT_PROMPT_SUITE))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    labels = list(args.label)
    adapters = list(args.adapter)
    if labels and len(labels) != len(args.model):
        raise SystemExit("If you pass --label, provide exactly one label per --model.")
    if not labels:
        labels = list(args.model)
    if adapters and len(adapters) != len(args.model):
        raise SystemExit("If you pass --adapter, provide exactly one adapter value per --model.")
    if not adapters:
        adapters = [None] * len(args.model)

    bundles = {
        label: load_generation_bundle(model_name_or_path=model, adapter_path=adapter, device=args.device)
        for label, model, adapter in zip(labels, args.model, adapters, strict=True)
    }
    prompt_suite = load_prompt_suite(args.prompt_suite)

    rows: list[dict[str, object]] = []
    for prompt_row in prompt_suite:
        if not isinstance(prompt_row, dict):
            raise SystemExit("Each prompt suite row must be a JSON object.")
        messages = normalize_messages(prompt_row)
        row = {
            "id": prompt_row.get("id"),
            "category": prompt_row.get("category"),
            "messages": messages,
            "responses": {},
        }
        for label, bundle in bundles.items():
            row["responses"][label] = generate_reply(
                bundle.model,
                bundle.tokenizer,
                messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
        rows.append(row)

    payload = {
        "prompt_suite": str(args.prompt_suite),
        "models": [
            {"label": label, "model": model, "adapter": adapter}
            for label, model, adapter in zip(labels, args.model, adapters, strict=True)
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "rows": rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
