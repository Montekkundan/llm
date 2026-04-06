from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a chat-format dataset for LoRA fine-tuning.")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        rows = [json.loads(line) for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [
            {
                "messages": [
                    {"role": "system", "content": "You are a concise teaching assistant."},
                    {"role": "user", "content": "Why is the sky blue?"},
                    {
                        "role": "assistant",
                        "content": "The sky looks blue because air scatters short blue wavelengths of sunlight more strongly than red wavelengths.",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a concise teaching assistant."},
                    {"role": "user", "content": "Write a two-line poem about attention."},
                    {
                        "role": "assistant",
                        "content": "Each token listens for the signals it should keep.\nAttention turns a sequence into context running deep.",
                    },
                ]
            },
        ]

    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
