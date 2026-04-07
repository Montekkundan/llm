from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from picollm.common import generate_reply, load_generation_bundle


DEMO_PROMPTS = [
    "Explain tokenization for a first-year student.",
    "Give me a two-step study plan for learning transformers.",
    "What is LoRA?",
    "Use one analogy to explain self-attention.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model answers with a LoRA-adapted model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    base = load_generation_bundle(args.base_model, device=args.device)
    adapted = load_generation_bundle(args.base_model, adapter_path=args.adapter, device=args.device)

    for prompt in DEMO_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        print("=" * 100)
        print("PROMPT:", prompt)
        print()
        print("LOOK FOR: a more lecture-like structure in the adapted model, especially 'Core idea', 'Example', and 'Takeaway'.")
        print()
        print("BASE:")
        print(
            generate_reply(
                base.model,
                base.tokenizer,
                messages,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print()
        print("LORA:")
        print(
            generate_reply(
                adapted.model,
                adapted.tokenizer,
                messages,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print()


if __name__ == "__main__":
    main()
