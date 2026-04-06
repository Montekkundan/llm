from __future__ import annotations

import argparse

from picollm.common import generate_reply, load_generation_bundle


DEMO_PROMPTS = [
    "Why is the sky blue?",
    "Explain self-attention to a beginner in four sentences.",
    "Write a short poem about the sky.",
    "Give me a two-step study plan for learning transformers.",
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
