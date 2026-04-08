from __future__ import annotations

import argparse

from picollm.common import generate_reply, load_generation_bundle, stream_reply


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CLI chat for picoLLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    bundle = load_generation_bundle(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        device=args.device,
        quantization=args.quantization,
    )
    messages = [{"role": "system", "content": args.system_prompt}]
    print("picoLLM CLI. Type 'exit' to quit.")
    while True:
        user_text = input("\nuser> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user_text})
        if args.no_stream:
            reply = generate_reply(
                bundle.model,
                bundle.tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"assistant> {reply}")
        else:
            print("assistant> ", end="", flush=True)
            pieces: list[str] = []
            for piece in stream_reply(
                bundle.model,
                bundle.tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            ):
                pieces.append(piece)
                print(piece, end="", flush=True)
            print()
            reply = "".join(pieces).strip()
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
