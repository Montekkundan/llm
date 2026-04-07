from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import torch

from picollm.common import generate_reply, load_generation_bundle
from picollm.pretrain_cloud.data import load_text_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a picoLLM pretraining checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-prompt", default="hi")
    args = parser.parse_args()

    dataset = load_text_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        text_files=args.text_file,
        alternating_chat_roles=args.alternating_chat_roles,
        streaming=args.streaming,
    )

    bundle = load_generation_bundle(args.model, device=args.device)
    model = bundle.model
    tokenizer = bundle.tokenizer
    losses: list[float] = []
    sampled_rows = list(islice(iter(dataset), 64))
    if not sampled_rows:
        raise SystemExit("No eval texts found.")
    for row in sampled_rows:
        encoded = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, labels=encoded["input_ids"])
        losses.append(float(outputs.loss.item()))
    average_loss = sum(losses) / len(losses)
    report = {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
        "sample_prompt": args.sample_prompt,
        "sample_response": generate_reply(
            model,
            tokenizer,
            [{"role": "user", "content": args.sample_prompt}],
            temperature=0.0,
            max_new_tokens=96,
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
