from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

from picollm.common import generate_reply, load_generation_bundle
from picollm.pretrain_cloud.dataset_aliases import resolve_dataset_name


def normalize_text(value: object, alternating_chat_roles: bool) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return ""
        if alternating_chat_roles:
            rendered = []
            for index, part in enumerate(parts):
                role = "<|user|>" if index % 2 == 0 else "<|assistant|>"
                rendered.append(f"{role} {part}")
            return "\n".join(rendered)
        return "\n".join(parts)
    return str(value).strip()


def load_texts(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    text_files: list[str],
    alternating_chat_roles: bool,
) -> Dataset:
    if dataset_name:
        dataset = load_dataset(resolve_dataset_name(dataset_name), dataset_config, split=dataset_split)
        rows = []
        for item in dataset:
            text = normalize_text(item[text_column], alternating_chat_roles)
            if text:
                rows.append({text_column: text})
        return Dataset.from_list(rows)
    items = []
    for path in text_files:
        items.extend({text_column: line.strip()} for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return Dataset.from_list(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a picoLLM pretraining checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample-prompt", default="hi")
    args = parser.parse_args()

    dataset = load_texts(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.text_column,
        args.text_file,
        args.alternating_chat_roles,
    )
    if len(dataset) == 0:
        raise SystemExit("No eval texts found.")

    bundle = load_generation_bundle(args.model, device=args.device)
    model = bundle.model
    tokenizer = bundle.tokenizer
    losses: list[float] = []
    for row in dataset.select(range(min(64, len(dataset)))):
        encoded = tokenizer(row[args.text_column], return_tensors="pt", truncation=True, max_length=512)
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
