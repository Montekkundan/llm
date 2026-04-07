from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


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


def iter_texts(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    text_files: list[str],
    alternating_chat_roles: bool,
) -> list[str]:
    if dataset_name:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        texts = []
        for item in dataset:
            text = normalize_text(item[text_column], alternating_chat_roles)
            if text:
                texts.append(text)
        return texts
    texts: list[str] = []
    for path in text_files:
        texts.extend(line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return texts


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for picoLLM cloud pretraining.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    texts = iter_texts(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.text_column,
        args.text_file,
        args.alternating_chat_roles,
    )
    if not texts:
        raise SystemExit("No text found. Pass --dataset-name or at least one --text-file.")

    tokenizer = Tokenizer(BPE(unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    fast_tokenizer.save_pretrained(output_dir)
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "num_texts": len(texts),
                "vocab_size": args.vocab_size,
                "min_frequency": args.min_frequency,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "text_column": args.text_column,
                "text_files": args.text_file,
                "alternating_chat_roles": args.alternating_chat_roles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
