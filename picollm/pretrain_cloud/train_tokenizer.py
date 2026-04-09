from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from picollm.common.conversation import BOS_TOKEN, EOS_TOKEN, GPT4_SPLIT_PATTERN, PAD_TOKEN, SPECIAL_TOKENS

from .data import iter_texts


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for picoLLM cloud pretraining.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=0)
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
        streaming=args.streaming,
        max_texts=args.max_texts,
    )
    num_texts = None
    if not args.dataset_name:
        num_texts = sum(
            1
            for path in args.text_file
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if args.max_texts is not None:
            num_texts = min(num_texts, args.max_texts)

    tokenizer = Tokenizer(
        BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        )
    )
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=Regex(GPT4_SPLIT_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=args.max_texts)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        additional_special_tokens=[token for token in SPECIAL_TOKENS if token not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}],
    )
    fast_tokenizer.save_pretrained(output_dir)
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "num_texts": num_texts,
                "vocab_size": args.vocab_size,
                "min_frequency": args.min_frequency,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "text_column": args.text_column,
                "text_files": args.text_file,
                "alternating_chat_roles": args.alternating_chat_roles,
                "streaming": args.streaming,
                "max_texts": args.max_texts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
