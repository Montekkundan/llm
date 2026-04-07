from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)


def build_examples(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    text_files: list[str],
) -> Dataset:
    if dataset_name:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        return dataset.select_columns([text_column])
    rows = []
    for file_path in text_files:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append({text_column: line})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT-style language model from scratch on cloud GPUs.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_examples(args.dataset_name, args.dataset_config, args.dataset_split, args.text_column, args.text_file)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[args.text_column], truncation=True, max_length=args.block_size)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=[args.text_column])
    vocab_size = args.vocab_size or tokenizer.vocab_size
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.hidden_size,
        n_layer=args.layers,
        n_head=args.heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
