from __future__ import annotations

import argparse
from itertools import chain

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from picollm.common.training_preview import SampleGenerationCallback, default_pretrain_preview_items
from picollm.common.telemetry import ensure_reporter_ready, trainer_report_to

from .data import load_text_dataset


def _dataset_columns(dataset: object) -> list[str] | None:
    if getattr(dataset, "column_names", None):
        return list(dataset.column_names)
    if getattr(dataset, "features", None):
        return list(dataset.features.keys())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT-style language model from scratch on cloud GPUs.")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--preview-every-steps", type=int, default=0)
    parser.add_argument("--preview-max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    ensure_reporter_ready(args.report_to)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_text_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        text_files=args.text_file,
        alternating_chat_roles=args.alternating_chat_roles,
        streaming=args.streaming,
    )
    if args.dataset_name:
        shuffle_kwargs = {"seed": args.seed}
        if args.streaming:
            shuffle_kwargs["buffer_size"] = 10_000
        dataset = dataset.shuffle(**shuffle_kwargs)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(batch: dict[str, list[object]]) -> dict[str, list[list[int]]]:
        sequence_keys = [
            key
            for key, values in batch.items()
            if values and isinstance(values[0], list)
        ]
        concatenated = {
            key: list(chain.from_iterable(batch[key]))  # type: ignore[arg-type]
            for key in sequence_keys
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.block_size) * args.block_size
        if total_length == 0:
            return {key: [] for key in sequence_keys}
        return {
            key: [values[index : index + args.block_size] for index in range(0, total_length, args.block_size)]
            for key, values in concatenated.items()
        }

    packed_remove_columns = _dataset_columns(tokenized)
    tokenized = tokenized.map(group_texts, batched=True, remove_columns=packed_remove_columns)
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
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        seed=args.seed,
        report_to=trainer_report_to(args.report_to),
        run_name=args.run_name,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if args.preview_every_steps > 0:
        trainer.add_callback(
            SampleGenerationCallback(
                tokenizer,
                default_pretrain_preview_items(),
                every_steps=args.preview_every_steps,
                max_new_tokens=args.preview_max_new_tokens,
                label="pretrain",
            )
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
