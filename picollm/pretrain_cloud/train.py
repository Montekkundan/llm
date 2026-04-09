from __future__ import annotations

import argparse
import math
import os
from itertools import chain

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
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


def _world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _count_non_embedding_params(model) -> int:
    embedding_param_names = {
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    }
    total = 0
    for name, parameter in model.named_parameters():
        if name in embedding_param_names:
            continue
        total += parameter.numel()
    return int(total)


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
    parser.add_argument("--architecture", choices=["gpt2", "llama"], default="llama")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--init-model",
        default=None,
        help="Optional checkpoint to continue base pretraining from instead of initializing fresh GPT weights.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--target-param-data-ratio", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--optimizer", choices=["auto", "adamw_torch", "adamw_torch_fused"], default="auto")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--torch-compile", action="store_true")
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
    vocab_size = args.vocab_size or len(tokenizer)
    if args.init_model:
        model = AutoModelForCausalLM.from_pretrained(args.init_model)
        model_vocab_size = int(getattr(model.config, "vocab_size", vocab_size))
        if model_vocab_size != vocab_size:
            raise SystemExit(
                "The tokenizer vocabulary does not match the init-model checkpoint. "
                f"tokenizer vocab_size={vocab_size}, init-model vocab_size={model_vocab_size}."
            )
        max_positions = (
            getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_position_embeddings", None)
            or args.block_size
        )
        if int(max_positions) != args.block_size:
            raise SystemExit(
                "The requested block size does not match the init-model checkpoint. "
                f"block_size={args.block_size}, init-model max_positions={int(max_positions)}."
            )
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        if args.architecture == "llama":
            kv_heads = args.kv_heads or max(1, args.heads // 2)
            if args.heads % kv_heads != 0:
                raise SystemExit(
                    f"--heads ({args.heads}) must be divisible by --kv-heads ({kv_heads}) for llama architecture."
                )
            config = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size or (args.hidden_size * 4),
                num_hidden_layers=args.layers,
                num_attention_heads=args.heads,
                num_key_value_heads=kv_heads,
                max_position_embeddings=args.block_size,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                hidden_act="silu",
                tie_word_embeddings=False,
                attention_bias=False,
                mlp_bias=False,
                rope_theta=args.rope_theta,
            )
            model = LlamaForCausalLM(config)
        else:
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
    resolved_optimizer = args.optimizer
    if resolved_optimizer == "auto":
        resolved_optimizer = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    resolved_max_steps = args.max_steps
    if resolved_max_steps <= 0:
        if args.target_param_data_ratio <= 0:
            raise SystemExit("Set either --max-steps to a positive integer or provide --target-param-data-ratio.")
        tokens_per_step = args.batch_size * args.grad_accum * args.block_size * _world_size()
        non_embedding_params = _count_non_embedding_params(model)
        target_tokens = int(non_embedding_params * args.target_param_data_ratio)
        resolved_max_steps = max(1, math.ceil(target_tokens / max(tokens_per_step, 1)))
        print(
            "Resolved pretraining horizon from target-param-data-ratio: "
            f"non_embedding_params={non_embedding_params:,}, "
            f"target_tokens={target_tokens:,}, "
            f"tokens_per_step={tokens_per_step:,}, "
            f"max_steps={resolved_max_steps:,}",
            flush=True,
        )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=resolved_max_steps,
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
        optim=resolved_optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        dataloader_num_workers=args.dataloader_num_workers,
        torch_compile=args.torch_compile,
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
