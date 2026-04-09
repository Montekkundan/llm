from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from picollm.common.device import default_dtype_for_device, resolve_device
from picollm.common.training_preview import SampleGenerationCallback, default_chat_preview_items
from picollm.common.telemetry import ensure_reporter_ready, trainer_report_to
from picollm.pretrain_cloud.data import (
    load_nanochat_like_sft_dataset,
    load_text_dataset,
    load_tokenized_chat_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full chat SFT on your own checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="messages")
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--alternating-chat-roles", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixture", choices=["none", "nanochat-lite"], default="none")
    parser.add_argument("--smoltalk-epochs", type=int, default=1)
    parser.add_argument("--everyday-epochs", type=int, default=1)
    parser.add_argument("--identity-epochs", type=int, default=2)
    parser.add_argument("--mmlu-epochs", type=int, default=1)
    parser.add_argument("--gsm8k-epochs", type=int, default=1)
    parser.add_argument("--simple-spelling-size", type=int, default=20000)
    parser.add_argument("--spelling-bee-size", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--optimizer", choices=["auto", "adamw_torch", "adamw_torch_fused"], default="auto")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--preview-every-steps", type=int, default=0)
    parser.add_argument("--preview-max-new-tokens", type=int, default=96)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    ensure_reporter_ready(args.report_to)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    use_tokenized_chat = bool(
        args.mixture != "none"
        or (args.dataset_name and args.text_column == "messages" and not args.text_file)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=default_dtype_for_device(device),
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    resolved_optimizer = args.optimizer
    if resolved_optimizer == "auto":
        resolved_optimizer = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    if args.mixture == "nanochat-lite":
        max_length = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
        dataset = load_nanochat_like_sft_dataset(
            tokenizer,
            max_length=int(max_length) if max_length is not None else None,
            smoltalk_epochs=args.smoltalk_epochs,
            everyday_epochs=args.everyday_epochs,
            identity_epochs=args.identity_epochs,
            mmlu_epochs=args.mmlu_epochs,
            gsm8k_epochs=args.gsm8k_epochs,
            simple_spelling_size=args.simple_spelling_size,
            spelling_bee_size=args.spelling_bee_size,
        )
    elif use_tokenized_chat:
        max_length = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
        dataset = load_tokenized_chat_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            messages_column=args.text_column,
            tokenizer=tokenizer,
            max_length=int(max_length) if max_length is not None else None,
            streaming=args.streaming,
        )
    else:
        dataset = load_text_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            text_files=args.text_file,
            alternating_chat_roles=args.alternating_chat_roles,
            streaming=args.streaming,
        )

    if use_tokenized_chat:
        config = TrainingArguments(
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
            args=config,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            ),
        )
    else:
        config = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=args.bf16,
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
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    if args.preview_every_steps > 0:
        trainer.add_callback(
            SampleGenerationCallback(
                tokenizer,
                default_chat_preview_items(),
                every_steps=args.preview_every_steps,
                max_new_tokens=args.preview_max_new_tokens,
                label="chat-sft",
            )
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
