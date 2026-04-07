from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from picollm.common.device import default_dtype_for_device, resolve_device
from picollm.common.telemetry import ensure_reporter_ready, trainer_report_to
from picollm.pretrain_cloud.data import load_text_dataset


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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    ensure_reporter_ready(args.report_to)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=default_dtype_for_device(device),
    )
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
        bf16=args.bf16,
        report_to=trainer_report_to(args.report_to),
        run_name=args.run_name,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
