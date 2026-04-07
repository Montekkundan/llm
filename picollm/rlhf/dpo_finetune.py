from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from picollm.common.device import default_dtype_for_device, resolve_device
from picollm.common.telemetry import ensure_reporter_ready, trainer_report_to


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal DPO on a chat checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="JSONL file or dataset name with prompt/chosen/rejected columns.")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    ensure_reporter_ready(args.report_to)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=default_dtype_for_device(device),
    )

    dataset = (
        load_dataset("json", data_files=args.dataset, split="train")
        if args.dataset.endswith(".jsonl")
        else load_dataset(args.dataset, split=args.dataset_split)
    )
    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise SystemExit(f"DPO dataset is missing required columns: {sorted(missing)}")

    config = DPOConfig(
        output_dir=args.output_dir,
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
        beta=args.beta,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(
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

