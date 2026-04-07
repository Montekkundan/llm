from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from picollm.common.device import default_dtype_for_device, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA SFT on a small instruct or base model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="JSONL file or dataset name with a messages column.")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if args.use_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = default_dtype_for_device(device)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if "device_map" not in model_kwargs:
        model = model.to(device)

    dataset = load_dataset("json", data_files=args.dataset, split="train") if args.dataset.endswith(".jsonl") else load_dataset(args.dataset, split=args.dataset_split)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        dataloader_pin_memory=device == "cuda",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
