from __future__ import annotations

import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model weights.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
