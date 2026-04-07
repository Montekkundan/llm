# `sft_lora/`

This is the serious fine-tuning path in the repo.

Use it when you want the final chatbot path to answer well without pretending that a tiny from-scratch model will behave like a modern assistant.

The recommended flow is:

1. load a small instruct-capable base model
2. prepare chat-format data
3. run LoRA fine-tuning
4. compare base vs adapter behavior
5. optionally merge the adapter
6. serve the tuned model locally

## Recommended models

These are good small models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

## 1. Prepare a dataset

```bash
uv run python -m picollm.sft_lora.prepare_dataset \
  --output-jsonl artifacts/picollm/sft/train.jsonl
```

Or convert your own JSONL that already contains `messages`.

## 2. Fine-tune with LoRA

```bash
uv run python -m picollm.sft_lora.finetune \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset artifacts/picollm/sft/train.jsonl \
  --output-dir artifacts/picollm/lora-run \
  --device auto \
  --max-steps 200
```

On Linux + CUDA only, you can add `--use-4bit` for QLoRA-style memory savings.

Relevant docs:

- [PEFT overview](https://huggingface.co/docs/peft/index)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

## 3. Evaluate before and after

```bash
uv run python -m picollm.sft_lora.evaluate \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto \
  --output artifacts/picollm/lora_eval.json
```

## 4. Merge the adapter if you want one standalone folder

```bash
uv run python -m picollm.sft_lora.merge_adapter \
  --adapter artifacts/picollm/lora-run \
  --output-dir artifacts/picollm/lora-merged
```

## Device guidance

- Mac M-series: `--device mps`
- NVIDIA: `--device cuda`
- CPU-only machines: `--device cpu`
- simple default: `--device auto`

## What this is for

This is the right path if you want responses that sound like a real chatbot instead of a toy language model.
