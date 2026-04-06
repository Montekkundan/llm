# picoLLM Runbook

This is the single student-facing doc for the serious chatbot path.

Use this file when you want to run:

- a baseline pretrained chatbot
- a LoRA fine-tuning demo
- a base-vs-LoRA comparison
- a served adapted model
- a Vast.ai cloud pretraining workflow
- a local checkpoint after cloud training

## 0. Install the repo

```bash
uv sync
```

Optional but recommended for Hugging Face:

```bash
hf auth login
```

## 1. Baseline Qwen

CLI:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

Web:

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

## 2. Prepare a LoRA dataset

```bash
uv run python -m picollm.sft_lora.prepare_dataset \
  --output-jsonl artifacts/picollm/sft/train.jsonl
```

## 3. Run LoRA fine-tuning

```bash
uv run python -m picollm.sft_lora.finetune \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset artifacts/picollm/sft/train.jsonl \
  --output-dir artifacts/picollm/lora-run \
  --device auto \
  --max-steps 200
```

## 4. Evaluate base vs LoRA

Simple evaluation:

```bash
uv run python -m picollm.sft_lora.evaluate \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

Lecture-friendly side-by-side comparison:

```bash
uv run python scripts/real_chatbot_workflow/compare_base_vs_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

## 5. Serve the adapted model

CLI:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

Web:

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

## 6. Demo prompts

Use these on both the base model and the LoRA-adapted model:

- `Why is the sky blue?`
- `Explain self-attention to a beginner in four sentences.`
- `Write a short poem about the sky.`
- `Give me a two-step study plan for learning transformers.`

## 7. Vast.ai helper commands

Export your key first:

```bash
export VAST_API_KEY="..."
```

Search offers:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 1 \
  --gpu-ram-gb 24
```

Create instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --label picollm-train
```

Show instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --instance-id 12345678
```

Print SSH and copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 12345678
```

## 8. Train your own checkpoint in the cloud

On the cloud machine:

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --text-file data/pretrain/sample_corpus.txt \
  --output-dir artifacts/picollm/tokenizer

uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --text-file data/pretrain/sample_corpus.txt \
  --output-dir artifacts/picollm/pretrain-run \
  --max-steps 2000 \
  --bf16
```

## 9. Reuse that checkpoint locally

If you copied the checkpoint folder back to your machine:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

Web:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

## 10. Related docs

- [README.md](/Users/montekkundan/Developer/ML/llm/picollm/README.md)
- [HUGGING_FACE_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/HUGGING_FACE_SETUP.md)
- [pretrain_cloud/VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)
- [sft_lora/README.md](/Users/montekkundan/Developer/ML/llm/picollm/sft_lora/README.md)
- [serve/README.md](/Users/montekkundan/Developer/ML/llm/picollm/serve/README.md)
- [../apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
- [../apps/opentui_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/opentui_ai_sdk_chat/README.md)
