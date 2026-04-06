# Base vs LoRA Demo Prompts

Use the same prompts on:

1. the base pretrained model
2. the same base model plus your LoRA adapter

Then compare:

- instruction following
- tone
- brevity vs verbosity
- domain alignment

## Core prompts

- `Why is the sky blue?`
- `Explain self-attention to a beginner in four sentences.`
- `Write a short poem about the sky.`
- `Give me a two-step study plan for learning transformers.`

## More lecture-friendly prompts

- `Explain tokenization to a first-year student in simple language.`
- `In three bullet points, explain what a decoder-only model does.`
- `Write a short analogy for attention without using math.`
- `Answer as a concise teaching assistant: what is LoRA?`

## Commands

Baseline CLI:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

Adapted CLI:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

Side-by-side comparison script:

```bash
uv run python scripts/real_chatbot_workflow/compare_base_vs_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```
