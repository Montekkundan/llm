# Base vs LoRA Demo Prompts

Use the same prompts on:

1. the base pretrained model
2. the same base model plus your LoRA adapter

Then compare:

- instruction following
- tone
- whether the adapted model answers like a lecture assistant
- whether it consistently uses a structured format
- whether the base model stays more general-purpose

## Core prompts

- `Explain tokenization for a first-year student.`
- `Give me a two-step study plan for learning transformers.`
- `What is LoRA?`
- `Use one analogy to explain self-attention.`

## More prompts

- `What does quantization do?`
- `What is the difference between a base model and an SFT model?`
- `What is a KV cache?`
- `Explain positional encoding.`
- `Write a short poem about embeddings in exactly two lines.`

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

What to look for:

- the base model is often good, but it stays like a general-purpose assistant
- the adapted model should start sounding like a compact lecture assistant
- the easiest visual cue is that the adapted model often uses a repeatable structure such as `Core idea`, `Example`, and `Takeaway`
