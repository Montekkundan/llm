# Base vs Chat-SFT Demo Prompts

Use the same prompts on:

1. the base pretrained checkpoint
2. the chat-SFT checkpoint built on top of that base model

Then compare:

- plain continuation quality versus assistant-style behavior
- instruction following
- tone
- structure
- whether the chat-SFT model is easier to talk to as a student-facing assistant

## Core prompts

- `Explain tokenization for a first-year student.`
- `Give me a two-step study plan for learning transformers.`
- `Why is the sky blue?`
- `Use one analogy to explain self-attention.`

## More prompts

- `What is the difference between a base model and a chat-SFT model?`
- `Explain positional encoding.`
- `What is a KV cache?`
- `Write a short poem about embeddings in exactly two lines.`

## Commands

Base-model sample evaluation:

```bash
uv run python -m picollm.accelerated.pretrain.eval \
  --eval sample
```

Chat-SFT single-prompt check:

```bash
uv run python -m picollm.accelerated.chat.cli \
  -i sft \
  -p "Explain tokenization for a first-year student."
```

Interactive chat check:

```bash
uv run python -m picollm.accelerated.chat.cli \
  -i sft
```

What to look for:

- the base model should show general language continuation ability
- the chat-SFT model should sound more like an assistant and less like a raw sequence completer
- the easiest visual cue is that the chat-SFT model should follow the user request more directly and with cleaner structure
