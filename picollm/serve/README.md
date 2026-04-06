# `serve/`

This path serves either:

- a pretrained instruct model
- a base model plus LoRA adapter
- a merged model checkpoint

It is OS-agnostic:

- macOS uses `mps`
- Linux and Windows with NVIDIA use `cuda`
- CPU fallback works everywhere

## CLI chat

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

With an adapter:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

## Web app

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

Then open:

`http://127.0.0.1:8008`

## Quantization

This repo exposes simple quantization flags for serving:

- `none`
- `8bit`
- `4bit`

Use them only on CUDA machines with the required backend installed.

Example:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --quantization 4bit
```

For Macs, keep it simple:

- use normal weights
- optionally use a smaller model
- leave aggressive quantized CUDA stacks out of the first lecture pass
