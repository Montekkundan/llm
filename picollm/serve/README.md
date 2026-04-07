# `serve/`

This path serves either:

- a pretrained instruct model
- a base model plus LoRA adapter
- your own from-scratch chat checkpoint
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

With your own from-scratch chatbot:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

## Web app

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

With your own from-scratch chatbot:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

Then open:

`http://127.0.0.1:8008`

If an adapter is configured, the web UI switches into compare mode automatically:

- left pane shows the base model
- right pane shows the LoRA-adapted model
- one prompt is sent to both panes at the same time
- both replies stream independently so you can compare them live in class

## OpenAI-compatible API

This server also exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

That makes it usable from clients that expect an OpenAI-style API, including the Vercel AI SDK openai-compatible provider path.

See:

- [apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
- [apps/opentui_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/opentui_ai_sdk_chat/README.md)

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
