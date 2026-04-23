# Run picoLLM

This guide is the student-facing runtime path for `picollm`.

Use it when you want to:

- restore and run a published Hugging Face picoLLM model
- run your own picoLLM checkpoints after training
- chat with the model in the CLI
- serve the local web UI
- connect the backend to the Vercel app
- connect the backend to the OpenTUI app

## What you need

Clone the repo and install dependencies first:

```bash
git clone https://github.com/Montekkundan/llm
cd llm
uv sync --extra cpu
```

Use `uv sync --extra gpu` instead if you are on a CUDA machine.

## Choose your model source

There are two common ways to get a runnable picoLLM model.

### Option 1: Use a published Hugging Face model repo

Use this when you already have a repo such as:

- `montekkundan/picollm-v1`

If the repo is private, export your Hugging Face token first:

```bash
export HF_TOKEN=...
```

Then restore the model into the local artifact layout expected by picoLLM:

```bash
export PICOLLM_BASE_DIR=$PWD/artifacts/picollm
uv run python scripts/restore_picollm_from_hf.py montekkundan/picollm-v1 --device-type cpu
```

Important:

- use the Hugging Face model repo for restore and inference
- do not use the archive dataset repo for normal student chat runs
- the archive repo is for preservation and resume-oriented backups

### Option 2: Use your own trained picoLLM model

Use this when you already ran the picoLLM training flow in this repo.

Typical local setup:

```bash
export PICOLLM_BASE_DIR=$PWD/artifacts/picollm
```

If your training run completed under that artifact root, picoLLM can load the latest checkpoint directly from there.

## Run the chat CLI

Start an interactive chat session:

```bash
uv run python -m picollm.accelerated.chat.cli -i sft --device-type cpu
```

Run one prompt and exit:

```bash
uv run python -m picollm.accelerated.chat.cli \
  -i sft \
  --device-type cpu \
  -p "Explain tokenization for a beginner."
```

Notes:

- `-i sft` uses the chat-tuned checkpoint
- `-i base` uses the base model instead
- on Apple Silicon, use `--device-type mps`
- on CUDA, use `--device-type cuda`

## Run the local web server

Start the OpenAI-compatible backend plus the built-in local web UI:

```bash
uv run python -m picollm.accelerated.chat.web \
  --source sft \
  --device-type cpu
```

Default local endpoint:

```text
http://127.0.0.1:8008
```

OpenAI-compatible base URL:

```text
http://127.0.0.1:8008/v1
```

This is the backend URL used by the Vercel app and the OpenTUI app.

## Run the Vercel AI SDK app locally

Start the picoLLM backend first:

```bash
uv run python -m picollm.accelerated.chat.web \
  --source sft \
  --device-type cpu
```

Then in a second terminal:

```bash
cd apps/vercel_ai_sdk_chat
bun install
cp .env.example .env.local
bun run dev
```

Set these values in `.env.local`:

```bash
PICOLLM_BASE_URL=http://127.0.0.1:8008/v1
PICOLLM_API_KEY=local-demo-key
PICOLLM_MODEL=picollm-chat
```

Then open:

```text
http://127.0.0.1:3000
```

Important:

- local development can use `127.0.0.1`
- a real Vercel deployment cannot call your laptop's localhost
- for deployed Vercel usage, host the picoLLM backend on a reachable VM or other public endpoint and set `PICOLLM_BASE_URL` to that URL instead

## Run the OpenTUI app

Start the picoLLM backend first:

```bash
uv run python -m picollm.accelerated.chat.web \
  --source sft \
  --device-type cpu
```

Then in a second terminal:

```bash
cd apps/opentui_ai_sdk_chat
bun install
cp .env.example .env
bun run dev
```

Set these values in `.env`:

```bash
PICOLLM_BASE_URL=http://127.0.0.1:8008/v1
PICOLLM_API_KEY=local-demo-key
PICOLLM_MODEL=picollm-chat
```

## Which Hugging Face repo should students use?

Use the model repo for normal student usage:

- example: `montekkundan/picollm-v1`

Use the archive dataset repo only when you need:

- preservation
- debugging
- resume-oriented backups

The normal student path is:

1. restore the model repo
2. run `chat.cli` or `chat.web`
3. optionally connect the backend to the Vercel app or OpenTUI app

## Quick decision guide

Use:

- CLI if you want the fastest local test
- local web if you want a simple browser demo
- Vercel app if you want a product-style Next.js frontend
- OpenTUI if you want a terminal chat app on top of the same backend

## Related docs

- [README.md](./README.md)
- [picollm/accelerated/README.md](./picollm/accelerated/README.md)
- [apps/vercel_ai_sdk_chat/README.md](./apps/vercel_ai_sdk_chat/README.md)
- [apps/opentui_ai_sdk_chat/README.md](./apps/opentui_ai_sdk_chat/README.md)
- [picollm/accelerated/docs/hf_backup_strategy.md](./picollm/accelerated/docs/hf_backup_strategy.md)
