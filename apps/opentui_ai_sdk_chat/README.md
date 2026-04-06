# OpenTUI AI SDK Chat App

This app is the terminal UI companion to the Vercel web app example.

It uses:

- OpenTUI for the terminal interface
- AI SDK `streamText`
- an OpenAI-compatible `picollm` backend

The teaching point is the same as the web app:

- the UI layer is separate
- the model backend is separate
- the client only needs a stable API contract

AI SDK reference:

- [OpenAI-compatible providers](https://ai-sdk.dev/providers/openai-compatible-providers)

OpenTUI reference:

- [Getting started](https://opentui.com/docs/getting-started/)
- [React bindings](https://opentui.com/docs/bindings/react/)

## Why this app exists

The Vercel AI SDK app shows how ChatGPT-style browser products are wired.

This app shows the parallel terminal story:

- terminal UI instead of browser UI
- same `picollm` backend
- same model identity
- same OpenAI-compatible contract

That makes it a good teaching bridge toward tools like Claude Code, Codex CLI, OpenCode, and Gemini CLI.

The important caveat is:

this app is a terminal chat client, not yet a full coding agent. It teaches the UI and backend contract first.

## Backend requirement

Run the backend first:

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

That exposes:

```text
http://127.0.0.1:8008/v1
```

## App setup

From this folder:

```bash
bun install
cp .env.example .env
bun run dev
```

## Environment variables

```bash
PICOLLM_BASE_URL=http://127.0.0.1:8008/v1
PICOLLM_API_KEY=local-demo-key
PICOLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

## Controls

- `Enter` submits the current prompt
- `Esc` exits the app

## Related course docs

- [README.md](/Users/montekkundan/Developer/ML/llm/README.md)
- [picollm/RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md)
- [apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
