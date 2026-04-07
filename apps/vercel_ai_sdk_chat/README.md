# Vercel AI SDK Chat App

This app is the production-style frontend companion to `picollm/`.

It keeps the UI on Next.js + Vercel AI SDK, while the model stays on a separate OpenAI-compatible backend. In this course, that backend is `picollm`.

AI SDK reference:

- [OpenAI-compatible providers](https://ai-sdk.dev/providers/openai-compatible-providers)

## Why this app exists

The earlier course code explains how the model works.

This app explains how a real web product consumes a model:

- browser UI in Next.js
- streaming chat through the AI SDK
- model served from an external endpoint
- deployment of the web app independent from GPU hosting

That split is important to understand:

- Vercel is a great place to host the app
- the model itself usually lives somewhere else

## Backend requirement

Run the `picollm` server first so this app has an OpenAI-compatible endpoint to call:

```bash
uv run python -m picollm.serve.chat_web \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

That starts the backend at:

```text
http://127.0.0.1:8008
```

Its OpenAI-compatible base URL is:

```text
http://127.0.0.1:8008/v1
```

## App setup

From this folder:

```bash
npm install
cp .env.example .env.local
npm run dev
```

Then open:

```text
http://127.0.0.1:3000
```

## Environment variables

Use these in `.env.local`:

```bash
PICOLLM_BASE_URL=http://127.0.0.1:8008/v1
PICOLLM_API_KEY=local-demo-key
PICOLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

If you deploy the model elsewhere, only `PICOLLM_BASE_URL` and possibly `PICOLLM_API_KEY` need to change.

## What changed from the original template

The original `chat-template` used AI Gateway with a hosted model.

This version uses:

- `@ai-sdk/openai-compatible`
- a `picollm` backend
- configurable base URL, API key, and model id

The route is:

- [app/api/chat/route.ts](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/app/api/chat/route.ts)

## Deploying the frontend

This app can be deployed on Vercel, but the model backend should remain on:

- your local machine for demos
- a cloud VM
- Vast.ai
- any host exposing an OpenAI-compatible API

The frontend and backend do not need to be deployed together.

## Related course docs

- [README.md](/Users/montekkundan/Developer/ML/llm/README.md)
- [picollm/RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md)
- [picollm/serve/README.md](/Users/montekkundan/Developer/ML/llm/picollm/serve/README.md)
