# Hugging Face Setup

You only need a Hugging Face token for these cases:

- pushing weights to the Hub
- downloading gated or private models
- avoiding anonymous rate limits

## Login

```bash
hf auth login
```

Or export:

```bash
export HF_TOKEN="..."
```

The one-command cloud path also accepts:

```bash
bash picollm/pretrain_cloud/speedrun.sh \
  --web \
  --hf-repo-id your-name/picollm-chat-sft
```

If you use that flag, the script checks for either `HF_TOKEN` or an existing `hf auth login` session before training starts.

Official docs:

- [User access tokens](https://huggingface.co/docs/hub/main/en/security-tokens)
- [CLI login](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Upload to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)

## Push a trained checkpoint

From-scratch chatbot:

```bash
uv run python -m picollm.pretrain_cloud.push_to_hub \
  --folder artifacts/picollm/chat-sft-run \
  --repo-id your-name/picollm-chat-sft
```

Base-only checkpoint:

```bash
uv run python -m picollm.pretrain_cloud.push_to_hub \
  --folder artifacts/picollm/pretrain-run \
  --repo-id your-name/picollm-pretrain
```

## Run a pushed model locally

From-scratch chatbot from the Hub:

```bash
uv run python -m picollm.serve.chat_cli \
  --model your-name/picollm-chat-sft \
  --device auto
```

Base-only checkpoint from the Hub:

```bash
uv run python -m picollm.serve.chat_cli \
  --model your-name/picollm-pretrain \
  --device auto
```

## Run a LoRA workflow with a public base model

Public model:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

Public base model plus local adapter:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```
