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

Important note:

This default LoRA demo is designed to show a visible behavior shift in:

- conciseness
- formatting
- instruction following

It is not meant to make the adapted model broadly "smarter" than Qwen on general knowledge. The right classroom comparison is:

- base model answers well but often ignores your exact formatting constraints
- adapted model follows the course-style answer format more consistently

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

With `--adapter` enabled, the web app opens in side-by-side compare mode:

- left pane: base Qwen
- right pane: base Qwen + your LoRA adapter
- the same prompt is sent to both panes
- both answers stream separately so the difference is easy to show live

## 6. Demo prompts

Use these on both the base model and the LoRA-adapted model:

- `Explain tokenization for a first-year student. Keep it to exactly two bullet points.`
- `Give me a two-step study plan for learning transformers. Use the format 'Step 1:' and 'Step 2:'.`
- `What is LoRA? Answer in two short sentences with no hype.`
- `Use one analogy to explain self-attention. Keep it under 35 words.`

## 7. Vast.ai helper commands

Export your key first:

```bash
export VAST_API_KEY="..."
```

Before you create the instance, add your SSH public key to Vast.ai.

Why this matters:

- Vast SSH access uses public-key authentication
- account-level SSH keys are added only to new instances
- if you add the key after the instance already exists, that running instance will not automatically receive it

Quick setup:

- macOS/Linux:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```

- Windows PowerShell:

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
Get-Content $HOME\.ssh\id_ed25519.pub
```

Paste that public key into Vast.ai under `Manage Keys` -> `SSH Keys`.

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

Important:

- `--offer-id` is the offer you picked from `vast_search_offers`
- the create command returns a JSON object
- use `new_contract` from that JSON as the real numeric instance id for the next commands
- do not use `instance_api_key` as the instance id

Example:

```json
{
  "success": true,
  "new_contract": 34276100,
  "instance_api_key": "..."
}
```

In that case, the instance id is `34276100`.

Show instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --instance-id 34276100
```

Print SSH and copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 34276100
```

## 8. Train your own checkpoint in the cloud

After `vast_show_instance` says the instance is running, use `vast_access` to print the SSH command.

If SSH says `Permission denied (publickey)`, the usual reason is that the SSH key was added after the instance was already created. In that case, recreate the instance or add the key through Vast's instance-specific SSH flow.

If you are on macOS and SSH authenticates but the session immediately closes with a message like `missing or unsuitable terminal: xterm-ghostty`, try:

```bash
TERM=xterm-256color ssh -p 36100 root@ssh1.vast.ai
```

That issue can happen with Ghostty because some minimal Ubuntu images do not have terminfo for `xterm-ghostty`.

Then SSH into the Vast machine and run.

Fastest working path:

- use a public Hugging Face dataset instead of a local text file
- `wikitext` is already supported by these scripts

Run:

```bash
git clone https://github.com/Montekkundan/llm.git
cd llm
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
uv sync

uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split train \
  --text-column text \
  --output-dir artifacts/picollm/tokenizer

uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split train \
  --text-column text \
  --output-dir artifacts/picollm/pretrain-run \
  --max-steps 2000 \
  --bf16
```

Optional:

- if you want the cloud checkpoint on the Hugging Face Hub later, export `HF_TOKEN` on the machine first
- if you only want the checkpoint locally, you can skip Hugging Face completely and copy the folder back directly

When training finishes, your checkpoint folder on the Vast machine will usually be:

```bash
/root/llm/artifacts/picollm/pretrain-run
```

If your repo lives somewhere else on the machine, adjust that path accordingly.

If you prefer your own corpus instead of `wikitext`, create a text file on the Vast machine and pass it with repeated `--text-file` flags.

## 9. Reuse that checkpoint locally

First, print the copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 34276100 \
  --local-dir artifacts/picollm/pretrain-run \
  --remote-dir /root/llm/artifacts/picollm/pretrain-run
```

That will print `scp` and `rsync` commands you can run from your laptop.

After you copy the checkpoint folder back to your machine, you can test it locally with the same serving tools:

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

So the practical cloud loop is:

1. rent the Vast instance
2. SSH into it
3. run tokenizer training and pretraining there
4. copy `artifacts/picollm/pretrain-run` back to your laptop
5. point `chat_cli` or `chat_web` at that local checkpoint path

## 10. Related docs

- [README.md](/Users/montekkundan/Developer/ML/llm/picollm/README.md)
- [HUGGING_FACE_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/HUGGING_FACE_SETUP.md)
- [pretrain_cloud/VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)
- [sft_lora/README.md](/Users/montekkundan/Developer/ML/llm/picollm/sft_lora/README.md)
- [serve/README.md](/Users/montekkundan/Developer/ML/llm/picollm/serve/README.md)
- [../apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
- [../apps/opentui_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/opentui_ai_sdk_chat/README.md)
