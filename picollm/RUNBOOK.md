# picoLLM Runbook

This is the single run doc for the serious chatbot path.

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

Side-by-side comparison:

```bash
uv run python scripts/real_chatbot_workflow/compare_base_vs_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

Shared prompt-suite comparison:

```bash
uv run python -m picollm.eval.compare_checkpoints \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --label base \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --label lora \
  --output artifacts/picollm/base_vs_lora_compare.json
```

## 5. Serve the adapted model

Important note:

This default LoRA demo is designed to show a visible behavior shift in:

- conciseness
- formatting
- instruction following

It is not meant to make the adapted model broadly "smarter" than Qwen on general knowledge. The useful comparison is:

- base model answers well but often ignores your exact formatting constraints
- adapted model follows the target answer format more consistently

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
  --num-gpus 2 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

That is the default recommendation now.

If you want a different setup later, you can change the GPU choice in the Vast.ai console and then adjust the preset or GPU count as needed. For example:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "A100 SXM4" \
  --num-gpus 1 \
  --gpu-ram-gb 80 \
  --reliability 0.995 \
  --limit 10
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
- use `new_contract` from that JSON as the real numeric contract value for the next commands
- do not use `instance_api_key` as the contract value

Example:

```json
{
  "success": true,
  "new_contract": 34276100,
  "instance_api_key": "..."
}
```

In that case, the `new_contract` value is `34276100`.

Show instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --new-contract 34276100
```

Print the SSH and copy commands you should run next:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100
```

`vast_access` only prints the commands. It does not SSH into the machine or copy the checkpoint by itself.

## 8. Train your own checkpoint in the cloud

After `vast_show_instance` says the instance is running, use `vast_access` to print the SSH and copy commands.

If SSH says `Permission denied (publickey)`, the usual reason is that the SSH key was added after the instance was already created. In that case, recreate the instance or add the key through Vast's instance-specific SSH flow.

If you are on macOS and SSH authenticates but the session immediately closes with a message like `missing or unsuitable terminal: xterm-ghostty`, try:

```bash
TERM=xterm-256color ssh -p 36100 root@ssh1.vast.ai
```

That issue can happen with Ghostty because some minimal Ubuntu images do not have terminfo for `xterm-ghostty`.

Then SSH into the Vast machine and run.

If you want hosted telemetry or Hub upload, do the auth step on the box before you start the run:

```bash
wandb login
hf auth login
```

If you already passed `--hf-token` or `--wandb-api-key` when creating the instance, you can skip those login commands.

Recommended cloud path:

- stage 1: base pretrain from scratch on general text
- stage 2: full chat SFT on your own checkpoint

After cloud training, compare the base checkpoint and the chat checkpoint on the shared prompt suite:

```bash
uv run python -m picollm.eval.compare_checkpoints \
  --model artifacts/picollm/pretrain-run \
  --label base \
  --model artifacts/picollm/chat-sft-run \
  --label chat \
  --output artifacts/picollm/checkpoint_compare.json
```

Then measure simple serving latency:

```bash
uv run python -m picollm.eval.latency_benchmark \
  --model artifacts/picollm/chat-sft-run \
  --output artifacts/picollm/latency_report.json
```

Run a small safety-oriented red-team pass:

```bash
uv run python -m picollm.eval.safety_red_team \
  --model artifacts/picollm/chat-sft-run \
  --output artifacts/picollm/safety_report.json
```

If you want a minimal preference-optimization demo after SFT, use DPO:

```bash
uv run python -m picollm.rlhf.dpo_finetune \
  --model artifacts/picollm/chat-sft-run \
  --dataset preferences.jsonl \
  --output-dir artifacts/picollm/dpo-run
```
- for the serious cloud capstone path, use `8x H100`
- if you want a slower but still strong alternative, use `8x A100`
- if you want a cheaper teaching-scale run, use `2x RTX 4090`

For the serious capstone path in this course:

- to get a real from-scratch conversational chatbot quickly, use the serious cloud capstone path on `8x H100`
- expect about `3 to 5 hours` for the calibrated capstone recipe
- at roughly `$18/hour`, that is about `$54 to $90` before bandwidth and storage overhead
- this follows the same general idea as `nanochat`'s serious cloud speedrun path
- if you do not want to pay for that run, use the shared Hugging Face checkpoint and still complete the rest of the workflow
- the current capstone preset is intentionally much shorter than the older `50000`-step long run and is calibrated around a smaller token budget for this `336M` parameter model

If you want the one-command path instead of typing every stage manually, first clone the repo on the Vast box:

```bash
git clone https://github.com/Montekkundan/llm.git
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh
```

That ends in the CLI by default, and it uses the budget teaching preset: `2x4090`.

If you want the serious capstone run instead, use:

```bash
git clone https://github.com/Montekkundan/llm.git
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 8xh100 --web
```

If you want the serious capstone run plus Hub upload and W&B telemetry, use:

```bash
git clone https://github.com/Montekkundan/llm.git
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --preset 8xh100 \
  --web \
  --hf-repo-id your-name/picollm-chat-sft \
  --hf-token "$HF_TOKEN" \
  --report-to wandb \
  --run-name picollm-capstone-8xh100 \
  --wandb-project picollm \
  --wandb-entity montekkundan \
  --wandb-api-key "$WANDB_API_KEY"
```

If you want the web UI instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --web
```

If you also want the final chatbot pushed to the Hub at the end:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --web \
  --hf-repo-id your-name/picollm-chat-sft
```

If you rent `8x A100` instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 8xa100 --web
```

If you rent `1x A100 80GB` instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset a100-80gb --web
```

If your Vast box has 2 RTX 4090 GPUs, this is the explicit version of the budget teaching preset:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 2x4090 --web
```

If your Vast box has 4 GPUs and you want to use all 4, start with this:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --web --nproc-per-node 4
```

This is the repo's `nanochat`-style speedrun path. The commands below stay available when you want to understand or modify each stage directly.

If you rent a different Vast box:

- first try the closest preset instead of editing the file
- use `--preset 8xh100` for the serious capstone run
- use `--preset 8xa100` if H100 is not available and you still want a strong multi-GPU run
- use `--preset 2x4090` for a 2-GPU midrange box
- use `--preset a100-80gb` for a single large-memory GPU
- only override further if you know why you are changing the training budget

If you need a more advanced 4-GPU or custom-hardware run, these are the main knobs:

- `--nproc-per-node N` overrides the GPU count
- `PICO_PRETRAIN_BATCH_SIZE`
- `PICO_PRETRAIN_GRAD_ACCUM`
- `PICO_SFT_BATCH_SIZE`
- `PICO_SFT_GRAD_ACCUM`

Most students should stop at `--nproc-per-node 4` and only touch the environment variables if they are tuning for throughput or fixing out-of-memory errors.

If you pass `--hf-repo-id`, the script checks Hugging Face auth before training starts.

It accepts either:

- `HF_TOKEN` exported in the shell
- an existing `hf auth login` session

If neither exists, it stops immediately with a clear error instead of waiting until the run is over.

Telemetry is optional:

- use `--report-to tensorboard` for local runs
- use `--report-to wandb` for long cloud runs
- keep `--report-to none` if you only want terminal logs

Local TensorBoard example:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --report-to tensorboard --cli
uv run tensorboard --logdir artifacts
```

Cloud W&B example:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --web \
  --nproc-per-node 4 \
  --report-to wandb \
  --run-name picollm-4x4090 \
  --wandb-project picollm
```

If you pass `--report-to wandb`, the script checks Weights & Biases auth before training starts.

It accepts either:

- `WANDB_API_KEY` exported in the shell
- an existing `wandb login` session

If you want the run in a specific W&B workspace, set that explicitly with `--wandb-entity`.
If you omit it, W&B uses the default entity from the active login or API key.

If neither exists, it stops immediately with a clear error instead of waiting until the run is already in progress.

If you want the full step-by-step setup instead of the one-command path, run:

```bash
git clone https://github.com/Montekkundan/llm.git
cd llm
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
uv sync

uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --max-texts 500000 \
  --vocab-size 32000 \
  --output-dir artifacts/picollm/tokenizer

uv run torchrun --nproc_per_node=2 -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 1024 \
  --layers 24 \
  --heads 16 \
  --hidden-size 1024 \
  --batch-size 2 \
  --grad-accum 16 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 5000 \
  --bf16

uv run torchrun --nproc_per_node=2 -m picollm.sft_full.finetune \
  --model artifacts/picollm/pretrain-run \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-run \
  --batch-size 2 \
  --grad-accum 16 \
  --learning-rate 2e-5 \
  --warmup-steps 10 \
  --save-steps 50 \
  --max-steps 150 \
  --bf16
```

Optional:

- if you want the cloud checkpoint on the Hugging Face Hub later, export `HF_TOKEN` on the machine first
- if you only want the checkpoint locally, you can skip Hugging Face completely and copy the folder back directly

If you want broader assistant behavior after the conversational pass, run one more full-SFT pass:

```bash
uv run python -m picollm.sft_full.finetune \
  --model artifacts/picollm/chat-sft-run \
  --dataset-name HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-ultrachat \
  --batch-size 2 \
  --grad-accum 16 \
  --learning-rate 1e-5 \
  --warmup-steps 200 \
  --save-steps 500 \
  --max-steps 4000 \
  --bf16
```

If the same Vast machine is still running, you can start another run there. You do not need to create a new machine every time. Before retraining, either remove the old artifacts or use a new output directory:

```bash
rm -rf artifacts/picollm/pretrain-run artifacts/picollm/chat-sft-run artifacts/picollm/tokenizer
```

This is the closer small-model version of the same idea: base pretraining first, then chat post-training on top of your own checkpoint.

When training finishes, your checkpoint folder on the Vast machine will usually be:

```bash
/root/llm/artifacts/picollm/pretrain-run
/root/llm/artifacts/picollm/chat-sft-run
```

If your repo lives somewhere else on the machine, adjust that path accordingly.

You may also see a log line like:

- `Writing model shards`

That is normal. It usually means the checkpoint is being saved into multiple files instead of one giant file. The model is still one checkpoint logically; it is just stored on disk as several shard files for easier saving and loading.

Before leaving the SSH session, confirm the final checkpoint exists:

```bash
ls -lah /root/llm/artifacts/picollm/chat-sft-run
```

If the folder is there and contains model files, you can exit SSH.

If you prefer your own corpus instead of those public datasets, create a text file on the Vast machine and pass it with repeated `--text-file` flags.

### How to swap in a different dataset

You can change the dataset, but do not change only `--dataset-name`. You also need to match the dataset schema.

Use this rule:

- base pretraining: plain text, usually `--text-column text`
- chat post-training: standard chat messages, usually `--text-column messages`
- list-of-turn dialogue data: use the dialogue column and add `--alternating-chat-roles`

Examples:

```bash
# base pretraining on general text
--dataset-name HuggingFaceFW/fineweb-edu
--dataset-config sample-10BT
--dataset-split train
--text-column text
--streaming
```

```bash
# default conversational SFT
--dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k
--dataset-split train_sft
--text-column messages
```

```bash
# broader assistant SFT
--dataset-name HuggingFaceH4/ultrachat_200k
--dataset-split train_sft
--text-column messages
```

So yes, you can swap datasets and start training, but the split and text column must match the dataset stage and format.

### Multi-GPU on Vast

For this run, multi-GPU is the normal path, not just an experiment.

Serious capstone recommendation for this course:

- `8x H100`

Use this search command first:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "H100 SXM5" \
  --num-gpus 8 \
  --gpu-ram-gb 80 \
  --reliability 0.995 \
  --limit 10
```

If the exact H100 label differs on Vast, use the closest H100 label shown in the Vast.ai console.

If you want a slower but still strong alternative, use `8x A100`.

If you want the cheaper teaching-scale run instead, use `2x RTX 4090`.

If you change hardware, you usually do not need to edit the script. Start with the closest speedrun preset instead:

- default: `2x4090`
- single large GPU: `a100-80gb`

Two GPUs on one machine:

```bash
uv run torchrun --nproc_per_node=2 -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 1024 \
  --layers 24 \
  --heads 16 \
  --hidden-size 1024 \
  --batch-size 2 \
  --grad-accum 16 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 5000 \
  --bf16
```

What changes when you move from one GPU to two:

- `--batch-size` is still per GPU
- total tokens per optimizer step increase
- training can run faster or support a larger effective batch
- the underlying objective does not change

The rough effective global batch is:

- `per_device_batch_size × num_gpus × grad_accum`

So with `--batch-size 2`, `--grad-accum 16`, and 2 GPUs, the effective global batch is `64` sequences per optimizer step.

To turn that into a rough ETA, add sequence length and live step time:

- `tokens_per_step = per_device_batch_size × num_gpus × grad_accum × block_size`
- `eta_seconds = remaining_steps × seconds_per_step`

Example from the calibrated `8xh100` capstone preset:

- `per_device_batch_size = 8`
- `num_gpus = 8`
- `grad_accum = 16`
- `block_size = 1024`

That gives:

- `tokens_per_step = 8 × 8 × 16 × 1024 = 1,048,576`

If your log shows about `2.8s/it`, the pretrain ETA is:

- `5000 × 2.8s = 14,000s`
- about `3.9 hours`

Then add chat SFT and final upload/serve startup on top of that. For the one-command capstone path, that usually means a full runtime around `4.5 to 5 hours` on a healthy `8x H100` box.

For the one-command speedrun:

- default preset: `2x4090`
- single large GPU preset: `a100-80gb`

Examples:

```bash
# course default
bash picollm/pretrain_cloud/speedrun.sh --web

# one large GPU
bash picollm/pretrain_cloud/speedrun.sh --preset a100-80gb --web
```

## 9. Reuse that checkpoint locally

Run the next commands on your local machine, not inside the Vast VM.

First, print the copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100 \
  --local-dir artifacts/picollm/chat-sft-run \
  --remote-dir /root/llm/artifacts/picollm/chat-sft-run
```

That will print `scp` and `rsync` commands you can run from your laptop. Running `vast_access` by itself does not transfer any files.

After you copy the checkpoint folder back to your machine, you can test it locally with the same serving tools:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

Web:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

If you also ran the broader assistant pass, use `artifacts/picollm/chat-sft-ultrachat` instead.

## 10. Push your own chatbot to Hugging Face Hub

If you want the final project to live on the Hub too, push the chat-SFT checkpoint:

```bash
hf auth login

uv run python -m picollm.pretrain_cloud.push_to_hub \
  --folder artifacts/picollm/chat-sft-run \
  --repo-id your-name/picollm-chat-sft
```

Then you can run the same chatbot locally from the Hub id:

```bash
uv run python -m picollm.serve.chat_cli \
  --model your-name/picollm-chat-sft \
  --device auto
```

## 11. Connect it to the Vercel AI SDK app

Start your chatbot backend:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

Then in [apps/vercel_ai_sdk_chat](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat):

```bash
npm install
cp .env.example .env.local
```

Set:

```bash
PICOLLM_BASE_URL=http://127.0.0.1:8008/v1
PICOLLM_API_KEY=local-demo-key
PICOLLM_MODEL=artifacts/picollm/chat-sft-run
```

Then run:

```bash
npm run dev
```

That gives you the full final-project story:

1. train your own model from scratch
2. chat-post-train it
3. serve it locally through an OpenAI-compatible backend
4. optionally push it to the Hub
5. connect it to a production-style browser app

So the practical cloud loop is:

1. rent the Vast instance
2. SSH into it
3. run tokenizer training, base pretraining, and full chat SFT there
4. copy `artifacts/picollm/chat-sft-run` back to your laptop
5. point `chat_cli` or `chat_web` at that local checkpoint path

When you are done with the cloud run, clean up in two places:

1. destroy the Vast instance from your local machine:

```bash
uv run python -m picollm.pretrain_cloud.vast_destroy_instance \
  --new-contract 34276100
```

2. remove the copied local checkpoint:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run
```

If you also want to remove the copied tokenizer:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run \
  --include-tokenizer
```

## 12. Related docs

- [README.md](/Users/montekkundan/Developer/ML/llm/picollm/README.md)
- [HUGGING_FACE_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/HUGGING_FACE_SETUP.md)
- [pretrain_cloud/VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)
- [sft_lora/README.md](/Users/montekkundan/Developer/ML/llm/picollm/sft_lora/README.md)
- [serve/README.md](/Users/montekkundan/Developer/ML/llm/picollm/serve/README.md)
- [../apps/vercel_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/vercel_ai_sdk_chat/README.md)
- [../apps/opentui_ai_sdk_chat/README.md](/Users/montekkundan/Developer/ML/llm/apps/opentui_ai_sdk_chat/README.md)
