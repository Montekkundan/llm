# Vast.ai Setup

This is the fastest CLI-first path for the cloud pretraining lecture.

## Prerequisites

1. Create a Vast.ai account and add billing.
2. Add an SSH key to Vast.ai before creating the instance.
3. Export your API key:

```bash
export VAST_API_KEY="..."
```

Official docs:

- [Overview and quickstart](https://docs.vast.ai/api-reference/overview-and-quickstart)
- [Creating instances with the API](https://docs.vast.ai/api-reference/creating-instances-with-api)
- [SSH connection](https://docs.vast.ai/documentation/instances/connect/ssh)
- [Show instance API](https://docs.vast.ai/api-reference/instances/show-instance)

## SSH key setup

Vast instances use SSH keys, not passwords.

Important:

- add your SSH key to Vast.ai before creating the instance
- account-level SSH keys are automatically added only to new instances
- if you add the key after the instance already exists, that running instance will not automatically receive it
- in that case, either recreate the instance or add the key through Vast's instance-specific SSH flow

### macOS and Linux

Generate a key:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Then print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the full output and paste it into Vast.ai under `Manage Keys` -> `SSH Keys`.

If you already have an SSH key, inspect it with:

```bash
ls ~/.ssh
cat ~/.ssh/id_ed25519.pub
```

### Windows PowerShell

Generate a key:

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Then print the public key:

```powershell
Get-Content $HOME\.ssh\id_ed25519.pub
```

Copy the full output and paste it into Vast.ai under `Manage Keys` -> `SSH Keys`.

### If SSH says `Permission denied (publickey)`

Check these first:

1. confirm the public key was added to Vast.ai
2. confirm the instance was created after the key was added
3. confirm the private key on your machine matches the uploaded public key
4. on macOS/Linux, ensure the private key permissions are correct:

```bash
chmod 600 ~/.ssh/id_ed25519
```

5. if you added the key after the instance was already created, recreate the instance or use Vast's instance-specific key flow

### If SSH closes immediately after login on macOS

One real-world issue on macOS is that some terminals, including Ghostty, may send:

- `TERM=xterm-ghostty`

Some minimal Ubuntu images on Vast do not have terminfo for that terminal type. In that case you may authenticate successfully and then get disconnected immediately with a message like:

- `missing or unsuitable terminal: xterm-ghostty`

Use this instead:

```bash
TERM=xterm-256color ssh -p 36100 root@ssh1.vast.ai
```

If needed, use the simpler fallback:

```bash
TERM=xterm ssh -p 36100 root@ssh1.vast.ai
```

If you want this fix to apply automatically for that host, add this to `~/.ssh/config` on your Mac:

```sshconfig
Host ssh1.vast.ai
  SetEnv TERM=xterm-256color
```

## 1. Search offers

For the serious capstone path in this course:

- to get a real from-scratch conversational chatbot quickly, use the serious cloud capstone path on `8x H100`
- expect about `4 hours` and about `$100` for the full run
- this follows the same general idea as `nanochat`'s serious cloud speedrun path
- if you do not want to pay for that run, use the shared Hugging Face checkpoint and still complete the rest of the workflow

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

If you want a slower but still strong alternative, use `8x A100`:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "A100 SXM4" \
  --num-gpus 8 \
  --gpu-ram-gb 80 \
  --reliability 0.995 \
  --limit 10
```

If you want the cheaper teaching-scale run instead, use:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 2 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

If you want, you can also choose a single larger GPU in the Vast.ai console. The simplest single-GPU alternative is `1x A100 80GB`.

Pick one `id` from the output.

## 2. Create the instance

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --label picollm-train \
  --disk-gb 80
```

If you want a Hugging Face token available on the box:

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --hf-token "$HF_TOKEN"
```

If you also want Weights & Biases available on the box:

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --hf-token "$HF_TOKEN" \
  --wandb-api-key "$WANDB_API_KEY" \
  --wandb-entity montekkundan
```

Important:

- the create command returns both `new_contract` and `instance_api_key`
- `new_contract` is the real numeric contract value
- `instance_api_key` is not the contract value and should not be passed to `vast_show_instance` or `vast_access`

Example:

```json
{
  "success": true,
  "new_contract": 34276100,
  "instance_api_key": "..."
}
```

From that output, use `34276100` as the `new_contract` value.

## 3. Check status

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --new-contract 34276100
```

Wait until the instance is `running`.

## 4. Get the SSH and copy commands

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100
```

`vast_access` only prints the SSH and copy commands. It does not connect or copy anything by itself.

## 5. Run training on the box

After SSH:

```bash
git clone https://github.com/montekkundan/llm.git
cd llm
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
uv sync
```

If you want hosted telemetry or Hub upload, do the auth step here before you start the run:

```bash
wandb login
hf auth login
```

If you already passed `--hf-token` or `--wandb-api-key` when creating the instance, you can skip those login commands.

If you want the one-command path instead of running each stage manually, start with:

```bash
git clone https://github.com/Montekkundan/llm.git
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh
```

That ends in the CLI by default and assumes the `2x4090` budget teaching preset.

If you want the serious capstone run instead, use:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 8xh100 --web
```

If you want the serious capstone run with Hub upload and W&B telemetry, use:

```bash
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

If you want the web UI at the end:

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

If your Vast box has 4 GPUs and you want to use all 4, start with:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --web --nproc-per-node 4
```

This script follows the same high-level idea as `nanochat`'s `runs/speedrun.sh`: tokenizer, base train, chat SFT, then immediate interaction.

If you rent a different Vast option:

- first try the closest preset instead of editing the file
- use `--preset 8xh100` for the serious capstone run
- use `--preset 8xa100` if H100 is unavailable and you still want a strong multi-GPU run
- use `--preset 2x4090` for a 2-GPU midrange box
- use `--preset a100-80gb` for a single large-memory GPU

If you still need to tune a 4-GPU or custom-hardware run without editing the file, use:

- `--nproc-per-node N`
- `PICO_PRETRAIN_BATCH_SIZE`
- `PICO_PRETRAIN_GRAD_ACCUM`
- `PICO_SFT_BATCH_SIZE`
- `PICO_SFT_GRAD_ACCUM`

Most students should stop at `--nproc-per-node 4` and only touch the environment variables if they are tuning for throughput or fixing out-of-memory errors.

If you pass `--hf-repo-id`, the script checks Hugging Face auth before training starts.

It accepts either:

- `HF_TOKEN` exported in the shell
- an existing `hf auth login` session

If neither exists, it stops immediately with a clear error.

Telemetry is optional:

- use `--report-to tensorboard` for local runs
- use `--report-to wandb` for long cloud runs
- keep `--report-to none` if you only want terminal logs

Example cloud run with Weights & Biases:

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

If you want the step-by-step path instead, keep going:

```bash

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
  --warmup-steps 1000 \
  --save-steps 5000 \
  --max-steps 50000 \
  --bf16
uv run torchrun --nproc_per_node=2 -m picollm.sft_full.finetune \
  --model artifacts/picollm/pretrain-run \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-run \
  --batch-size 4 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --save-steps 250 \
  --max-steps 1500 \
  --bf16
```

What this is doing:

- stage 1 learns language from general text
- stage 2 teaches the already-trained checkpoint to answer like a conversational chatbot
- this is still your own model, not a pretrained Qwen checkpoint

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

If you only have one large GPU, the same commands still work. Lower `--batch-size` or increase `--grad-accum` if memory is tight.

The expected checkpoint locations after training are usually:

```bash
/root/llm/artifacts/picollm/pretrain-run
/root/llm/artifacts/picollm/chat-sft-run
```

If you cloned the repo into a different directory, use that path instead.

You may also see a log line like:

- `Writing model shards`

That is normal. It usually means the checkpoint is being saved into multiple files rather than one large file. The training job is not doing anything strange there; it is just writing the checkpoint in shard form on disk.

Before leaving SSH, confirm the final checkpoint exists:

```bash
ls -lah /root/llm/artifacts/picollm/chat-sft-run
```

If the folder is there and contains model files, you can exit the SSH session.

## Dataset swaps

You can change datasets, but the stage matters.

Use this rule:

- base pretraining: plain text, usually `--text-column text`
- chat post-training: standard chat messages, usually `--text-column messages`
- list-of-turn dialogue dataset: use the dialogue column and add `--alternating-chat-roles`

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

This keeps the from-scratch path close to how modern small chat models are usually built: general-text base pretraining first, then chat post-training.

## 6. Bring the checkpoint back

Run the next commands on your laptop or desktop, not inside the Vast VM.

From your laptop, ask the helper to print copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100 \
  --local-dir artifacts/picollm/chat-sft-run \
  --remote-dir /root/llm/artifacts/picollm/chat-sft-run
```

Then run either the printed `scp` command or the printed `rsync` command. Running `vast_access` alone does not transfer any files.

After the checkpoint folder is back on your machine, run it locally:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

Or in the web UI:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

If you ran the broader assistant pass, replace `chat-sft-run` with `chat-sft-ultrachat`.

## 7. Clean up after the run

When you are done, destroy the Vast instance from your local machine:

```bash
uv run python -m picollm.pretrain_cloud.vast_destroy_instance \
  --new-contract 34276100
```

Add `--yes` to skip the confirmation prompt.

If you also want to clear the copied local files:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run
```

Or remove both the local checkpoint and tokenizer:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run \
  --include-tokenizer
```
