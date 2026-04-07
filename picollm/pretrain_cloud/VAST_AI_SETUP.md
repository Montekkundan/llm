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

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 1 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

If you want to experiment with multi-GPU, increase `--num-gpus`:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 2 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

The training script itself can run under `torchrun`, so multi-GPU is available when you want to try it.

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
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --text-column messages \
  --vocab-size 16000 \
  --output-dir artifacts/picollm/tokenizer

uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 256 \
  --layers 8 \
  --heads 8 \
  --hidden-size 512 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 8000 \
  --bf16
```

If you want a small coherent story model instead, use `TinyStories`:

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name roneneldan/TinyStories \
  --dataset-split train \
  --text-column text \
  --vocab-size 32000 \
  --output-dir artifacts/picollm/tokenizer

uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name roneneldan/TinyStories \
  --dataset-split train \
  --text-column text \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 256 \
  --layers 12 \
  --heads 12 \
  --hidden-size 768 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 12000 \
  --bf16
```

If the same Vast machine is still running, you can start another run there. You do not need to create a new machine every time. Before retraining, either remove the old artifacts or use a new output directory:

```bash
rm -rf artifacts/picollm/pretrain-run artifacts/picollm/tokenizer
```

If your Vast instance has multiple GPUs, launch the same training script with `torchrun` instead of plain `python`:

```bash
uv run torchrun --nproc_per_node=2 -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 256 \
  --layers 8 \
  --heads 8 \
  --hidden-size 512 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 8000 \
  --bf16
```

What changes:

- `--batch-size` stays per GPU
- total effective batch increases with the number of GPUs
- throughput usually improves
- the model and objective stay the same

Rough effective global batch:

- `per_device_batch_size × num_gpus × grad_accum`

The expected checkpoint location after training is usually:

```bash
/root/llm/artifacts/picollm/pretrain-run
```

If you cloned the repo into a different directory, use that path instead.

You may also see a log line like:

- `Writing model shards`

That is normal. It usually means the checkpoint is being saved into multiple files rather than one large file. The training job is not doing anything strange there; it is just writing the checkpoint in shard form on disk.

Before leaving SSH, confirm the checkpoint exists:

```bash
ls -lah /root/llm/artifacts/picollm/pretrain-run
```

If the folder is there and contains model files, you can exit the SSH session.

Why these datasets:

- `HuggingFaceH4/ultrachat_200k` is the default if you want a small conversational model with an industry-standard `messages` format
- `roneneldan/TinyStories` is better if you want a small coherent story model
- both are public and work out of the box with these scripts

## 6. Bring the checkpoint back

Run the next commands on your laptop or desktop, not inside the Vast VM.

From your laptop, ask the helper to print copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100 \
  --local-dir artifacts/picollm/pretrain-run \
  --remote-dir /root/llm/artifacts/picollm/pretrain-run
```

Then run either the printed `scp` command or the printed `rsync` command. Running `vast_access` alone does not transfer any files.

After the checkpoint folder is back on your machine, run it locally:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

Or in the web UI:

```bash
uv run python -m picollm.serve.chat_web \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

For a more usable tiny model, prefer the `HuggingFaceH4/ultrachat_200k` or `TinyStories` commands above instead of the older `wikitext` path.

## 7. Clean up after the run

When you are done, destroy the Vast instance from your local machine:

```bash
uv run python -m picollm.pretrain_cloud.vast_destroy_instance \
  --new-contract 34276100
```

Add `--yes` to skip the confirmation prompt.

If you also want to clear the copied local files:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts
```

Or remove both the local checkpoint and tokenizer:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --include-tokenizer
```
