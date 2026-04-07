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
  --gpu-ram-gb 24
```

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
- `new_contract` is the real numeric instance id
- `instance_api_key` is not the id and should not be passed to `vast_show_instance` or `vast_access`

Example:

```json
{
  "success": true,
  "new_contract": 34276100,
  "instance_api_key": "..."
}
```

From that output, use `34276100` as the instance id.

## 3. Check status

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --instance-id 34276100
```

Wait until the instance is `running`.

## 4. Get the SSH and copy commands

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 34276100
```

## 5. Run training on the box

After SSH:

```bash
git clone https://github.com/montekkundan/llm.git
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

The expected checkpoint location after training is usually:

```bash
/root/llm/artifacts/picollm/pretrain-run
```

If you cloned the repo into a different directory, use that path instead.

Why `wikitext` here:

- the repo does not ship a large sample corpus file
- `wikitext` is public and works out of the box with these scripts
- this is the fastest way to get a real checkpoint training on the cloud box

## 6. Bring the checkpoint back

From your laptop, ask the helper to print copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 34276100 \
  --local-dir artifacts/picollm/pretrain-run \
  --remote-dir /root/llm/artifacts/picollm/pretrain-run
```

Then run either the printed `scp` command or the printed `rsync` command.

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

For a good chat demo, do not stop at pretraining. Add SFT or LoRA after that.
