# picoLLM Training Stack

This package contains the single serious training and inference path for `picollm`.

Default artifact root:

- `artifacts/picollm`

Override with:

- `PICOLLM_BASE_DIR=/abs/path`

## Before You Start

Export these first on the training machine:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=your-wandb-entity
export HF_TOKEN=...
export PICOLLM_FLASH_IMPL=sdpa
export PICOLLM_BASE_DIR=/abs/path/to/artifacts/picollm
export HF_UPLOAD_REPO_ID=your-username/your-picollm-backup
export HF_UPLOAD_PRIVATE=1
```

Notes:

- `WANDB_ENTITY` is required for any non-dummy W&B run in this repo.
- `HF_TOKEN` is recommended for Hugging Face downloads and rate limits. Public datasets may still work without it.
- `PICOLLM_FLASH_IMPL=sdpa` is the stable path we validated on 8x H100 80GB for full training.
- `HF_UPLOAD_REPO_ID` is optional. If set, the speedrun uploads the final runtime artifacts to a Hugging Face model repo.
- `HF_UPLOAD_PRIVATE=1` keeps that repo private by default.

## Full Run

If you want the `nanochat`-style end-to-end workflow on a fresh machine, use the speedrun script:

```bash
bash picollm/accelerated/speedrun.sh cli
```

Web UI instead of CLI:

```bash
bash picollm/accelerated/speedrun.sh web
```

This is the closest equivalent to `nanochat`'s end-to-end script in this repo, but matching `nanochat` quality is not automatic. Quality depends on the training recipe, total tokens, data mixture, and evaluation results. The script now uses the stable `SDPA + batch-size 1` path we proved on the H100 box.

## Optional Hugging Face Backup

If `HF_UPLOAD_REPO_ID` is set, `speedrun.sh` will upload a curated set of runtime artifacts to that model repo at the end of the run:

- `tokenizer/`
- `base_checkpoints/`
- `chatsft_checkpoints/`
- `report/` if present
- `identity_conversations.jsonl` if present

It does not upload the downloaded ClimbMix parquet shards under `base_data_climbmix/`.

To restore those artifacts later onto another machine:

```bash
git clone https://github.com/Montekkundan/llm
cd llm
export PICOLLM_BASE_DIR=$PWD/artifacts/picollm
export PICOLLM_FLASH_IMPL=sdpa
hf download "$HF_UPLOAD_REPO_ID" --repo-type model --local-dir "$PICOLLM_BASE_DIR"
uv sync --extra gpu
source .venv/bin/activate
python -m picollm.accelerated.chat.cli -i sft
```

Main entrypoints:

- `python -m picollm.accelerated.dataset`
- `python -m picollm.accelerated.pretrain.train_tokenizer`
- `python -m picollm.accelerated.pretrain.train --depth=24 --target-param-data-ratio=8 --device-batch-size=1 --total-batch-size=65536 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.pretrain.eval --device-batch-size=1`
- `python -m picollm.accelerated.chat.sft --device-batch-size=1 --total-batch-size=65536 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.chat.eval -i sft`
- `python -m picollm.accelerated.chat.cli`
- `python -m picollm.accelerated.chat.web`
