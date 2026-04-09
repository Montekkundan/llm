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
export PICOLLM_BASE_DIR=/abs/path/to/artifacts/picollm
export PICOLLM_ACTIVATION_CHECKPOINTING=0
export PICOLLM_PRETRAIN_ACTIVATION_CHECKPOINTING=0
export PICOLLM_ENABLE_FP8=0
export PICOLLM_DEVICE_BATCH_SIZE=1
export PICOLLM_TOTAL_BATCH_SIZE=65536
export PICOLLM_TRAIN_LOSS_CHUNK_ROWS=4
export HF_UPLOAD_REPO_ID=your-username/your-picollm-backup
export HF_UPLOAD_PRIVATE=1
```

Notes:

- `WANDB_ENTITY` is required for any non-dummy W&B run in this repo.
- `HF_TOKEN` is recommended for Hugging Face downloads and rate limits. Public datasets may still work without it.
- `speedrun.sh` defaults to the proven stable path by setting `PICOLLM_FLASH_IMPL=sdpa`.
- `speedrun.sh` pins its stable training values internally, so stale shell exports do not silently change the reference recipe.
- `PICOLLM_ACTIVATION_CHECKPOINTING=0` keeps checkpointing off unless you explicitly opt into it.
- `PICOLLM_ENABLE_FP8=0` keeps the stable reference script off the experimental FP8 path.
- `PICOLLM_DEVICE_BATCH_SIZE=1` is the current stable default on the proven H100 route.
- `PICOLLM_TOTAL_BATCH_SIZE=65536` matches the stable end-to-end path we already validated.
- `PICOLLM_TRAIN_LOSS_CHUNK_ROWS=4` reduces the LM-head loss projection peak memory.
- `HF_UPLOAD_REPO_ID` is optional. If set, the speedrun uploads the final runtime artifacts to a Hugging Face model repo.
- `HF_UPLOAD_PRIVATE=1` keeps that repo private by default.

## Full Run

If you want the end-to-end workflow on a fresh machine, use the speedrun script:

```bash
bash picollm/accelerated/speedrun.sh cli
```

Web UI instead of CLI:

```bash
bash picollm/accelerated/speedrun.sh web
```

`speedrun.sh` is the stable reference script. It uses the conservative `sdpa` path with `device-batch-size=1`, `total-batch-size=65536`, bounded base eval (`sample,core`), and bounded chat eval (`max_problems=8`).

If you want to experiment with the aggressive FA3/FP8 recipe, use:

```bash
bash picollm/accelerated/speedrun_fast.sh cli
```

That fast script is explicitly experimental until it completes a full run cleanly.

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
hf download "$HF_UPLOAD_REPO_ID" --repo-type model --local-dir "$PICOLLM_BASE_DIR"
uv sync --extra gpu
source .venv/bin/activate
python -m picollm.accelerated.chat.cli -i sft
```

Main entrypoints:

- `python -m picollm.accelerated.dataset`
- `python -m picollm.accelerated.pretrain.train_tokenizer`
- `python -m picollm.accelerated.pretrain.train --depth=24 --target-param-data-ratio=8 --device-batch-size=1 --total-batch-size=65536 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.pretrain.eval --eval sample,core --max-per-task=8 --device-batch-size=1`
- `python -m picollm.accelerated.chat.sft --device-batch-size=1 --total-batch-size=65536 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.chat.eval -i sft`
- `python -m picollm.accelerated.chat.cli`
- `python -m picollm.accelerated.chat.web`
