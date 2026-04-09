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
export HF_UPLOAD_REPO_ID=your-username/your-picollm-backup
export HF_UPLOAD_PRIVATE=1
```

Notes:

- `WANDB_ENTITY` is required for any non-dummy W&B run in this repo.
- `HF_TOKEN` is recommended for Hugging Face downloads and rate limits. Public datasets may still work without it.
- `speedrun.sh` is the single reference script. It clears stale flash/compile overrides and runs the accelerated path automatically when the environment supports it.
- `speedrun.sh` now starts with a distributed preflight that does a small synthetic forward/backward/optimizer smoke test on the same FA3/FP8/compile stack before dataset work begins.
- The reference speedrun uses auto FA3 selection, pretrain `--fp8`, `device-batch-size=16`, auto total batch size, and the full eval path instead of the earlier bounded proof recipe.
- Rebuild the environment with `uv sync --extra gpu` after pulling, because picoLLM now pins `torch==2.9.1` for this runtime path.
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

`speedrun.sh` is the single reference script. It goes from dataset/tokenizer work through pretraining, pretrain eval, SFT, chat eval, report generation, optional Hugging Face backup, and then opens the CLI or web chat UI.

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
- `python -m picollm.accelerated.pretrain.train --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.pretrain.eval --device-batch-size=16`
- `python -m picollm.accelerated.chat.sft --device-batch-size=16 --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.chat.eval -i sft`
- `python -m picollm.accelerated.chat.cli`
- `python -m picollm.accelerated.chat.web`
