# picoLLM Training Stack

This package contains the single serious training and inference path for `picollm`.

Docs:

- [docs/local_smoke.md](./docs/local_smoke.md)
- [docs/branding_identity.md](./docs/branding_identity.md)
- [docs/rerun_sft_only.md](./docs/rerun_sft_only.md)
- [docs/hosted_assets.md](./docs/hosted_assets.md)
- [docs/hf_backup_strategy.md](./docs/hf_backup_strategy.md)
- [docs/artifact_types.md](./docs/artifact_types.md)
- [docs/release_naming.md](./docs/release_naming.md)

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
export HF_ARCHIVE_REPO_ID=your-username/your-picollm-archive
export HF_UPLOAD_PRIVATE=1
```

Notes:

- `WANDB_ENTITY` is required for any non-dummy W&B run in this repo.
- `HF_TOKEN` is recommended for Hugging Face downloads and rate limits. Public datasets may still work without it.
- `uv sync --extra gpu` is the CUDA path.
- `uv sync --extra cpu` is the local CPU or macOS path.
- `speedrun.sh` is the single reference script. It now auto-detects visible CUDA GPU count, memory, capability, FA3/FP8 eligibility, batch size, activation checkpointing, and a safer attention window pattern.
- `speedrun.sh` now starts with a distributed preflight that does a small synthetic forward/backward/optimizer smoke test on the same FA3/FP8/compile stack before dataset work begins.
- On Hopper-class boxes it keeps the fast defaults. On non-Hopper CUDA boxes it automatically disables FP8, forces SDPA, and switches to `window-pattern=L` so the run is slower but materially more portable.
- Rebuild the environment with `uv sync --extra gpu` after pulling, because picoLLM now pins `torch==2.9.1` for this runtime path.
- The default SFT identity data now comes from the repo-local `picollm/accelerated/data/identity_conversations.jsonl` instead of relying on a legacy external identity file.
- The canonical identity dataset now ships with `picollm/accelerated/data/identity_conversations.manifest.json`, which records the row count, SHA-256 checksum, schema contract, and intended hosted mirror URL.
- `HF_UPLOAD_REPO_ID` is optional. If set, the speedrun uploads the final runtime artifacts to a Hugging Face model repo.
- `HF_ARCHIVE_REPO_ID` is optional. If set, the speedrun also uploads the fuller run archive to a Hugging Face dataset repo.
- `HF_UPLOAD_PRIVATE=1` keeps that repo private by default.
- `HF_PERIODIC_SYNC=1` enables the optional archive-sync loop during long runs.

Optional manual overrides if you need to pin a different configuration:

- `PICOLLM_NPROC_PER_NODE`
- `PICOLLM_DEVICE_BATCH_SIZE`
- `PICOLLM_TOTAL_BATCH_SIZE`
- `PICOLLM_ENABLE_FP8`
- `PICOLLM_ACTIVATION_CHECKPOINTING`
- `PICOLLM_WINDOW_PATTERN`
- `PICOLLM_IDENTITY_CONVERSATIONS_FILE`
- `PICOLLM_IDENTITY_CONVERSATIONS_URL`
- `PICOLLM_IDENTITY_CONVERSATIONS_MANIFEST`
- `PICOLLM_BASE_SAVE_EVERY`
- `HF_PERIODIC_SYNC`
- `HF_SYNC_INTERVAL_SECONDS`
- `HF_SYNC_MIN_FILE_AGE_SECONDS`

## Full Run

If you want the end-to-end workflow on a fresh machine, use the speedrun script:

```bash
bash picollm/accelerated/speedrun.sh cli
```

Web UI instead of CLI:

```bash
bash picollm/accelerated/speedrun.sh web
```

Smoke-test a running accelerated backend:

```bash
python scripts/deployment/smoke_test_accelerated.py
```

Print the recommended local environment commands for this machine:

```bash
python scripts/print_picollm_env.py
```

Run the standard local regression bundle for CPU, MPS, or CUDA:

```bash
python scripts/run_picollm_local_checks.py --device mps
python scripts/run_picollm_local_checks.py --device cuda
```

Run the branding smoke test against the latest SFT checkpoint:

```bash
python -m picollm.accelerated.chat.identity_smoke
```

Verify the canonical identity dataset locally, or compare it to the hosted Cloudflare mirror after upload:

```bash
python scripts/verify_identity_asset.py --local-only
python scripts/verify_identity_asset.py
```

If you want `speedrun.sh` to consume the hosted identity mirror instead of the repo-local file, set the hosted URL and keep the manifest path available so the download is verified before SFT starts:

```bash
export PICOLLM_IDENTITY_CONVERSATIONS_URL=https://assets.montek.dev/identity_conversations.jsonl
bash picollm/accelerated/speedrun.sh cli
```

Restore a published picoLLM model repo into a local artifact directory and run a one-prompt smoke test:

```bash
python scripts/restore_picollm_from_hf.py your-username/your-picollm-backup --device-type cpu
```

Or use the lighter local model-repo smoke helper that downloads only the minimal inference bundle:

```bash
python scripts/smoke_picollm_model_repo.py your-username/your-picollm-backup --device-type cpu
```

Upload the runnable picoLLM artifact set from a local `PICOLLM_BASE_DIR` into a model repo:

```bash
python scripts/upload_picollm_model_to_hf.py your-username/your-picollm-backup
```

That model repo is now inference-focused:

- it includes tokenizer files, model checkpoints, metadata, reports, and the identity file when present
- it intentionally excludes optimizer shards
- it is meant for restore + chat, not exact resume-training

The model-upload helper can also re-download the just-published repo into a fresh temp directory and run one CLI smoke prompt:

```bash
python scripts/upload_picollm_model_to_hf.py your-username/your-picollm-backup --post-upload-smoke --post-upload-smoke-device-type cpu
```

Upload the fuller run archive into a dataset repo:

```bash
python scripts/upload_picollm_archive_to_hf.py your-username/your-picollm-archive
```

That archive repo is the resume-oriented path:

- it keeps the fuller checkpoint trees, including optimizer shards when present
- it is the right place for preservation, debugging, and training-state backups
- it stays separate from the lighter inference repo on purpose

Release both repos with a consistent versioned name, and optionally mirror the latest inference bundle into a stable alias repo:

```bash
python scripts/release_picollm_to_hf.py --namespace your-username --release-name april-h200-run --latest-repo-id your-username/picollm-latest
```

Export the latest native checkpoint into a Transformers-compatible `trust_remote_code` bundle:

```bash
python scripts/export_picollm_to_transformers.py --source sft
```

Export the latest native checkpoint into GGUF format:

```bash
python scripts/export_picollm_to_gguf.py --source sft
```

The GGUF export is still architecture-specific:

- it writes a real GGUF file for picoLLM
- stock llama.cpp does not yet include a picoLLM runtime implementation
- the native picoLLM checkpoint format remains the only fully supported runtime path in this repo today

Every accelerated speedrun now writes `run_manifest.json` into `PICOLLM_BASE_DIR` before the upload step so the archive has a machine-readable record of the repo commit, torch version, chosen speedrun config, identity source, and latest base/SFT checkpoint pointers.

`speedrun.sh` is the single reference script. It goes from dataset/tokenizer work through pretraining, pretrain eval, SFT, chat eval, report generation, optional Hugging Face backup, and then opens the CLI or web chat UI.

At launch it prints the detected hardware summary and the chosen speedrun settings so you can see exactly what it decided for the current machine.

Before the heavy training work starts, `speedrun.sh` now runs `python -m picollm.accelerated.speedrun_doctor` to validate CUDA visibility, HF upload prerequisites, artifact-dir writability, free disk space, and a coarse free-VRAM check for the chosen config.

## Optional Hugging Face Backup

If `HF_UPLOAD_REPO_ID` is set, `speedrun.sh` will upload a curated set of runtime artifacts to that model repo at the end of the run:

- `tokenizer/`
- `base_checkpoints/`
- `chatsft_checkpoints/`
- `report/` if present
- `identity_conversations.jsonl` if present
- `run_manifest.json` if present
- `picollm_model_metadata.json`

It does not upload the downloaded ClimbMix parquet shards under `base_data_climbmix/`, and it now filters optimizer shards out of the inference bundle by design.

If `HF_ARCHIVE_REPO_ID` is also set, the dataset repo is the place for resume-training artifacts and fuller checkpoint history.

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
- `python -m picollm.accelerated.pretrain.train --depth=24 --target-param-data-ratio=8 --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" --total-batch-size="$PICOLLM_TOTAL_BATCH_SIZE" --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.pretrain.eval --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE"`
- `python -m picollm.accelerated.chat.sft --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" --total-batch-size="$PICOLLM_TOTAL_BATCH_SIZE" --wandb-entity "$WANDB_ENTITY"`
- `python -m picollm.accelerated.chat.eval -i sft`
- `python -m picollm.accelerated.chat.cli`
- `python -m picollm.accelerated.chat.web`
- `python -m picollm.accelerated.chat.identity_smoke`
- `python scripts/deployment/smoke_test_accelerated.py`
