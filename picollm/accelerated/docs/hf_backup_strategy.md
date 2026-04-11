# HF Backup Strategy

Use two Hugging Face repos on purpose.

## Inference Model Repo

Produced by:

- `python scripts/upload_picollm_model_to_hf.py ...`

Contains:

- tokenizer
- model checkpoints
- metadata
- report
- identity dataset
- run manifest

Excludes:

- optimizer shards
- large downloaded training datasets

## Archive Dataset Repo

Produced by:

- `python scripts/upload_picollm_archive_to_hf.py ...`

Contains:

- fuller checkpoint trees
- optimizer shards when present
- reports and manifests

Use this repo for:

- preservation
- debugging
- resume-oriented backups

## Release Helper

Use:

```bash
python scripts/release_picollm_to_hf.py --namespace your-username --release-name my-run --latest-repo-id your-username/picollm-latest
```

Optional long-run sync:

```bash
export HF_PERIODIC_SYNC=1
export HF_ARCHIVE_REPO_ID=your-username/your-picollm-archive
bash picollm/accelerated/speedrun.sh
```
