# `pretrain_cloud/`

This path is for the "we trained our own model" part of the repo.

Use it when you want a full pretraining path:

- train a tokenizer
- pretrain a GPT-style model from scratch
- evaluate the checkpoint
- push the weights to Hugging Face Hub
- pull the same weights back to a laptop for inference

## Recommended use

Use a rented GPU for this track.

- Vast.ai is the simplest low-cost route for ad hoc GPU rentals: [Instances overview](https://docs.vast.ai/documentation/instances)
- Hugging Face Jobs is cleaner when you want a managed training workflow and Hub integration: [Jobs overview](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- If you want a CLI-first Vast workflow in this repo, use [VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)

This path is best for:

- showing what full pretraining looks like
- showing why cloud GPUs matter
- producing a checkpoint you can later download and run locally

## 1. Train a tokenizer

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name daily_dialog \
  --dataset-split train \
  --text-column dialog \
  --alternating-chat-roles \
  --vocab-size 16000 \
  --output-dir artifacts/picollm/tokenizer
```

In this repo, `daily_dialog` is a friendly alias. Under the hood the loader resolves it to a Hub-hosted mirror that works with current `datasets` releases, so you can keep using the shorter name in commands.

You can also train from your own local text files:

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --text-file your_corpus.txt \
  --output-dir artifacts/picollm/tokenizer
```

## 2. Pretrain in the cloud

```bash
uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name daily_dialog \
  --dataset-split train \
  --text-column dialog \
  --alternating-chat-roles \
  --output-dir artifacts/picollm/pretrain-run \
  --layers 8 \
  --heads 8 \
  --hidden-size 512 \
  --block-size 256 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 8000
```

On CUDA machines, add `--bf16` when supported.

If the same cloud machine is still running, you can start another training run there. Either remove the old artifacts first or write to a new output directory:

```bash
rm -rf artifacts/picollm/pretrain-run artifacts/picollm/tokenizer
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
  --layers 12 \
  --heads 12 \
  --hidden-size 768 \
  --block-size 256 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-steps 500 \
  --save-steps 1000 \
  --max-steps 12000
```

## Vast.ai helper scripts

Search offers:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 1 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

Start with `--num-gpus 1` if you want the simplest setup. The current script also supports multi-GPU via `torchrun` when you want to experiment further.

Create instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --label picollm-train
```

The create command returns a JSON payload. Use `new_contract` from that payload as the contract value for the next steps.

Show instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --new-contract 34276100
```

Print SSH / copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100
```

`vast_access` only prints the commands. It does not execute SSH, `scp`, or `rsync` for you.

## 3. Evaluate the checkpoint

```bash
uv run python -m picollm.pretrain_cloud.eval \
  --model artifacts/picollm/pretrain-run \
  --dataset-name daily_dialog \
  --dataset-split validation \
  --text-column dialog \
  --alternating-chat-roles \
  --sample-prompt "hi" \
  --output artifacts/picollm/pretrain_eval.json
```

## 4. Push weights to Hugging Face Hub

Login first:

```bash
uv run huggingface-cli login
```

Then push:

```bash
uv run python -m picollm.pretrain_cloud.push_to_hub \
  --folder artifacts/picollm/pretrain-run \
  --repo-id your-name/picollm-pretrain
```

Official Hub docs:

- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Download model snapshots](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Token setup and auth](https://huggingface.co/docs/hub/main/en/security-tokens)

## 5. Pull weights back to a laptop

The normal flow is:

1. create the Vast instance
2. use `new_contract` as the contract value
3. SSH into the machine
4. run tokenizer training and pretraining there
5. verify the remote checkpoint exists
6. exit SSH
7. on your laptop, run the `scp` or `rsync` command printed by `vast_access` to copy `artifacts/picollm/pretrain-run` back

Mac, Linux, and Windows can all use the same checkpoint folder with `transformers`.

Example local chat after you copied the folder back:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

For the default conversational path, use `daily_dialog` with `--alternating-chat-roles`. If you want a story-style tiny model instead, use `TinyStories`.

## 6. Clean up after the run

Destroy the Vast instance from your local machine:

```bash
uv run python -m picollm.pretrain_cloud.vast_destroy_instance \
  --new-contract 34276100
```

Add `--yes` if you want to skip the confirmation prompt.

Remove the copied local checkpoint:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts
```

If you also want to remove the copied tokenizer:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --include-tokenizer
```

Device rules in this repo:

- `cuda`: NVIDIA GPUs
- `mps`: Apple Silicon
- `cpu`: fallback on any machine
- `auto`: choose the best available device

## Cloud note

This path is intentionally simpler than nanochat.

It does not try to teach:

- FP8 kernels
- custom fused optimizers
- distributed sharding stacks
- aggressive throughput tuning

Those are real topics, but they belong after the concepts are already clear.
