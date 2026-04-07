# `pretrain_cloud/`

This path is for the "we trained our own model" part of the course.

Use it when you want a full pretraining story:

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

For the lecture, this path is best for:

- showing what full pretraining looks like
- showing why cloud GPUs matter
- producing a checkpoint that students can later download and run locally

## 1. Train a tokenizer

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split train \
  --text-column text \
  --output-dir artifacts/picollm/tokenizer
```

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
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split train \
  --text-column text \
  --output-dir artifacts/picollm/pretrain-run \
  --layers 8 \
  --heads 8 \
  --hidden-size 512 \
  --block-size 512 \
  --batch-size 2 \
  --grad-accum 16 \
  --max-steps 2000
```

On CUDA machines, add `--bf16` when supported.

## Vast.ai helper scripts

Search offers:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 1 \
  --gpu-ram-gb 24
```

Create instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_create_instance \
  --offer-id 12345678 \
  --label picollm-train
```

The create command returns a JSON payload. Use `new_contract` from that payload as the instance id for the next steps.

Show instance:

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --instance-id 34276100
```

Print SSH / copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 34276100
```

## 3. Evaluate the checkpoint

```bash
uv run python -m picollm.pretrain_cloud.eval \
  --model artifacts/picollm/pretrain-run \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split validation \
  --text-column text \
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
2. use `new_contract` as the instance id
3. SSH into the machine
4. run tokenizer training and pretraining there
5. copy `artifacts/picollm/pretrain-run` back to your laptop with the `scp` or `rsync` command printed by `vast_access`

Mac, Linux, and Windows can all use the same checkpoint folder with `transformers`.

Example local chat:

```bash
uv run python -m picollm.serve.chat_cli \
  --model your-name/picollm-pretrain \
  --device auto
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
