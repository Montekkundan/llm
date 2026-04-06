# Vast.ai Setup

This is the fastest CLI-first path for the cloud pretraining lecture.

## Prerequisites

1. Create a Vast.ai account and add billing.
2. Add an SSH key to Vast.ai.
3. Export your API key:

```bash
export VAST_API_KEY="..."
```

Official docs:

- [Overview and quickstart](https://docs.vast.ai/api-reference/overview-and-quickstart)
- [Creating instances with the API](https://docs.vast.ai/api-reference/creating-instances-with-api)
- [Show instance API](https://docs.vast.ai/api-reference/instances/show-instance)

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

## 3. Check status

```bash
uv run python -m picollm.pretrain_cloud.vast_show_instance \
  --instance-id 12345678
```

Wait until the instance is `running`.

## 4. Get the SSH and copy commands

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --instance-id 12345678
```

## 5. Run training on the box

After SSH:

```bash
git clone https://github.com/montekkundan/llm.git
cd llm
uv sync
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --text-file data/pretrain/sample_corpus.txt \
  --output-dir artifacts/picollm/tokenizer

uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --text-file data/pretrain/sample_corpus.txt \
  --output-dir artifacts/picollm/pretrain-run \
  --max-steps 2000 \
  --bf16
```

## 6. Bring the checkpoint back

Use the command from `vast_access`, then run locally:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/pretrain-run \
  --device auto
```

For a good chat demo, do not stop at pretraining. Add SFT or LoRA after that.
