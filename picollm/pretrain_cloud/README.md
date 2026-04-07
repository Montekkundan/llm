# `pretrain_cloud/`

This path is for the "we trained our own model" part of the repo.

Use it when you want a real from-scratch chatbot path:

- train a tokenizer
- pretrain a base GPT-style model from scratch on general text
- full-SFT that checkpoint on conversational data
- pull the final checkpoint back to a laptop for inference

## Recommended use

Use a rented GPU for this track.

- Vast.ai is the simplest low-cost route for ad hoc GPU rentals: [Instances overview](https://docs.vast.ai/documentation/instances)
- Hugging Face Jobs is cleaner when you want a managed training workflow and Hub integration: [Jobs overview](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- If you want a CLI-first Vast workflow in this repo, use [VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)

This path is best for:

- showing what real base pretraining looks like
- showing why chat behavior is usually a second stage, not the first stage
- producing a checkpoint that feels like your own small chatbot, not just a tiny language model

## Serious Capstone Path

For the serious capstone path in this course:

- to get a real from-scratch conversational chatbot quickly, use the serious cloud capstone path on `8x H100`
- expect about `4 hours` and about `$100` for the full run
- this follows the same general idea as `nanochat`'s serious cloud speedrun path
- if you do not want to pay for that run, use the shared Hugging Face checkpoint and still complete the rest of the workflow

## One-command speedrun

If you want a single command that runs tokenizer training, base pretraining, full chat SFT, and then opens an interface at the end, start with:

```bash
git clone https://github.com/Montekkundan/llm.git
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh
```

That ends in the CLI by default.

For this repo, the default preset inside the script is:

- `2x4090`

That is the budget teaching preset. For the serious capstone run, use `--preset 8xh100`.

Serious capstone run:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 8xh100 --web
```

Serious capstone run with Hub upload and W&B telemetry:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --preset 8xh100 \
  --web \
  --hf-repo-id your-name/picollm-chat-sft \
  --hf-token "$HF_TOKEN" \
  --report-to wandb \
  --run-name picollm-capstone-8xh100 \
  --wandb-project picollm \
  --wandb-api-key "$WANDB_API_KEY"
```

If you want the web UI instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --web
```

If you also want the final chatbot pushed to the Hub at the end:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --web \
  --hf-repo-id your-name/picollm-chat-sft
```

If you rent `8x A100` instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset 8xa100 --web
```

If you rent `1x A100 80GB` instead:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --preset a100-80gb --web
```

If your box has 4 GPUs and you want to use all 4, start with:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --web --nproc-per-node 4
```

This script is the repo's equivalent of a `nanochat`-style speedrun: one command, same stages, less manual typing. Use the step-by-step commands below when you want to understand each stage directly.

If you rent a different machine:

- start with the closest preset instead of editing the file
- `8xh100` for the serious capstone run
- `8xa100` for a slightly slower but still strong cloud run
- `2x4090` for two midrange GPUs
- `a100-80gb` for one large-memory GPU

If you need to tune a 4-GPU or custom-hardware run without editing the file, use:

- `--nproc-per-node N`
- `PICO_PRETRAIN_BATCH_SIZE`
- `PICO_PRETRAIN_GRAD_ACCUM`
- `PICO_SFT_BATCH_SIZE`
- `PICO_SFT_GRAD_ACCUM`

Most students should stop at `--nproc-per-node 4` and only touch the environment variables if they are tuning for throughput or fixing out-of-memory errors.

If you pass `--hf-repo-id`, the script will check Hugging Face auth before training starts.

It accepts either:

- `HF_TOKEN` exported in the shell
- an existing `hf auth login` session

If neither exists, the script stops immediately with an error instead of waiting until the end of the run.

## Telemetry

Telemetry is optional.

- use `--report-to tensorboard` when you are running locally and want a simple local dashboard
- use `--report-to wandb` when you are running a long cloud job and want a hosted dashboard
- keep `--report-to none` if you just want the terminal logs

Local TensorBoard example:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh --report-to tensorboard --cli
```

Then open TensorBoard from the same machine:

```bash
uv run tensorboard --logdir artifacts
```

Cloud W&B example:

```bash
cd ~/llm
bash picollm/pretrain_cloud/speedrun.sh \
  --web \
  --nproc-per-node 4 \
  --report-to wandb \
  --run-name picollm-4x4090 \
  --wandb-project picollm
```

If you pass `--report-to wandb`, the script checks Weights & Biases auth before training starts.

It accepts either:

- `WANDB_API_KEY` exported in the shell
- an existing `wandb login` session

If neither exists, the script stops immediately with an error instead of waiting until the run is already in progress.

## 1. Train a tokenizer

For a serious run, train the tokenizer on a large general-text corpus. You do not need the full corpus just to learn the tokenizer; sampling a large stream is normal.

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --max-texts 500000 \
  --vocab-size 32000 \
  --output-dir artifacts/picollm/tokenizer
```

You can also train from your own local text files:

```bash
uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --text-file your_corpus.txt \
  --output-dir artifacts/picollm/tokenizer
```

## 2. Base pretrain from scratch

Use general text here, not chat SFT data. This is where the model learns language before it learns dialogue behavior.

Recommended base recipe:

- dataset: `HuggingFaceFW/fineweb-edu`
- config: `sample-10BT`
- model size: roughly GPT-2 Medium class
- sequence length: `1024`
- training mode: `torchrun` on more than one GPU

Two RTX 4090s:

```bash
uv run torchrun --nproc_per_node=2 -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 1024 \
  --layers 24 \
  --heads 16 \
  --hidden-size 1024 \
  --batch-size 2 \
  --grad-accum 16 \
  --warmup-steps 1000 \
  --save-steps 5000 \
  --max-steps 50000 \
  --bf16
```

One A100 80GB:

```bash
uv run python -m picollm.pretrain_cloud.train \
  --tokenizer-path artifacts/picollm/tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --output-dir artifacts/picollm/pretrain-run \
  --block-size 1024 \
  --layers 24 \
  --heads 16 \
  --hidden-size 1024 \
  --batch-size 4 \
  --grad-accum 16 \
  --warmup-steps 1000 \
  --save-steps 5000 \
  --max-steps 50000 \
  --bf16
```

Why this is different from the older tiny recipes:

- the loader now supports `--streaming`, so larger corpora do not get materialized into Python lists first
- the trainer now packs text into `block-size` chunks instead of wasting short rows
- the dataset is general text, which is the right first stage for a model you want to post-train into a chatbot

If the same cloud machine is still running, you can start another training run there. Either remove the old artifacts first or write to a new output directory:

```bash
rm -rf artifacts/picollm/pretrain-run artifacts/picollm/chat-sft-run artifacts/picollm/tokenizer
```

## 3. Full chat SFT on your own checkpoint

After the base checkpoint is trained, run full SFT on top of that checkpoint. This is still your own model. You are not switching to Qwen or LoRA here.

Default conversational post-train dataset:

- `HuggingFaceTB/everyday-conversations-llama3.1-2k`

This dataset is small, clean, and strongly conversational. It is a better default if you want the model to answer simple prompts like `hello` or `how are you?` in a natural way.

Single GPU:

```bash
uv run python -m picollm.sft_full.finetune \
  --model artifacts/picollm/pretrain-run \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-run \
  --batch-size 4 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --save-steps 250 \
  --max-steps 1500 \
  --bf16
```

Two GPUs:

```bash
uv run torchrun --nproc_per_node=2 -m picollm.sft_full.finetune \
  --model artifacts/picollm/pretrain-run \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-run \
  --batch-size 4 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --save-steps 250 \
  --max-steps 1500 \
  --bf16
```

If you want broader assistant behavior after that, run a second SFT pass on `HuggingFaceH4/ultrachat_200k`:

```bash
uv run python -m picollm.sft_full.finetune \
  --model artifacts/picollm/chat-sft-run \
  --dataset-name HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir artifacts/picollm/chat-sft-ultrachat \
  --batch-size 2 \
  --grad-accum 16 \
  --learning-rate 1e-5 \
  --warmup-steps 200 \
  --save-steps 500 \
  --max-steps 4000 \
  --bf16
```

Use the final conversational checkpoint locally with:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

Or, if you ran the broader assistant pass:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-ultrachat \
  --device auto
```

## 4. Evaluate the checkpoint

Base model evaluation:

```bash
uv run python -m picollm.pretrain_cloud.eval \
  --model artifacts/picollm/pretrain-run \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --sample-prompt "Once upon a time" \
  --output artifacts/picollm/pretrain_eval.json
```

Chat-check after post-training:

```bash
uv run python -m picollm.pretrain_cloud.eval \
  --model artifacts/picollm/chat-sft-run \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split test_sft \
  --text-column messages \
  --sample-prompt "hello" \
  --output artifacts/picollm/chat_eval.json
```

## 5. How to swap in a different dataset

You can change datasets, but you must match the dataset schema to the stage.

Use this rule:

- base pretraining: plain text, usually `--text-column text`
- chat post-training: standard chat messages, usually `--text-column messages`
- list-of-turn dialogue data: use the dialogue column and add `--alternating-chat-roles`

Examples:

```bash
# base pretraining on general text
--dataset-name HuggingFaceFW/fineweb-edu
--dataset-config sample-10BT
--dataset-split train
--text-column text
--streaming
```

```bash
# chat post-training with standard messages
--dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k
--dataset-split train_sft
--text-column messages
```

```bash
# broader assistant post-training
--dataset-name HuggingFaceH4/ultrachat_200k
--dataset-split train_sft
--text-column messages
```

```bash
# alternating dialogue turns
--dataset-name some-dialogue-dataset
--dataset-split train
--text-column dialog
--alternating-chat-roles
```

## 6. Vast.ai helper scripts

Default recommendation for this course:

- `2x RTX 4090`

That is the default because it is usually the best budget/performance path for a serious small-model run.

Use this search command:

```bash
uv run python -m picollm.pretrain_cloud.vast_search_offers \
  --gpu-name "RTX 4090" \
  --num-gpus 2 \
  --gpu-ram-gb 24 \
  --reliability 0.995 \
  --limit 10
```

That is the one students should use by default.

If you want, you can choose a different GPU or a single larger GPU in the Vast.ai console. The simplest alternative is `1x A100 80GB`.

If you change hardware, you usually do not need to edit `speedrun.sh`. Start with the closest preset instead:

- default: `2x4090`
- single large GPU: `a100-80gb`

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

Print SSH and copy commands:

```bash
uv run python -m picollm.pretrain_cloud.vast_access \
  --new-contract 34276100
```

`vast_access` only prints the commands. It does not execute SSH, `scp`, or `rsync` for you.

## 7. Pull weights back to a laptop

The normal flow is:

1. create the Vast instance
2. use `new_contract` as the contract value
3. SSH into the machine
4. run tokenizer training, base pretraining, and full chat SFT there
5. verify the remote checkpoint exists
6. exit SSH
7. on your laptop, run the `scp` or `rsync` command printed by `vast_access` to copy the final checkpoint back

For the default conversational path, copy:

- `artifacts/picollm/chat-sft-run`

If you ran the broader assistant pass, copy:

- `artifacts/picollm/chat-sft-ultrachat`

Mac, Linux, and Windows can all use the same checkpoint folder with `transformers`.

Example local chat after you copied the folder back:

```bash
uv run python -m picollm.serve.chat_cli \
  --model artifacts/picollm/chat-sft-run \
  --device auto
```

## 8. Push weights to Hugging Face Hub

Login first:

```bash
uv run huggingface-cli login
```

Then push:

```bash
uv run python -m picollm.pretrain_cloud.push_to_hub \
  --folder artifacts/picollm/chat-sft-run \
  --repo-id your-name/picollm-chat-sft
```

Official Hub docs:

- [Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Download model snapshots](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Token setup and auth](https://huggingface.co/docs/hub/main/en/security-tokens)

## 9. Clean up after the run

Destroy the Vast instance from your local machine:

```bash
uv run python -m picollm.pretrain_cloud.vast_destroy_instance \
  --new-contract 34276100
```

Add `--yes` if you want to skip the confirmation prompt.

Remove the copied local checkpoint:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run
```

If you also want to remove the copied tokenizer:

```bash
uv run python -m picollm.pretrain_cloud.cleanup_local_artifacts \
  --checkpoint-dir artifacts/picollm/chat-sft-run \
  --include-tokenizer
```

Device rules in this repo:

- use `--bf16` on CUDA when the GPU supports it
- use `--device auto` for local serving unless you need to force a specific backend
- for multi-GPU training, launch the same Python module under `torchrun`
