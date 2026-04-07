# picoLLM

`picollm/` is the serious-model track in this repo.

Start with [RUNBOOK.md](/Users/montekkundan/Developer/ML/llm/picollm/RUNBOOK.md) if you want the full execution flow in one place.

The rest of this repo teaches how language models work from scratch.

This folder is the practical bridge to a real chatbot:

- cloud pretraining when you want your own checkpoint
- LoRA fine-tuning when you want a realistic tuning workflow
- local serving on Mac, Windows, or Linux
- frontend integration through a Vercel AI SDK app

## Why this exists next to the from-scratch code

The tiny model in the notebooks is good for explanation.
It is not good enough to act like a modern chatbot.

So the repo has two layers:

- `notebooks/` and `scripts/`: concept-first runnable code
- `picollm/`: serious model workflow for the final demo

## Should we copy nanochat's FP8 path?

No, not for this folder.

`nanochat/fp8.py` exists because nanochat is pushing much harder on throughput and memory efficiency on supported GPU stacks. That is a systems-optimization choice, not a conceptual requirement.

For this repo:

- keep `picollm/` simple
- use standard `transformers` + `peft` + `trl`
- focus on portable training, serving, and evaluation
- move to nanochat later if you want deeper systems optimization work

In other words: yes, nanochat is much more optimized. That is exactly why it is a good "what to study next" repo after the concepts are already clear.

## Recommended flow

### Path A: Serious chatbot demo on a laptop

Use a small instruct model:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Then run:

```bash
uv run python -m picollm.serve.chat_cli \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto
```

### Path B: Serious fine-tuning demo

```bash
uv run python -m picollm.sft_lora.prepare_dataset \
  --output-jsonl artifacts/picollm/sft/train.jsonl

uv run python -m picollm.sft_lora.finetune \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset artifacts/picollm/sft/train.jsonl \
  --output-dir artifacts/picollm/lora-run \
  --device auto \
  --max-steps 200
```

Then compare:

```bash
uv run python -m picollm.sft_lora.evaluate \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --device auto
```

### Path C: Full cloud pretraining

Use:

- [Vast.ai instances](https://docs.vast.ai/documentation/instances) for rentable GPUs
- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) when you want a managed cloud run tied to the Hub

Typical flow:

1. train tokenizer
2. pretrain a base checkpoint in cloud
3. full-SFT that checkpoint into a chatbot
4. evaluate
5. optionally push the final chatbot to the Hub
6. pull the chatbot checkpoint locally
7. run local inference on `cuda`, `mps`, or `cpu`
8. connect the same backend to the Vercel AI SDK app

See:

- [pretrain_cloud/README.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/README.md)
- [sft_lora/README.md](/Users/montekkundan/Developer/ML/llm/picollm/sft_lora/README.md)
- [rlhf/README.md](/Users/montekkundan/Developer/ML/llm/picollm/rlhf/README.md)
- [serve/README.md](/Users/montekkundan/Developer/ML/llm/picollm/serve/README.md)
- [eval/README.md](/Users/montekkundan/Developer/ML/llm/picollm/eval/README.md)
- [HUGGING_FACE_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/HUGGING_FACE_SETUP.md)
- [pretrain_cloud/VAST_AI_SETUP.md](/Users/montekkundan/Developer/ML/llm/picollm/pretrain_cloud/VAST_AI_SETUP.md)

## Advanced research utilities

For the advanced GPT-only lecture layer, `picollm/` now also includes:

- `picollm/rlhf/dpo_finetune.py`
  - minimal DPO post-training path
- `picollm/eval/run_lm_eval.py`
  - wrapper for `lm-eval-harness`
- `picollm/eval/safety_red_team.py`
  - small safety and jailbreak-oriented prompt suite runner
- `picollm/analysis/inspect_activations.py`
  - small hidden-state and attention inspection helper
- `picollm/eval/report_template.md`
  - student-facing experiment report template

## OS guidance

### macOS

- best local device flag: `mps`
- ideal use: inference and small LoRA runs
- full pretraining: use cloud

### Linux with NVIDIA

- best local device flag: `cuda`
- supports larger local runs
- optional 4-bit / 8-bit paths for serving and LoRA

### Windows with NVIDIA

- `cuda` works if your PyTorch install matches the driver/toolkit stack
- if not, fall back to CPU or use WSL2

### CPU-only machines

- use smaller models
- good for functional demos, not fast training

## This completes the lecture arc

With `picollm/`, the course now covers:

- concepts from first principles
- a tiny model you can fully understand
- chat formatting and SFT
- a serious LoRA fine-tuning path
- cloud pretraining as a realistic next step
- local inference and deployment on real hardware
