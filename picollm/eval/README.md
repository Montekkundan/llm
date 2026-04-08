# picoLLM Evaluation Utilities

This folder contains small teaching-scale evaluation helpers for the `picollm` workflows.

## Files

- `prompt_suite.json`
  - a small fixed chat-oriented prompt suite for comparing checkpoints
- `compare_checkpoints.py`
  - runs one or more checkpoints against the same prompt suite and prints a JSON comparison
- `chat_smoke.py`
  - runs the fixed prompt suite against one checkpoint, prints the replies directly, and flags obvious empty or looping failures
- `latency_benchmark.py`
  - measures simple prompt latency and tokens-per-second behavior for a chat checkpoint
- `safety_prompt_suite.json`
  - a small harmful / jailbreak-oriented prompt set for qualitative safety review
- `safety_red_team.py`
  - runs the safety prompt suite and saves responses for review
- `run_lm_eval.py`
  - thin wrapper around `lm-eval-harness` for formal benchmark runs
- `report_template.md`
  - markdown template for writing up experiment results

## Compare checkpoints

```bash
uv run python -m picollm.eval.compare_checkpoints \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --label base \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/picollm/lora-run \
  --label lora \
  --output artifacts/picollm/base_vs_lora_compare.json
```

## Measure latency

```bash
uv run python -m picollm.eval.latency_benchmark \
  --model artifacts/picollm/chat-sft-run \
  --output artifacts/picollm/latency_report.json
```

## Run a small safety review

```bash
uv run python -m picollm.eval.safety_red_team \
  --model artifacts/picollm/chat-sft-run \
  --output artifacts/picollm/safety_report.json
```

## Run lm-eval-harness

Install `lm-eval-harness` first if you do not already have `lm_eval` available:

```bash
uv tool install lm-eval
```

Then run:

```bash
uv run python -m picollm.eval.run_lm_eval \
  --model artifacts/picollm/chat-sft-run \
  --tasks hellaswag,piqa \
  --device cpu \
  --output-path artifacts/picollm/lm_eval_results
```
