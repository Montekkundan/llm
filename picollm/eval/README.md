# picoLLM Evaluation Utilities

This folder contains small teaching-scale evaluation helpers for the `picollm` workflows.

## Files

- `prompt_suite.json`
  - a small fixed chat-oriented prompt suite for comparing checkpoints
- `compare_checkpoints.py`
  - runs one or more checkpoints against the same prompt suite and prints a JSON comparison
- `latency_benchmark.py`
  - measures simple prompt latency and tokens-per-second behavior for a chat checkpoint

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
