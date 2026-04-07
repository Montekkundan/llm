# picoLLM Preference Optimization

This folder contains a minimal DPO path for the advanced GPT-only lecture layer.

## Expected dataset format

The dataset must provide:

- `prompt`
- `chosen`
- `rejected`

You can use either:

- a local `.jsonl` file
- a Hugging Face dataset name

## Run DPO

```bash
uv run python -m picollm.rlhf.dpo_finetune \
  --model artifacts/picollm/chat-sft-run \
  --dataset preferences.jsonl \
  --output-dir artifacts/picollm/dpo-run
```
