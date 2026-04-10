# Re-Run Only SFT

Use this when the base model is already trained and you only want to iterate on assistant behavior.

## Inputs

- latest base checkpoint under `PICOLLM_BASE_DIR/base_checkpoints`
- tokenizer under `PICOLLM_BASE_DIR/tokenizer`
- identity dataset under `picollm/accelerated/data/identity_conversations.jsonl`

## Steps

```bash
uv sync --extra gpu
source .venv/bin/activate
export PICOLLM_BASE_DIR=/abs/path/to/artifacts/picollm
cp picollm/accelerated/data/identity_conversations.jsonl "$PICOLLM_BASE_DIR/identity_conversations.jsonl"
torchrun --standalone --nproc_per_node=1 -m picollm.accelerated.chat.sft -- --run=dummy
python -m picollm.accelerated.chat.identity_smoke
```

Useful knobs:

- `--device-batch-size`
- `--total-batch-size`
- `--mmlu-epochs`
- `--gsm8k-epochs`
- `--load-optimizer 0` if you intentionally do not want the pretrain optimizer state
