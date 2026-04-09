# PicoLLM Nanochat Backend

This backend vendors the Nanochat-style tokenizer, base training, SFT, eval, CLI, and web chat path into `picollm`.

Default artifact root:

- `artifacts/picollm_nanochat`

Override with:

- `PICOLLM_NANOCHAT_BASE_DIR=/abs/path`

End-to-end speedrun on a fresh machine:

```bash
bash picollm/nanochat_backend/speedrun.sh cli
```

Web UI instead of CLI:

```bash
bash picollm/nanochat_backend/speedrun.sh web
```

Main entrypoints:

- `python -m picollm.nanochat_backend.dataset`
- `python -m picollm.nanochat_backend.scripts.tok_train`
- `python -m picollm.nanochat_backend.scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8`
- `python -m picollm.nanochat_backend.scripts.base_eval -- --device-batch-size=16`
- `python -m picollm.nanochat_backend.scripts.chat_sft -- --device-batch-size=16`
- `python -m picollm.nanochat_backend.scripts.chat_eval -- -i sft`
- `python -m picollm.nanochat_backend.scripts.chat_cli`
- `python -m picollm.nanochat_backend.scripts.chat_web`
