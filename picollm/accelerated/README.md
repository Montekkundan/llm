# picoLLM Training Stack

This package contains the single serious training and inference path for `picollm`.

Default artifact root:

- `artifacts/picollm`

Override with:

- `PICOLLM_BASE_DIR=/abs/path`

End-to-end speedrun on a fresh machine:

```bash
bash picollm/accelerated/speedrun.sh cli
```

Web UI instead of CLI:

```bash
bash picollm/accelerated/speedrun.sh web
```

Main entrypoints:

- `python -m picollm.accelerated.dataset`
- `python -m picollm.accelerated.pretrain.train_tokenizer`
- `python -m picollm.accelerated.pretrain.train --depth=24 --target-param-data-ratio=8 --device-batch-size=4 --fp8`
- `python -m picollm.accelerated.pretrain.eval --device-batch-size=4`
- `python -m picollm.accelerated.chat.sft --device-batch-size=4`
- `python -m picollm.accelerated.chat.eval -i sft`
- `python -m picollm.accelerated.chat.cli`
- `python -m picollm.accelerated.chat.web`
