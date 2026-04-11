# picoLLM

`picollm/` is the main from-scratch chatbot path in this repo.

There is one canonical path now:

- [accelerated/README.md](./accelerated/README.md)
- [accelerated/speedrun.sh](./accelerated/speedrun.sh)

That path owns the full workflow:

- tokenizer training
- base pretraining
- chat SFT
- evaluation
- report generation
- CLI or web chat after the run

## File structure

```text
picollm/
├── README.md
├── accelerated/
│   ├── speedrun.sh
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── gpt.py
│   ├── optim.py
│   ├── engine.py
│   ├── checkpoint_manager.py
│   ├── report.py
│   ├── pretrain/
│   │   ├── train_tokenizer.py
│   │   ├── tokenizer_eval.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── chat/
│   │   ├── sft.py
│   │   ├── eval.py
│   │   ├── cli.py
│   │   └── web.py
│   └── tasks/
│       ├── smoltalk.py
│       ├── gsm8k.py
│       ├── mmlu.py
│       ├── arc.py
│       ├── humaneval.py
│       └── spellingbee.py
└── common/
    └── device.py
```

## Recommended usage

Train the full model:

```bash
bash picollm/accelerated/speedrun.sh |& tee "$PICOLLM_BASE_DIR/speedrun.log"
```

If you do not want to capture the console log, run:

```bash
bash picollm/accelerated/speedrun.sh
```

Run the API and browser UI after training:

```bash
python -m picollm.accelerated.chat.web
```

Run the terminal chat directly:

```bash
python -m picollm.accelerated.chat.cli -i sft
```

## Frontend integration

The two product-style apps in `apps/` both talk to the accelerated web server through its OpenAI-compatible API:

- [../apps/vercel_ai_sdk_chat/README.md](../apps/vercel_ai_sdk_chat/README.md)
- [../apps/opentui_ai_sdk_chat/README.md](../apps/opentui_ai_sdk_chat/README.md)
