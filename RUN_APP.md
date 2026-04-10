# Run The Course Demo App

Local run path:

1. `uv sync`
2. `uv run python scripts/deployment/build_demo_checkpoint.py`
3. `uv run python scripts/fastapi_chat_app/serve.py`
4. Open `http://127.0.0.1:8000`

Useful checks:

- `curl http://127.0.0.1:8000/health`
- `curl http://127.0.0.1:8000/metadata`
- `uv run python scripts/deployment/smoke_test.py`

Accelerated picoLLM product-track check:

- Run `python -m picollm.accelerated.chat.web`
- Then run `python scripts/deployment/smoke_test_accelerated.py`

Product-track alternatives:

- CLI: `uv run python scripts/cli_and_web_chat/chat_cli.py`
- Web launcher: `uv run python scripts/cli_and_web_chat/chat_web.py`
