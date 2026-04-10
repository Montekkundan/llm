# Local Smoke Flow

Use this on a laptop or workstation when you want to validate the picoLLM repo without rerunning the full accelerated training path.

## macOS

Recommended setup:

```bash
uv sync --extra cpu
source .venv/bin/activate
python scripts/print_picollm_env.py
python scripts/run_picollm_local_checks.py --device mps
```

Notes:

- use `--device-type mps` for local chat and model smoke commands
- expect unit tests, API wiring checks, and restore smoke checks, not a full CUDA training run

## CUDA Box

Recommended setup:

```bash
uv sync --extra gpu
source .venv/bin/activate
python scripts/print_picollm_env.py
python scripts/run_picollm_local_checks.py --device cuda
```

If you already published a model repo and want the end-to-end restore smoke:

```bash
python scripts/run_picollm_local_checks.py --device cuda --hf-repo-id your-username/your-picollm-backup
```

## CPU-Only

Recommended setup:

```bash
uv sync --extra cpu
source .venv/bin/activate
python scripts/run_picollm_local_checks.py --device cpu
```
