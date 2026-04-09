#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-cli}"
if [[ "$MODE" != "cli" && "$MODE" != "web" ]]; then
  echo "Usage: bash picollm/nanochat_backend/speedrun.sh [cli|web]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export PICOLLM_NANOCHAT_BASE_DIR="${PICOLLM_NANOCHAT_BASE_DIR:-$REPO_ROOT/artifacts/picollm_nanochat}"
mkdir -p "$PICOLLM_NANOCHAT_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
export PYTHONUNBUFFERED=1

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m picollm.nanochat_backend.report reset

python -m picollm.nanochat_backend.dataset -n 8
python -m picollm.nanochat_backend.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m picollm.nanochat_backend.scripts.tok_train
python -m picollm.nanochat_backend.scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node=8 -m picollm.nanochat_backend.scripts.base_train -- \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --fp8 \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.nanochat_backend.scripts.base_eval -- \
  --device-batch-size=16

curl -L -o "$PICOLLM_NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m picollm.nanochat_backend.scripts.chat_sft -- \
  --device-batch-size=16 \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.nanochat_backend.scripts.chat_eval -- \
  -i sft

python -m picollm.nanochat_backend.report generate

if [[ "$MODE" == "web" ]]; then
  exec python -m picollm.nanochat_backend.scripts.chat_web
else
  exec python -m picollm.nanochat_backend.scripts.chat_cli
fi
