#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-cli}"
if [[ "$MODE" != "cli" && "$MODE" != "web" ]]; then
  echo "Usage: bash picollm/accelerated/speedrun.sh [cli|web]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export PICOLLM_BASE_DIR="${PICOLLM_BASE_DIR:-$REPO_ROOT/artifacts/picollm}"
export PICOLLM_FLASH_IMPL="${PICOLLM_FLASH_IMPL:-sdpa}"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-warning}"
mkdir -p "$PICOLLM_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export PYTHONUNBUFFERED=1

if [[ "$WANDB_RUN" != "dummy" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY (or use WANDB_RUN=dummy)}"
  : "${WANDB_ENTITY:?Set WANDB_ENTITY (or use WANDB_RUN=dummy)}"
fi

WANDB_ARGS=()
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb-entity="$WANDB_ENTITY")
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Public dataset downloads may still work, but authenticated Hub access is recommended." >&2
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m picollm.accelerated.report reset

python -m picollm.accelerated.dataset -n 8
python -m picollm.accelerated.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m picollm.accelerated.pretrain.train_tokenizer
python -m picollm.accelerated.pretrain.tokenizer_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.pretrain.train -- \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=1 \
  --total-batch-size=65536 \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.pretrain.eval -- \
  --device-batch-size=1

curl -L -o "$PICOLLM_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.chat.sft -- \
  --device-batch-size=1 \
  --total-batch-size=65536 \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.chat.eval -- \
  -i sft

python -m picollm.accelerated.report generate

if [[ "$MODE" == "web" ]]; then
  exec python -m picollm.accelerated.chat.web
else
  exec python -m picollm.accelerated.chat.cli
fi
