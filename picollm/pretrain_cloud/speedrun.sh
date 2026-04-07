#!/usr/bin/env bash
set -euo pipefail

# Inspired by nanochat's one-go speedrun workflow:
# https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh

MODE="cli"
PRESET="${PICO_PRESET:-2x4090}"
NPROC_PER_NODE="${PICO_NPROC_PER_NODE:-}"
SERVE_DEVICE="${PICO_SERVE_DEVICE:-auto}"
PRETRAIN_BATCH_SIZE="${PICO_PRETRAIN_BATCH_SIZE:-}"
PRETRAIN_GRAD_ACCUM="${PICO_PRETRAIN_GRAD_ACCUM:-}"
SFT_BATCH_SIZE="${PICO_SFT_BATCH_SIZE:-}"
SFT_GRAD_ACCUM="${PICO_SFT_GRAD_ACCUM:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli)
      MODE="cli"
      shift
      ;;
    --web)
      MODE="web"
      shift
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --device)
      SERVE_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash picollm/pretrain_cloud/speedrun.sh [--cli|--web] [--preset 2x4090|a100-80gb] [--nproc-per-node N] [--device DEVICE]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
if [[ -f "$HOME/.local/bin/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env"
fi

uv sync

TOKENIZER_DIR="artifacts/picollm/tokenizer"
PRETRAIN_DIR="artifacts/picollm/pretrain-run"
CHAT_SFT_DIR="artifacts/picollm/chat-sft-run"

case "$PRESET" in
  2x4090)
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-2}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
    ;;
  a100-80gb)
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-8}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Supported presets: 2x4090, a100-80gb" >&2
    exit 1
    ;;
esac

run_launcher() {
  local module="$1"
  shift
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m "$module" "$@"
  else
    uv run python -m "$module" "$@"
  fi
}

uv run python -m picollm.pretrain_cloud.train_tokenizer \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --max-texts 500000 \
  --vocab-size 32000 \
  --output-dir "$TOKENIZER_DIR"

run_launcher picollm.pretrain_cloud.train \
  --tokenizer-path "$TOKENIZER_DIR" \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --text-column text \
  --streaming \
  --output-dir "$PRETRAIN_DIR" \
  --block-size 1024 \
  --layers 24 \
  --heads 16 \
  --hidden-size 1024 \
  --batch-size "$PRETRAIN_BATCH_SIZE" \
  --grad-accum "$PRETRAIN_GRAD_ACCUM" \
  --warmup-steps 1000 \
  --save-steps 5000 \
  --max-steps 50000 \
  --bf16

run_launcher picollm.sft_full.finetune \
  --model "$PRETRAIN_DIR" \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir "$CHAT_SFT_DIR" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$SFT_GRAD_ACCUM" \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --save-steps 250 \
  --max-steps 1500 \
  --bf16

if [[ "$MODE" == "web" ]]; then
  exec uv run python -m picollm.serve.chat_web \
    --model "$CHAT_SFT_DIR" \
    --device "$SERVE_DEVICE"
fi

exec uv run python -m picollm.serve.chat_cli \
  --model "$CHAT_SFT_DIR" \
  --device "$SERVE_DEVICE"
