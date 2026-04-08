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
PRETRAIN_WARMUP_STEPS="${PICO_PRETRAIN_WARMUP_STEPS:-}"
PRETRAIN_SAVE_STEPS="${PICO_PRETRAIN_SAVE_STEPS:-}"
PRETRAIN_MAX_STEPS="${PICO_PRETRAIN_MAX_STEPS:-}"
SFT_BATCH_SIZE="${PICO_SFT_BATCH_SIZE:-}"
SFT_GRAD_ACCUM="${PICO_SFT_GRAD_ACCUM:-}"
SFT_WARMUP_STEPS="${PICO_SFT_WARMUP_STEPS:-}"
SFT_SAVE_STEPS="${PICO_SFT_SAVE_STEPS:-}"
SFT_MAX_STEPS="${PICO_SFT_MAX_STEPS:-}"
HF_REPO_ID="${PICO_HF_REPO_ID:-}"
HF_TOKEN_VALUE="${PICO_HF_TOKEN:-}"
REPORT_TO="${PICO_REPORT_TO:-none}"
RUN_NAME="${PICO_RUN_NAME:-}"
WANDB_PROJECT="${PICO_WANDB_PROJECT:-}"
WANDB_ENTITY="${PICO_WANDB_ENTITY:-}"
WANDB_API_KEY_VALUE="${PICO_WANDB_API_KEY:-}"

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
    --hf-repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --report-to)
      REPORT_TO="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-api-key)
      WANDB_API_KEY_VALUE="$2"
      shift 2
      ;;
    --device)
      SERVE_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash picollm/pretrain_cloud/speedrun.sh [--cli|--web] [--preset 2x4090|a100-80gb|8xh100|8xa100] [--nproc-per-node N] [--hf-repo-id REPO] [--hf-token TOKEN] [--report-to none|tensorboard|wandb] [--run-name NAME] [--wandb-project NAME] [--wandb-entity ENTITY] [--wandb-api-key KEY] [--device DEVICE]" >&2
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

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  export HF_TOKEN="$HF_TOKEN_VALUE"
fi

if [[ -n "$WANDB_API_KEY_VALUE" ]]; then
  export WANDB_API_KEY="$WANDB_API_KEY_VALUE"
fi

if [[ "$REPORT_TO" == "wandb" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Using WANDB_API_KEY from the environment for Weights & Biases logging."
  elif uv run python - <<'PY' >/dev/null 2>&1
from picollm.common.telemetry import has_wandb_auth
raise SystemExit(0 if has_wandb_auth() else 1)
PY
  then
    echo "Using existing Weights & Biases login for telemetry."
  else
    echo "You passed --report-to wandb, but no Weights & Biases auth was found." >&2
    echo "Fix one of these before rerunning:" >&2
    echo "  1. export WANDB_API_KEY=\"...\"" >&2
    echo "  2. run: wandb login" >&2
    exit 1
  fi
fi

if [[ -n "$HF_REPO_ID" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Using HF_TOKEN from the environment for Hub upload."
  elif uv run hf auth whoami >/dev/null 2>&1; then
    echo "Using existing Hugging Face CLI login for Hub upload."
  else
    echo "You passed --hf-repo-id, but no Hugging Face auth was found." >&2
    echo "Fix one of these before rerunning:" >&2
    echo "  1. export HF_TOKEN=\"...\"" >&2
    echo "  2. run: hf auth login" >&2
    exit 1
  fi
fi

if [[ "$REPORT_TO" == "wandb" && -n "$WANDB_PROJECT" ]]; then
  export WANDB_PROJECT="$WANDB_PROJECT"
fi

if [[ "$REPORT_TO" == "wandb" && -n "$WANDB_ENTITY" ]]; then
  export WANDB_ENTITY="$WANDB_ENTITY"
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="picollm-${PRESET}-${MODE}"
fi

TOKENIZER_DIR="artifacts/picollm/tokenizer"
PRETRAIN_DIR="artifacts/picollm/pretrain-run"
CHAT_SFT_DIR="artifacts/picollm/chat-sft-run"

case "$PRESET" in
  2x4090)
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-2}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    ;;
  a100-80gb)
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    ;;
  8xh100)
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    ;;
  8xa100)
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-4}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Supported presets: 2x4090, a100-80gb, 8xh100, 8xa100" >&2
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
  --warmup-steps "$PRETRAIN_WARMUP_STEPS" \
  --save-steps "$PRETRAIN_SAVE_STEPS" \
  --max-steps "$PRETRAIN_MAX_STEPS" \
  --bf16 \
  --report-to "$REPORT_TO" \
  --run-name "${RUN_NAME}-pretrain"

run_launcher picollm.sft_full.finetune \
  --model "$PRETRAIN_DIR" \
  --dataset-name HuggingFaceTB/everyday-conversations-llama3.1-2k \
  --dataset-split train_sft \
  --text-column messages \
  --output-dir "$CHAT_SFT_DIR" \
  --batch-size "$SFT_BATCH_SIZE" \
  --grad-accum "$SFT_GRAD_ACCUM" \
  --learning-rate 2e-5 \
  --warmup-steps "$SFT_WARMUP_STEPS" \
  --save-steps "$SFT_SAVE_STEPS" \
  --max-steps "$SFT_MAX_STEPS" \
  --bf16 \
  --report-to "$REPORT_TO" \
  --run-name "${RUN_NAME}-sft"

if [[ -n "$HF_REPO_ID" ]]; then
  uv run python -m picollm.pretrain_cloud.push_to_hub \
    --folder "$CHAT_SFT_DIR" \
    --repo-id "$HF_REPO_ID"
fi

if [[ "$MODE" == "web" ]]; then
  exec uv run python -m picollm.serve.chat_web \
    --model "$CHAT_SFT_DIR" \
    --device "$SERVE_DEVICE"
fi

exec uv run python -m picollm.serve.chat_cli \
  --model "$CHAT_SFT_DIR" \
  --device "$SERVE_DEVICE"
