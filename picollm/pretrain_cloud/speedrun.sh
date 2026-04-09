#!/usr/bin/env bash
set -Eeuo pipefail

# Inspired by nanochat's one-go speedrun workflow:
# https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh

MODE="cli"
PRESET="${PICO_PRESET:-8xh100}"
NPROC_PER_NODE="${PICO_NPROC_PER_NODE:-}"
SERVE_DEVICE="${PICO_SERVE_DEVICE:-auto}"
PRETRAIN_BATCH_SIZE="${PICO_PRETRAIN_BATCH_SIZE:-}"
PRETRAIN_GRAD_ACCUM="${PICO_PRETRAIN_GRAD_ACCUM:-}"
PRETRAIN_WARMUP_STEPS="${PICO_PRETRAIN_WARMUP_STEPS:-}"
PRETRAIN_SAVE_STEPS="${PICO_PRETRAIN_SAVE_STEPS:-}"
PRETRAIN_SAVE_TOTAL_LIMIT="${PICO_PRETRAIN_SAVE_TOTAL_LIMIT:-}"
PRETRAIN_MAX_STEPS="${PICO_PRETRAIN_MAX_STEPS:-}"
PRETRAIN_TARGET_PARAM_DATA_RATIO="${PICO_PRETRAIN_TARGET_PARAM_DATA_RATIO:-}"
PRETRAIN_PREVIEW_EVERY_STEPS="${PICO_PRETRAIN_PREVIEW_EVERY_STEPS:-}"
PRETRAIN_ARCHITECTURE="${PICO_PRETRAIN_ARCHITECTURE:-}"
PRETRAIN_BLOCK_SIZE="${PICO_PRETRAIN_BLOCK_SIZE:-}"
PRETRAIN_LAYERS="${PICO_PRETRAIN_LAYERS:-}"
PRETRAIN_HEADS="${PICO_PRETRAIN_HEADS:-}"
PRETRAIN_KV_HEADS="${PICO_PRETRAIN_KV_HEADS:-}"
PRETRAIN_HIDDEN_SIZE="${PICO_PRETRAIN_HIDDEN_SIZE:-}"
PRETRAIN_INTERMEDIATE_SIZE="${PICO_PRETRAIN_INTERMEDIATE_SIZE:-}"
PRETRAIN_DATALOADER_WORKERS="${PICO_PRETRAIN_DATALOADER_WORKERS:-}"
PRETRAIN_OPTIMIZER="${PICO_PRETRAIN_OPTIMIZER:-}"
PRETRAIN_LR_SCHEDULER_TYPE="${PICO_PRETRAIN_LR_SCHEDULER_TYPE:-}"
PRETRAIN_TORCH_COMPILE="${PICO_PRETRAIN_TORCH_COMPILE:-0}"
TOKENIZER_MAX_TEXTS="${PICO_TOKENIZER_MAX_TEXTS:-}"
TOKENIZER_VOCAB_SIZE="${PICO_TOKENIZER_VOCAB_SIZE:-}"
TOKENIZER_MIN_FREQUENCY="${PICO_TOKENIZER_MIN_FREQUENCY:-}"
SFT_DATASET_NAME="${PICO_SFT_DATASET_NAME:-}"
SFT_DATASET_CONFIG="${PICO_SFT_DATASET_CONFIG:-}"
SFT_DATASET_SPLIT="${PICO_SFT_DATASET_SPLIT:-}"
SFT_MIXTURE="${PICO_SFT_MIXTURE:-}"
SFT_SMOLTALK_EPOCHS="${PICO_SFT_SMOLTALK_EPOCHS:-}"
SFT_EVERYDAY_EPOCHS="${PICO_SFT_EVERYDAY_EPOCHS:-}"
SFT_IDENTITY_EPOCHS="${PICO_SFT_IDENTITY_EPOCHS:-}"
SFT_MMLU_EPOCHS="${PICO_SFT_MMLU_EPOCHS:-}"
SFT_GSM8K_EPOCHS="${PICO_SFT_GSM8K_EPOCHS:-}"
SFT_SIMPLE_SPELLING_SIZE="${PICO_SFT_SIMPLE_SPELLING_SIZE:-}"
SFT_SPELLING_BEE_SIZE="${PICO_SFT_SPELLING_BEE_SIZE:-}"
SFT_BATCH_SIZE="${PICO_SFT_BATCH_SIZE:-}"
SFT_GRAD_ACCUM="${PICO_SFT_GRAD_ACCUM:-}"
SFT_LEARNING_RATE="${PICO_SFT_LEARNING_RATE:-}"
SFT_WARMUP_STEPS="${PICO_SFT_WARMUP_STEPS:-}"
SFT_SAVE_STEPS="${PICO_SFT_SAVE_STEPS:-}"
SFT_SAVE_TOTAL_LIMIT="${PICO_SFT_SAVE_TOTAL_LIMIT:-}"
SFT_MAX_STEPS="${PICO_SFT_MAX_STEPS:-}"
SFT_PREVIEW_EVERY_STEPS="${PICO_SFT_PREVIEW_EVERY_STEPS:-}"
SFT_DATALOADER_WORKERS="${PICO_SFT_DATALOADER_WORKERS:-}"
SFT_OPTIMIZER="${PICO_SFT_OPTIMIZER:-}"
SFT_LR_SCHEDULER_TYPE="${PICO_SFT_LR_SCHEDULER_TYPE:-}"
SFT_TORCH_COMPILE="${PICO_SFT_TORCH_COMPILE:-0}"
MIN_FREE_GB="${PICO_MIN_FREE_GB:-}"
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

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="picollm-${PRESET}-${MODE}"
fi

LOG_DIR="artifacts/picollm/logs"
EVAL_DIR="artifacts/picollm/evals"
mkdir -p "$LOG_DIR" "$EVAL_DIR"
RUN_STAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="$LOG_DIR/${RUN_NAME}-${RUN_STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

CURRENT_STAGE="setup"

on_error() {
  local line="$1"
  local command="$2"
  local exit_code="$3"
  echo
  echo "ERROR: speedrun failed." >&2
  echo "  stage: $CURRENT_STAGE" >&2
  echo "  line: $line" >&2
  echo "  command: $command" >&2
  echo "  exit_code: $exit_code" >&2
  echo "  log_file: $LOG_FILE" >&2
  echo "  disk snapshot:" >&2
  df -h "$REPO_ROOT" || true
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}" "$?"' ERR

TOKENIZER_DIR="artifacts/picollm/tokenizer"
PRETRAIN_DIR="artifacts/picollm/pretrain-run"
CHAT_SFT_DIR="artifacts/picollm/chat-sft-run"

case "$PRESET" in
  2x4090)
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    TOKENIZER_MAX_TEXTS="${TOKENIZER_MAX_TEXTS:-500000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
    TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-0}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-2}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_SAVE_TOTAL_LIMIT="${PRETRAIN_SAVE_TOTAL_LIMIT:-1}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    PRETRAIN_PREVIEW_EVERY_STEPS="${PRETRAIN_PREVIEW_EVERY_STEPS:-1000}"
    PRETRAIN_ARCHITECTURE="${PRETRAIN_ARCHITECTURE:-llama}"
    PRETRAIN_BLOCK_SIZE="${PRETRAIN_BLOCK_SIZE:-1024}"
    PRETRAIN_LAYERS="${PRETRAIN_LAYERS:-24}"
    PRETRAIN_HEADS="${PRETRAIN_HEADS:-16}"
    PRETRAIN_KV_HEADS="${PRETRAIN_KV_HEADS:-8}"
    PRETRAIN_HIDDEN_SIZE="${PRETRAIN_HIDDEN_SIZE:-1024}"
    PRETRAIN_INTERMEDIATE_SIZE="${PRETRAIN_INTERMEDIATE_SIZE:-4096}"
    PRETRAIN_DATALOADER_WORKERS="${PRETRAIN_DATALOADER_WORKERS:-2}"
    PRETRAIN_OPTIMIZER="${PRETRAIN_OPTIMIZER:-auto}"
    PRETRAIN_LR_SCHEDULER_TYPE="${PRETRAIN_LR_SCHEDULER_TYPE:-cosine}"
    SFT_DATASET_NAME="${SFT_DATASET_NAME:-HuggingFaceTB/everyday-conversations-llama3.1-2k}"
    SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train_sft}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-5000}"
    SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-1}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    SFT_PREVIEW_EVERY_STEPS="${SFT_PREVIEW_EVERY_STEPS:-50}"
    SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-2}"
    SFT_OPTIMIZER="${SFT_OPTIMIZER:-auto}"
    SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"
    MIN_FREE_GB="${MIN_FREE_GB:-25}"
    ;;
  a100-80gb)
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
    TOKENIZER_MAX_TEXTS="${TOKENIZER_MAX_TEXTS:-500000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
    TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-0}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-500}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-1000}"
    PRETRAIN_SAVE_TOTAL_LIMIT="${PRETRAIN_SAVE_TOTAL_LIMIT:-1}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-5000}"
    PRETRAIN_PREVIEW_EVERY_STEPS="${PRETRAIN_PREVIEW_EVERY_STEPS:-1000}"
    PRETRAIN_ARCHITECTURE="${PRETRAIN_ARCHITECTURE:-llama}"
    PRETRAIN_BLOCK_SIZE="${PRETRAIN_BLOCK_SIZE:-1024}"
    PRETRAIN_LAYERS="${PRETRAIN_LAYERS:-24}"
    PRETRAIN_HEADS="${PRETRAIN_HEADS:-16}"
    PRETRAIN_KV_HEADS="${PRETRAIN_KV_HEADS:-8}"
    PRETRAIN_HIDDEN_SIZE="${PRETRAIN_HIDDEN_SIZE:-1024}"
    PRETRAIN_INTERMEDIATE_SIZE="${PRETRAIN_INTERMEDIATE_SIZE:-4096}"
    PRETRAIN_DATALOADER_WORKERS="${PRETRAIN_DATALOADER_WORKERS:-2}"
    PRETRAIN_OPTIMIZER="${PRETRAIN_OPTIMIZER:-auto}"
    PRETRAIN_LR_SCHEDULER_TYPE="${PRETRAIN_LR_SCHEDULER_TYPE:-cosine}"
    SFT_DATASET_NAME="${SFT_DATASET_NAME:-HuggingFaceTB/everyday-conversations-llama3.1-2k}"
    SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train_sft}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-10}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-5000}"
    SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-1}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-150}"
    SFT_PREVIEW_EVERY_STEPS="${SFT_PREVIEW_EVERY_STEPS:-50}"
    SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-2}"
    SFT_OPTIMIZER="${SFT_OPTIMIZER:-auto}"
    SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"
    MIN_FREE_GB="${MIN_FREE_GB:-40}"
    ;;
  8xh100)
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    TOKENIZER_MAX_TEXTS="${TOKENIZER_MAX_TEXTS:-1000000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
    TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-0}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-100}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-500}"
    PRETRAIN_SAVE_TOTAL_LIMIT="${PRETRAIN_SAVE_TOTAL_LIMIT:-1}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:--1}"
    PRETRAIN_TARGET_PARAM_DATA_RATIO="${PRETRAIN_TARGET_PARAM_DATA_RATIO:-8}"
    PRETRAIN_PREVIEW_EVERY_STEPS="${PRETRAIN_PREVIEW_EVERY_STEPS:-500}"
    PRETRAIN_ARCHITECTURE="${PRETRAIN_ARCHITECTURE:-llama}"
    PRETRAIN_BLOCK_SIZE="${PRETRAIN_BLOCK_SIZE:-2048}"
    PRETRAIN_LAYERS="${PRETRAIN_LAYERS:-24}"
    PRETRAIN_HEADS="${PRETRAIN_HEADS:-12}"
    PRETRAIN_KV_HEADS="${PRETRAIN_KV_HEADS:-12}"
    PRETRAIN_HIDDEN_SIZE="${PRETRAIN_HIDDEN_SIZE:-1536}"
    PRETRAIN_INTERMEDIATE_SIZE="${PRETRAIN_INTERMEDIATE_SIZE:-6144}"
    PRETRAIN_DATALOADER_WORKERS="${PRETRAIN_DATALOADER_WORKERS:-4}"
    PRETRAIN_OPTIMIZER="${PRETRAIN_OPTIMIZER:-auto}"
    PRETRAIN_LR_SCHEDULER_TYPE="${PRETRAIN_LR_SCHEDULER_TYPE:-cosine}"
    SFT_DATASET_NAME="${SFT_DATASET_NAME:-HuggingFaceTB/smol-smoltalk}"
    SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train}"
    SFT_MIXTURE="${SFT_MIXTURE:-nanochat-lite}"
    SFT_SMOLTALK_EPOCHS="${SFT_SMOLTALK_EPOCHS:-1}"
    SFT_EVERYDAY_EPOCHS="${SFT_EVERYDAY_EPOCHS:-1}"
    SFT_IDENTITY_EPOCHS="${SFT_IDENTITY_EPOCHS:-2}"
    SFT_MMLU_EPOCHS="${SFT_MMLU_EPOCHS:-1}"
    SFT_GSM8K_EPOCHS="${SFT_GSM8K_EPOCHS:-1}"
    SFT_SIMPLE_SPELLING_SIZE="${SFT_SIMPLE_SPELLING_SIZE:-20000}"
    SFT_SPELLING_BEE_SIZE="${SFT_SPELLING_BEE_SIZE:-8000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-100}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
    SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-1}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-1500}"
    SFT_PREVIEW_EVERY_STEPS="${SFT_PREVIEW_EVERY_STEPS:-250}"
    SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-4}"
    SFT_OPTIMIZER="${SFT_OPTIMIZER:-auto}"
    SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"
    MIN_FREE_GB="${MIN_FREE_GB:-120}"
    ;;
  8xa100)
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    TOKENIZER_MAX_TEXTS="${TOKENIZER_MAX_TEXTS:-1000000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
    TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-0}"
    PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-2}"
    PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-16}"
    PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-100}"
    PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-500}"
    PRETRAIN_SAVE_TOTAL_LIMIT="${PRETRAIN_SAVE_TOTAL_LIMIT:-1}"
    PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:--1}"
    PRETRAIN_TARGET_PARAM_DATA_RATIO="${PRETRAIN_TARGET_PARAM_DATA_RATIO:-8}"
    PRETRAIN_PREVIEW_EVERY_STEPS="${PRETRAIN_PREVIEW_EVERY_STEPS:-500}"
    PRETRAIN_ARCHITECTURE="${PRETRAIN_ARCHITECTURE:-llama}"
    PRETRAIN_BLOCK_SIZE="${PRETRAIN_BLOCK_SIZE:-2048}"
    PRETRAIN_LAYERS="${PRETRAIN_LAYERS:-24}"
    PRETRAIN_HEADS="${PRETRAIN_HEADS:-12}"
    PRETRAIN_KV_HEADS="${PRETRAIN_KV_HEADS:-12}"
    PRETRAIN_HIDDEN_SIZE="${PRETRAIN_HIDDEN_SIZE:-1536}"
    PRETRAIN_INTERMEDIATE_SIZE="${PRETRAIN_INTERMEDIATE_SIZE:-6144}"
    PRETRAIN_DATALOADER_WORKERS="${PRETRAIN_DATALOADER_WORKERS:-4}"
    PRETRAIN_OPTIMIZER="${PRETRAIN_OPTIMIZER:-auto}"
    PRETRAIN_LR_SCHEDULER_TYPE="${PRETRAIN_LR_SCHEDULER_TYPE:-cosine}"
    SFT_DATASET_NAME="${SFT_DATASET_NAME:-HuggingFaceTB/smol-smoltalk}"
    SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train}"
    SFT_MIXTURE="${SFT_MIXTURE:-nanochat-lite}"
    SFT_SMOLTALK_EPOCHS="${SFT_SMOLTALK_EPOCHS:-1}"
    SFT_EVERYDAY_EPOCHS="${SFT_EVERYDAY_EPOCHS:-1}"
    SFT_IDENTITY_EPOCHS="${SFT_IDENTITY_EPOCHS:-2}"
    SFT_MMLU_EPOCHS="${SFT_MMLU_EPOCHS:-1}"
    SFT_GSM8K_EPOCHS="${SFT_GSM8K_EPOCHS:-1}"
    SFT_SIMPLE_SPELLING_SIZE="${SFT_SIMPLE_SPELLING_SIZE:-20000}"
    SFT_SPELLING_BEE_SIZE="${SFT_SPELLING_BEE_SIZE:-8000}"
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
    SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
    SFT_WARMUP_STEPS="${SFT_WARMUP_STEPS:-100}"
    SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
    SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-1}"
    SFT_MAX_STEPS="${SFT_MAX_STEPS:-1500}"
    SFT_PREVIEW_EVERY_STEPS="${SFT_PREVIEW_EVERY_STEPS:-250}"
    SFT_DATALOADER_WORKERS="${SFT_DATALOADER_WORKERS:-4}"
    SFT_OPTIMIZER="${SFT_OPTIMIZER:-auto}"
    SFT_LR_SCHEDULER_TYPE="${SFT_LR_SCHEDULER_TYPE:-cosine}"
    MIN_FREE_GB="${MIN_FREE_GB:-120}"
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

prune_checkpoint_dirs() {
  local parent_dir="$1"
  if [[ -d "$parent_dir" ]]; then
    find "$parent_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
  fi
}

stage() {
  CURRENT_STAGE="$1"
  echo
  echo "===== $CURRENT_STAGE ====="
  date
}

available_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
  else
    echo 0
  fi
}

require_gpu_count() {
  local required="$1"
  local detected
  detected="$(available_gpus)"
  if [[ "$detected" -lt "$required" ]]; then
    echo "Expected at least $required GPU(s), but detected $detected." >&2
    exit 1
  fi
}

available_gb() {
  df -Pk "$REPO_ROOT" | awk 'NR==2 { printf "%.0f", $4 / 1024 / 1024 }'
}

require_free_space() {
  local required_gb="$1"
  local available
  available="$(available_gb)"
  if [[ "$available" -lt "$required_gb" ]]; then
    echo "Not enough free disk space in $REPO_ROOT: need about ${required_gb}GB free, found ${available}GB." >&2
    exit 1
  fi
}

model_exists() {
  local dir="$1"
  [[ -f "$dir/model.safetensors" || -f "$dir/model.safetensors.index.json" ]]
}

latest_checkpoint_dir() {
  local parent_dir="$1"
  find "$parent_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

run_pretrain_eval() {
  local pretrain_eval_args=(
    --model "$PRETRAIN_DIR"
    --dataset-name HuggingFaceFW/fineweb-edu
    --dataset-config sample-10BT
    --dataset-split train
    --text-column text
    --streaming
    --sample-prompt "Machine learning is"
    --output "$EVAL_DIR/pretrain_eval.json"
  )

  stage "pretrain-eval"
  uv run python -m picollm.pretrain_cloud.eval "${pretrain_eval_args[@]}"
  uv run python -m picollm.eval.chat_smoke \
    --model "$PRETRAIN_DIR" \
    --label pretrain \
    --output "$EVAL_DIR/pretrain_chat_smoke.json"
}

run_chat_eval() {
  local chat_eval_args=(
    --model "$CHAT_SFT_DIR"
    --dataset-name "$SFT_DATASET_NAME"
    --dataset-split "$SFT_DATASET_SPLIT"
    --text-column messages
    --streaming
    --sample-prompt "hi"
    --output "$EVAL_DIR/chat_eval.json"
  )

  if [[ -n "$SFT_DATASET_CONFIG" ]]; then
    chat_eval_args+=(--dataset-config "$SFT_DATASET_CONFIG")
  fi

  stage "chat-eval"
  uv run python -m picollm.pretrain_cloud.eval "${chat_eval_args[@]}"
  uv run python -m picollm.eval.chat_smoke \
    --model "$CHAT_SFT_DIR" \
    --label chat-sft \
    --output "$EVAL_DIR/chat_smoke.json" \
    --fail-on-catastrophic
  uv run python -m picollm.eval.compare_checkpoints \
    --model "$PRETRAIN_DIR" \
    --label pretrain \
    --model "$CHAT_SFT_DIR" \
    --label chat-sft \
    --output "$EVAL_DIR/checkpoint_compare.json"
}

stage "preflight"
echo "run_name: $RUN_NAME"
echo "preset: $PRESET"
echo "mode: $MODE"
echo "log_file: $LOG_FILE"
echo "free_disk_gb: $(available_gb)"
df -h "$REPO_ROOT"
require_gpu_count "$NPROC_PER_NODE"
require_free_space "$MIN_FREE_GB"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

stage "setup"
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

if [[ -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
  stage "tokenizer-skip"
  echo "Tokenizer already exists at $TOKENIZER_DIR. Skipping tokenizer training."
else
  stage "tokenizer-train"
  uv run python -m picollm.pretrain_cloud.train_tokenizer \
    --dataset-name HuggingFaceFW/fineweb-edu \
    --dataset-config sample-10BT \
    --dataset-split train \
    --text-column text \
    --streaming \
    --max-texts "$TOKENIZER_MAX_TEXTS" \
    --vocab-size "$TOKENIZER_VOCAB_SIZE" \
    --min-frequency "$TOKENIZER_MIN_FREQUENCY" \
    --output-dir "$TOKENIZER_DIR"
fi

if model_exists "$PRETRAIN_DIR"; then
  stage "pretrain-skip"
  echo "Final pretrain checkpoint already exists at $PRETRAIN_DIR. Skipping base pretraining."
else
  stage "pretrain-train"
  PRETRAIN_ARGS=(
    --tokenizer-path "$TOKENIZER_DIR"
    --dataset-name HuggingFaceFW/fineweb-edu
    --dataset-config sample-10BT
    --dataset-split train
    --text-column text
    --streaming
    --output-dir "$PRETRAIN_DIR"
    --architecture "$PRETRAIN_ARCHITECTURE"
    --block-size "$PRETRAIN_BLOCK_SIZE"
    --layers "$PRETRAIN_LAYERS"
    --heads "$PRETRAIN_HEADS"
    --kv-heads "$PRETRAIN_KV_HEADS"
    --hidden-size "$PRETRAIN_HIDDEN_SIZE"
    --intermediate-size "$PRETRAIN_INTERMEDIATE_SIZE"
    --batch-size "$PRETRAIN_BATCH_SIZE"
    --grad-accum "$PRETRAIN_GRAD_ACCUM"
    --optimizer "$PRETRAIN_OPTIMIZER"
    --lr-scheduler-type "$PRETRAIN_LR_SCHEDULER_TYPE"
    --dataloader-num-workers "$PRETRAIN_DATALOADER_WORKERS"
    --warmup-steps "$PRETRAIN_WARMUP_STEPS"
    --save-steps "$PRETRAIN_SAVE_STEPS"
    --save-total-limit "$PRETRAIN_SAVE_TOTAL_LIMIT"
    --preview-every-steps "$PRETRAIN_PREVIEW_EVERY_STEPS"
    --bf16
    --report-to "$REPORT_TO"
    --run-name "${RUN_NAME}-pretrain"
  )
  if [[ "$PRETRAIN_MAX_STEPS" -gt 0 ]]; then
    PRETRAIN_ARGS+=(--max-steps "$PRETRAIN_MAX_STEPS")
  fi
  if [[ "${PRETRAIN_TARGET_PARAM_DATA_RATIO:-0}" != "0" && "${PRETRAIN_TARGET_PARAM_DATA_RATIO:-}" != "" ]]; then
    PRETRAIN_ARGS+=(--target-param-data-ratio "$PRETRAIN_TARGET_PARAM_DATA_RATIO")
  fi
  if [[ "$PRETRAIN_TORCH_COMPILE" == "1" ]]; then
    PRETRAIN_ARGS+=(--torch-compile)
  fi
  PRETRAIN_RESUME="$(latest_checkpoint_dir "$PRETRAIN_DIR" || true)"
  if [[ -n "${PRETRAIN_RESUME:-}" ]]; then
    echo "Resuming pretraining from $PRETRAIN_RESUME"
    PRETRAIN_ARGS+=(--resume-from-checkpoint "$PRETRAIN_RESUME")
  fi
  run_launcher picollm.pretrain_cloud.train "${PRETRAIN_ARGS[@]}"
  prune_checkpoint_dirs "$PRETRAIN_DIR"
fi

require_free_space "$MIN_FREE_GB"
run_pretrain_eval

SFT_ARGS=(
  --model "$PRETRAIN_DIR"
  --output-dir "$CHAT_SFT_DIR"
  --batch-size "$SFT_BATCH_SIZE"
  --grad-accum "$SFT_GRAD_ACCUM"
  --learning-rate "$SFT_LEARNING_RATE"
  --warmup-steps "$SFT_WARMUP_STEPS"
  --optimizer "$SFT_OPTIMIZER"
  --lr-scheduler-type "$SFT_LR_SCHEDULER_TYPE"
  --dataloader-num-workers "$SFT_DATALOADER_WORKERS"
  --save-steps "$SFT_SAVE_STEPS"
  --save-total-limit "$SFT_SAVE_TOTAL_LIMIT"
  --max-steps "$SFT_MAX_STEPS"
  --preview-every-steps "$SFT_PREVIEW_EVERY_STEPS"
  --bf16
  --report-to "$REPORT_TO"
  --run-name "${RUN_NAME}-sft"
)
if [[ "$SFT_MIXTURE" == "nanochat-lite" ]]; then
  SFT_ARGS+=(
    --mixture nanochat-lite
    --smoltalk-epochs "$SFT_SMOLTALK_EPOCHS"
    --everyday-epochs "$SFT_EVERYDAY_EPOCHS"
    --identity-epochs "$SFT_IDENTITY_EPOCHS"
    --mmlu-epochs "$SFT_MMLU_EPOCHS"
    --gsm8k-epochs "$SFT_GSM8K_EPOCHS"
    --simple-spelling-size "$SFT_SIMPLE_SPELLING_SIZE"
    --spelling-bee-size "$SFT_SPELLING_BEE_SIZE"
  )
else
  SFT_ARGS+=(
    --dataset-name "$SFT_DATASET_NAME"
    --dataset-split "$SFT_DATASET_SPLIT"
    --text-column messages
  )
  if [[ -n "$SFT_DATASET_CONFIG" ]]; then
    SFT_ARGS+=(--dataset-config "$SFT_DATASET_CONFIG")
  fi
fi
if [[ "$SFT_TORCH_COMPILE" == "1" ]]; then
  SFT_ARGS+=(--torch-compile)
fi

if model_exists "$CHAT_SFT_DIR"; then
  stage "chat-sft-skip"
  echo "Final chat checkpoint already exists at $CHAT_SFT_DIR. Skipping chat SFT."
else
  stage "chat-sft-train"
  require_free_space "$MIN_FREE_GB"
  SFT_RESUME="$(latest_checkpoint_dir "$CHAT_SFT_DIR" || true)"
  if [[ -n "${SFT_RESUME:-}" ]]; then
    echo "Resuming chat SFT from $SFT_RESUME"
    SFT_ARGS+=(--resume-from-checkpoint "$SFT_RESUME")
  fi
  run_launcher picollm.sft_full.finetune "${SFT_ARGS[@]}"
  prune_checkpoint_dirs "$CHAT_SFT_DIR"
fi

run_chat_eval

if [[ -n "$HF_REPO_ID" ]]; then
  stage "hub-upload"
  uv run python -m picollm.pretrain_cloud.push_to_hub \
    --folder "$CHAT_SFT_DIR" \
    --repo-id "$HF_REPO_ID"
fi

if [[ "$MODE" == "web" ]]; then
  stage "serve-web"
  echo "Speedrun complete. Logs: $LOG_FILE"
  echo "Eval artifacts: $EVAL_DIR"
  exec uv run python -m picollm.serve.chat_web \
    --model "$CHAT_SFT_DIR" \
    --device "$SERVE_DEVICE"
fi

stage "serve-cli"
echo "Speedrun complete. Logs: $LOG_FILE"
echo "Eval artifacts: $EVAL_DIR"
exec uv run python -m picollm.serve.chat_cli \
  --model "$CHAT_SFT_DIR" \
  --device "$SERVE_DEVICE"
