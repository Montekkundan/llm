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
if [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${PYTORCH_ALLOC_CONF:-}" ]]; then
  export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-warning}"
mkdir -p "$PICOLLM_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-}"
HF_ARCHIVE_REPO_ID="${HF_ARCHIVE_REPO_ID:-}"
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
HF_PERIODIC_SYNC="${HF_PERIODIC_SYNC:-0}"
HF_SYNC_INTERVAL_SECONDS="${HF_SYNC_INTERVAL_SECONDS:-900}"
HF_SYNC_MIN_FILE_AGE_SECONDS="${HF_SYNC_MIN_FILE_AGE_SECONDS:-180}"
PICOLLM_BASE_SAVE_EVERY="${PICOLLM_BASE_SAVE_EVERY:--1}"
export PYTHONUNBUFFERED=1

unset PICOLLM_TRAIN_LOSS_CHUNK_ROWS

if [[ "$WANDB_RUN" != "dummy" ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY (or use WANDB_RUN=dummy)}"
  : "${WANDB_ENTITY:?Set WANDB_ENTITY (or use WANDB_RUN=dummy)}"
fi

WANDB_ARGS=()
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb-entity="$WANDB_ENTITY")
fi

if [[ "$HF_PERIODIC_SYNC" == "1" || "$HF_PERIODIC_SYNC" == "true" ]]; then
  : "${HF_ARCHIVE_REPO_ID:?Set HF_ARCHIVE_REPO_ID when enabling HF_PERIODIC_SYNC}"
  if [[ "$PICOLLM_BASE_SAVE_EVERY" == "-1" ]]; then
    PICOLLM_BASE_SAVE_EVERY=1000
  fi
fi

PERIODIC_SYNC_PID=""

upload_to_hf() {
  local repo_id="$1"
  local visibility_flag=()
  : "${HF_TOKEN:?Set HF_TOKEN to upload artifacts to the Hugging Face Hub}"

  if [[ "$HF_UPLOAD_PRIVATE" == "0" || "$HF_UPLOAD_PRIVATE" == "false" ]]; then
    visibility_flag+=(--public)
  fi

  python scripts/upload_picollm_model_to_hf.py \
    "$repo_id" \
    --base-dir "$PICOLLM_BASE_DIR" \
    --post-upload-smoke \
    "${visibility_flag[@]}"
}

upload_archive_to_hf() {
  local repo_id="$1"
  local visibility_flag=()
  : "${HF_TOKEN:?Set HF_TOKEN to upload artifacts to the Hugging Face Hub}"

  if [[ "$HF_UPLOAD_PRIVATE" == "0" || "$HF_UPLOAD_PRIVATE" == "false" ]]; then
    visibility_flag+=(--public)
  fi

  python scripts/upload_picollm_archive_to_hf.py \
    "$repo_id" \
    --base-dir "$PICOLLM_BASE_DIR" \
    "${visibility_flag[@]}"
}

start_periodic_archive_sync() {
  if [[ "$HF_PERIODIC_SYNC" != "1" && "$HF_PERIODIC_SYNC" != "true" ]]; then
    return
  fi
  : "${HF_TOKEN:?Set HF_TOKEN to upload artifacts to the Hugging Face Hub}"

  local visibility_flag=()
  if [[ "$HF_UPLOAD_PRIVATE" == "0" || "$HF_UPLOAD_PRIVATE" == "false" ]]; then
    visibility_flag+=(--public)
  fi

  echo "HF periodic sync: archive repo=$HF_ARCHIVE_REPO_ID interval=${HF_SYNC_INTERVAL_SECONDS}s min_file_age=${HF_SYNC_MIN_FILE_AGE_SECONDS}s save_every=${PICOLLM_BASE_SAVE_EVERY}"
  python scripts/sync_picollm_archive_to_hf.py \
    "$HF_ARCHIVE_REPO_ID" \
    --base-dir "$PICOLLM_BASE_DIR" \
    --interval-seconds "$HF_SYNC_INTERVAL_SECONDS" \
    --min-file-age-seconds "$HF_SYNC_MIN_FILE_AGE_SECONDS" \
    "${visibility_flag[@]}" &
  PERIODIC_SYNC_PID=$!
}

stop_periodic_archive_sync() {
  if [[ -n "$PERIODIC_SYNC_PID" ]]; then
    kill "$PERIODIC_SYNC_PID" >/dev/null 2>&1 || true
    wait "$PERIODIC_SYNC_PID" >/dev/null 2>&1 || true
    PERIODIC_SYNC_PID=""
  fi
}

trap stop_periodic_archive_sync EXIT

print_stage() {
  local name="$1"
  echo
  echo "========== $name =========="
}

latest_checkpoint_path() {
  local checkpoints_dir="$1"
  if [[ ! -d "$checkpoints_dir" ]]; then
    return 0
  fi
  python - <<'PY' "$checkpoints_dir"
from pathlib import Path
import sys

from picollm.accelerated.checkpoint_manager import find_largest_model, find_last_step

root = Path(sys.argv[1])
if not root.exists() or not any(root.iterdir()):
    raise SystemExit(0)
model_tag = find_largest_model(str(root))
step = find_last_step(str(root / model_tag))
print(root / model_tag / f"model_{step:06d}.pt")
PY
}

print_run_summary() {
  local base_checkpoint_path
  local sft_checkpoint_path
  local report_path
  local run_manifest_path
  base_checkpoint_path="$(latest_checkpoint_path "$PICOLLM_BASE_DIR/base_checkpoints")"
  sft_checkpoint_path="$(latest_checkpoint_path "$PICOLLM_BASE_DIR/chatsft_checkpoints")"
  report_path="$PICOLLM_BASE_DIR/report"
  run_manifest_path="$PICOLLM_BASE_DIR/run_manifest.json"

  echo
  echo "========== Run Summary =========="
  echo "Base checkpoint: ${base_checkpoint_path:-not found}"
  echo "SFT checkpoint: ${sft_checkpoint_path:-not found}"
  if [[ -d "$report_path" ]]; then
    echo "Report path: $report_path"
  else
    echo "Report path: not found"
  fi
  if [[ -f "$run_manifest_path" ]]; then
    echo "Run manifest: $run_manifest_path"
  else
    echo "Run manifest: not found"
  fi
  if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
    echo "HF model repo: https://huggingface.co/$HF_UPLOAD_REPO_ID"
  fi
  if [[ -n "$HF_ARCHIVE_REPO_ID" ]]; then
    echo "HF archive dataset: https://huggingface.co/datasets/$HF_ARCHIVE_REPO_ID"
  fi
}

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

eval "$(python -m picollm.accelerated.speedrun_config --format shell)"

if [[ "$PICOLLM_SUPPORTED" != "1" ]]; then
  echo "$PICOLLM_UNSUPPORTED_REASON" >&2
  exit 1
fi

if [[ "$PICOLLM_FORCE_FLASH_IMPL" == "sdpa" ]]; then
  export PICOLLM_FLASH_IMPL=sdpa
else
  unset PICOLLM_FLASH_IMPL
fi

export PICOLLM_ACTIVATION_CHECKPOINTING
export PICOLLM_DEVICE_BATCH_SIZE
export PICOLLM_TOTAL_BATCH_SIZE

echo "Speedrun hardware: $PICOLLM_HARDWARE_SUMMARY"
echo "Speedrun settings: $PICOLLM_SETTINGS_SUMMARY"
if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
  echo "HF upload target (model repo): $HF_UPLOAD_REPO_ID"
fi
if [[ -n "$HF_ARCHIVE_REPO_ID" ]]; then
  echo "HF upload target (archive dataset): $HF_ARCHIVE_REPO_ID"
fi
if [[ -n "$HF_UPLOAD_REPO_ID" && -n "$HF_ARCHIVE_REPO_ID" ]]; then
  echo "HF destinations: model repo is for runnable artifacts, archive dataset is for fuller run history."
fi
if [[ "$HF_PERIODIC_SYNC" == "1" || "$HF_PERIODIC_SYNC" == "true" ]]; then
  echo "HF periodic archive sync: enabled"
fi
if [[ -n "${PICOLLM_IDENTITY_CONVERSATIONS_URL:-}" ]]; then
  echo "Hosted identity mirror: $PICOLLM_IDENTITY_CONVERSATIONS_URL"
fi

FP8_ARGS=()
if [[ "$PICOLLM_ENABLE_FP8" == "1" ]]; then
  FP8_ARGS+=(--fp8)
fi

print_stage "Preflight"
python -m picollm.accelerated.speedrun_doctor
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.pretrain.preflight -- \
  --depth=24 \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --window-pattern="$PICOLLM_WINDOW_PATTERN" \
  "${FP8_ARGS[@]}"

print_stage "Dataset Bootstrap"
start_periodic_archive_sync
python -m picollm.accelerated.report reset

python -m picollm.accelerated.dataset -n 8
python -m picollm.accelerated.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

print_stage "Tokenizer"
python -m picollm.accelerated.pretrain.train_tokenizer
python -m picollm.accelerated.pretrain.tokenizer_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

print_stage "Base Pretrain"
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.pretrain.train -- \
  --depth=24 \
  --window-pattern="$PICOLLM_WINDOW_PATTERN" \
  --target-param-data-ratio=8 \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --total-batch-size="$PICOLLM_TOTAL_BATCH_SIZE" \
  --save-every="$PICOLLM_BASE_SAVE_EVERY" \
  "${FP8_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

print_stage "Base Eval"
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.pretrain.eval -- \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE"

IDENTITY_SOURCE="${PICOLLM_IDENTITY_CONVERSATIONS_FILE:-$REPO_ROOT/picollm/accelerated/data/identity_conversations.jsonl}"
IDENTITY_MANIFEST="${PICOLLM_IDENTITY_CONVERSATIONS_MANIFEST:-$REPO_ROOT/picollm/accelerated/data/identity_conversations.manifest.json}"
IDENTITY_URL="${PICOLLM_IDENTITY_CONVERSATIONS_URL:-}"
if [[ -n "$IDENTITY_URL" ]]; then
  IDENTITY_SOURCE="$PICOLLM_BASE_DIR/identity_sources/identity_conversations.jsonl"
  echo "Fetching identity dataset from hosted mirror: $IDENTITY_URL"
  python scripts/verify_identity_asset.py \
    --manifest "$IDENTITY_MANIFEST" \
    --hosted-url "$IDENTITY_URL" \
    --download-to "$IDENTITY_SOURCE"
elif [[ ! -f "$IDENTITY_SOURCE" ]]; then
  echo "Missing identity conversations file: $IDENTITY_SOURCE" >&2
  echo "Set PICOLLM_IDENTITY_CONVERSATIONS_FILE to override, or commit the repo-local picoLLM identity file." >&2
  exit 1
fi
print_stage "SFT"
echo "Identity source: $IDENTITY_SOURCE"
cp "$IDENTITY_SOURCE" "$PICOLLM_BASE_DIR/identity_conversations.jsonl"

torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.chat.sft -- \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --total-batch-size="$PICOLLM_TOTAL_BATCH_SIZE" \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

print_stage "Chat Eval"
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.chat.eval -- \
  -i sft \
  -b "$PICOLLM_CHAT_EVAL_BATCH_SIZE"

print_stage "Report"
python -m picollm.accelerated.report generate
python scripts/write_picollm_run_manifest.py --base-dir "$PICOLLM_BASE_DIR" --identity-source "$IDENTITY_SOURCE"

if [[ -n "$HF_UPLOAD_REPO_ID" || -n "$HF_ARCHIVE_REPO_ID" ]]; then
  print_stage "HF Upload"
fi
stop_periodic_archive_sync
if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
  upload_to_hf "$HF_UPLOAD_REPO_ID"
fi
if [[ -n "$HF_ARCHIVE_REPO_ID" ]]; then
  upload_archive_to_hf "$HF_ARCHIVE_REPO_ID"
fi

print_run_summary

if [[ "$MODE" == "web" ]]; then
  print_stage "Launch Web Chat"
  exec python -m picollm.accelerated.chat.web
else
  print_stage "Launch CLI Chat"
  exec python -m picollm.accelerated.chat.cli
fi
