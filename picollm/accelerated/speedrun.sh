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
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
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

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Public dataset downloads may still work, but authenticated Hub access is recommended." >&2
fi

upload_to_hf() {
  local repo_id="$1"
  local visibility_flag="--private"
  local restore_readme

  : "${HF_TOKEN:?Set HF_TOKEN to upload artifacts to the Hugging Face Hub}"

  if [[ "$HF_UPLOAD_PRIVATE" == "0" || "$HF_UPLOAD_PRIVATE" == "false" ]]; then
    visibility_flag="--public"
  fi

  echo "Uploading runtime artifacts to Hugging Face: $repo_id"
  hf repos create "$repo_id" --type model "$visibility_flag" --exist-ok --token "$HF_TOKEN"

  hf upload "$repo_id" "$PICOLLM_BASE_DIR/tokenizer" tokenizer \
    --repo-type model \
    --token "$HF_TOKEN" \
    --commit-message "Upload picoLLM tokenizer"

  hf upload "$repo_id" "$PICOLLM_BASE_DIR/base_checkpoints" base_checkpoints \
    --repo-type model \
    --token "$HF_TOKEN" \
    --commit-message "Upload picoLLM base checkpoints"

  hf upload "$repo_id" "$PICOLLM_BASE_DIR/chatsft_checkpoints" chatsft_checkpoints \
    --repo-type model \
    --token "$HF_TOKEN" \
    --commit-message "Upload picoLLM SFT checkpoints"

  if [[ -f "$PICOLLM_BASE_DIR/identity_conversations.jsonl" ]]; then
    hf upload "$repo_id" "$PICOLLM_BASE_DIR/identity_conversations.jsonl" identity_conversations.jsonl \
      --repo-type model \
      --token "$HF_TOKEN" \
      --commit-message "Upload picoLLM SFT data"
  fi

  if [[ -d "$PICOLLM_BASE_DIR/report" ]]; then
    hf upload "$repo_id" "$PICOLLM_BASE_DIR/report" report \
      --repo-type model \
      --token "$HF_TOKEN" \
      --commit-message "Upload picoLLM reports"
  fi

  restore_readme="$(mktemp)"
  cat > "$restore_readme" <<EOF
# picoLLM Speedrun Artifacts

This repo contains the runtime artifacts from a full \`picollm/accelerated/speedrun.sh\` run.

Included:

- \`tokenizer/\`
- \`base_checkpoints/\`
- \`chatsft_checkpoints/\`
- \`report/\` (if generated)
- \`identity_conversations.jsonl\` (if present)

Not included:

- \`base_data_climbmix/\` parquet shards
- the local virtualenv
- W&B logs

## Restore Locally

\`\`\`bash
git clone https://github.com/Montekkundan/llm
cd llm
export PICOLLM_BASE_DIR=\$PWD/artifacts/picollm
hf download $repo_id --repo-type model --local-dir "\$PICOLLM_BASE_DIR"
uv sync --extra gpu
source .venv/bin/activate
python -m picollm.accelerated.chat.cli -i sft
\`\`\`

If you want the latest SFT checkpoint explicitly:

\`\`\`bash
python -m picollm.accelerated.chat.cli -i sft -g d24
\`\`\`
EOF

  hf upload "$repo_id" "$restore_readme" README.md \
    --repo-type model \
    --token "$HF_TOKEN" \
    --commit-message "Upload picoLLM restore instructions"

  rm -f "$restore_readme"
  echo "Finished uploading runtime artifacts to https://huggingface.co/$repo_id"
}

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
  base_checkpoint_path="$(latest_checkpoint_path "$PICOLLM_BASE_DIR/base_checkpoints")"
  sft_checkpoint_path="$(latest_checkpoint_path "$PICOLLM_BASE_DIR/chatsft_checkpoints")"
  report_path="$PICOLLM_BASE_DIR/report"

  echo
  echo "========== Run Summary =========="
  echo "Base checkpoint: ${base_checkpoint_path:-not found}"
  echo "SFT checkpoint: ${sft_checkpoint_path:-not found}"
  if [[ -d "$report_path" ]]; then
    echo "Report path: $report_path"
  else
    echo "Report path: not found"
  fi
  if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
    echo "HF model repo: https://huggingface.co/$HF_UPLOAD_REPO_ID"
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

FP8_ARGS=()
if [[ "$PICOLLM_ENABLE_FP8" == "1" ]]; then
  FP8_ARGS+=(--fp8)
fi

print_stage "Preflight"
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.pretrain.preflight -- \
  --depth=24 \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --window-pattern="$PICOLLM_WINDOW_PATTERN" \
  "${FP8_ARGS[@]}"

print_stage "Dataset Bootstrap"
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
  "${FP8_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

print_stage "Base Eval"
torchrun --standalone --nproc_per_node="$PICOLLM_NPROC_PER_NODE" -m picollm.accelerated.pretrain.eval -- \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE"

IDENTITY_SOURCE="${PICOLLM_IDENTITY_CONVERSATIONS_FILE:-$REPO_ROOT/picollm/accelerated/data/identity_conversations.jsonl}"
if [[ ! -f "$IDENTITY_SOURCE" ]]; then
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

if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
  print_stage "HF Upload"
  upload_to_hf "$HF_UPLOAD_REPO_ID"
fi

print_run_summary

if [[ "$MODE" == "web" ]]; then
  print_stage "Launch Web Chat"
  exec python -m picollm.accelerated.chat.web
else
  print_stage "Launch CLI Chat"
  exec python -m picollm.accelerated.chat.cli
fi
