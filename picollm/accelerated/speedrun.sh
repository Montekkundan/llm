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
export PICOLLM_ACTIVATION_CHECKPOINTING="0"
export PICOLLM_DEVICE_BATCH_SIZE="16"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-warning}"
mkdir -p "$PICOLLM_BASE_DIR"

WANDB_RUN="${WANDB_RUN:-dummy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-}"
HF_UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
export PYTHONUNBUFFERED=1

unset PICOLLM_FLASH_IMPL
unset PICOLLM_TOTAL_BATCH_SIZE
unset PICOLLM_TRAIN_LOSS_CHUNK_ROWS
unset TORCH_COMPILE_DISABLE

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

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.pretrain.preflight -- \
  --depth=24 \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --fp8

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
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  --fp8 \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.pretrain.eval -- \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE"

curl -L -o "$PICOLLM_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.chat.sft -- \
  --device-batch-size="$PICOLLM_DEVICE_BATCH_SIZE" \
  "${WANDB_ARGS[@]}" \
  --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=8 -m picollm.accelerated.chat.eval -- \
  -i sft

python -m picollm.accelerated.report generate

if [[ -n "$HF_UPLOAD_REPO_ID" ]]; then
  upload_to_hf "$HF_UPLOAD_REPO_ID"
fi

if [[ "$MODE" == "web" ]]; then
  exec python -m picollm.accelerated.chat.web
else
  exec python -m picollm.accelerated.chat.cli
fi
