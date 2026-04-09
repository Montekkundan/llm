# picoLLM Code TODO

This file is for quality improvements, cleanup, branding independence, packaging, and guardrails.

Principle:
- Keep the current training and inference behavior stable unless a change is explicitly approved.
- Prefer bug fixes, ergonomics, test coverage, portability, and removal of accidental `nanochat` dependence.

## P0: Identity And Branding Independence

- [ ] Replace the current rewritten identity dataset with a fully picoLLM-native generator and source prompts so we are not derived from Karpathy's identity file at all.
- [ ] Add a small branded identity regression suite:
  - ask `Who are you?`
  - ask `Who created you?`
  - ask `What project are you part of?`
  - fail if the answer contains `nanochat`, `Andrej`, or `Karpathy`
- [ ] Audit the repo for remaining `nanochat` wording in code paths that affect runtime UX, model prompts, logs, UI strings, model cards, and generated artifacts.
- [ ] Move all remaining runtime downloads that point to Karpathy-hosted URLs to picoLLM-owned local files or picoLLM-owned hosted artifacts where appropriate.
- [ ] Version the identity dataset under `picollm/accelerated/data/` and document the schema, source, and generation process.
- [ ] Add a tiny `identity smoke test` command that loads the latest SFT checkpoint and checks a few branding questions automatically.

## P0: Inference Controls And Chat UX

- [ ] Add `--max-tokens` to [picollm/accelerated/chat/cli.py](/Users/montekkundan/Developer/ML/llm/picollm/accelerated/chat/cli.py) instead of hardcoding `256`.
- [ ] Add `--top-p` to CLI generation controls.
- [ ] Evaluate adding `--min-p` to CLI generation controls if we want a cleaner low-probability cutoff than `top-k` alone.
- [ ] Add `--seed` to CLI so single-prompt behavior can be reproduced exactly.
- [ ] Add `--system-prompt` or `--system-file` to CLI so targeted steering can be tested without code edits.
- [ ] Expose the same generation controls in the web UI:
  - temperature
  - top-k
  - top-p
  - max tokens
  - seed
- [ ] Persist those web UI generation controls in local storage.
- [ ] Show loaded model source, tag, step, and device in the web UI footer or settings panel.
- [ ] Add a simple `reset generation settings` button in the web UI.

## P0: Recovery, Resume, And HF Workflow

- [ ] Add a documented `resume-sft-only` path so branding or chat-style fixes do not require rerunning base pretraining.
- [ ] Add a helper script for `upload inference model to HF` so the model-repo workflow is one command instead of a manual series of `hf upload` calls.
- [ ] Add a helper script for `upload archive dataset to HF` so reports, eval outputs, and checkpoints are grouped consistently.
- [ ] Add a helper script for `download from HF and run locally` that sets `PICOLLM_BASE_DIR` correctly and verifies the required files exist.
- [ ] Add an automatic post-upload smoke test that downloads the just-uploaded model repo into a fresh temp dir and runs one CLI prompt.
- [ ] Add optional periodic checkpoint sync during long runs so a machine loss does not waste the full run.
- [ ] Separate `inference artifacts` from `resume-training artifacts` clearly in docs and upload helpers.

## P1: Warnings, Logging, And Rough Edges

- [ ] Replace deprecated `PYTORCH_CUDA_ALLOC_CONF` usage with `PYTORCH_ALLOC_CONF` in training scripts and `speedrun.sh`.
- [ ] Reduce noisy repeated HF warning output during runs by centralizing Hub auth messaging.
- [ ] Make `speedrun.sh` print a short stage banner before each major phase:
  - tokenizer
  - base pretrain
  - base eval
  - SFT
  - chat eval
  - report
  - HF upload
- [ ] Print a clear `identity source:` line during SFT startup so the operator knows exactly which identity dataset was used.
- [ ] Improve missing-file hints in runtime errors so they mention the repo-local picoLLM paths first, not old external URLs.
- [ ] Add a compact end-of-run summary with:
  - base checkpoint path
  - SFT checkpoint path
  - report path
  - optional HF repo ids

## P1: Tests And Regression Coverage

- [ ] Add unit tests for checkpoint discovery and loading:
  - latest-step resolution
  - missing optimizer handling
  - base vs SFT repo layout
- [ ] Add a test that `speedrun.sh` uses the repo-local identity file by default.
- [ ] Add a test that `CustomJSON` accepts the picoLLM identity dataset and loads exactly `1000` rows.
- [ ] Add a test that the generated identity file contains zero forbidden terms.
- [ ] Add a smoke test for:
  - `python -m picollm.accelerated.chat.cli -i sft -p "..."`
  - `python -m picollm.accelerated.chat.web`
- [ ] Add a restore test that downloads a model repo into a fresh temp directory and loads it through `checkpoint_manager`.

## P1: Data And Artifact Independence

- [ ] Stop relying on external one-off S3 files for important runtime assets where reasonable.
- [ ] Decide which external assets should be mirrored under picoLLM control:
  - identity conversations
  - optional eval bundle copies or manifests
  - model-card templates
  - example restore scripts
- [ ] Add explicit manifests for important public dependencies so the exact upstream source is recorded without reuploading huge datasets like ClimbMix.
- [ ] Generate a machine-readable `run_manifest.json` after every speedrun with:
  - repo commit
  - torch version
  - detected hardware
  - chosen speedrun config
  - identity dataset path
  - base checkpoint step
  - SFT checkpoint step

## P1: Local And Cross-Platform Developer Experience

- [ ] Document the proper local smoke-test flow for macOS separately from CUDA boxes.
- [ ] Add a one-command local test target for:
  - CPU
  - MPS
  - CUDA
- [ ] Make `uv sync --extra cpu` and `uv sync --extra gpu` the clearly documented paths in all relevant docs.
- [ ] Add a small script that prints the exact environment commands needed for the current platform.
- [ ] Make the web server print the local URL and loaded checkpoint more prominently on startup.

## P1: Model Repo And Packaging Quality

- [ ] Improve the HF model card so it includes:
  - exact checkpoint provenance
  - intended usage
  - known limitations
  - local restore commands
  - note that it is picoLLM-native, not Transformers-native
- [ ] Add a small metadata file in the model repo describing:
  - model family name
  - checkpoint type
  - tokenizer path
  - compatible picoLLM commit range
- [ ] Add a helper for creating a `latest` alias or documented stable repo naming convention for releases.

## P2: Web UI Quality

- [ ] Add a small model/settings drawer in the web UI instead of hiding defaults in server args only.
- [ ] Show response generation status more clearly while tokens stream.
- [ ] Add a copy button for assistant messages.
- [ ] Add a `clear conversation` button in the main UI, not only command-style behavior.
- [ ] Show a short disclaimer in the UI that picoLLM is a smaller open model and can hallucinate.
- [ ] Add a visible badge for the loaded source:
  - `base`
  - `sft`

## P2: Docs And Runbooks

- [ ] Write a dedicated `branding + identity` doc explaining how the identity data affects the SFT model.
- [ ] Write a `re-run only SFT` guide for fast iteration on assistant behavior.
- [ ] Write a `HF backup strategy` guide covering:
  - inference-only artifacts
  - archive artifacts
  - resume-training artifacts
  - what not to upload
- [ ] Update docs to explain the difference between:
  - picoLLM-native checkpoints
  - HF model repo visibility
  - Transformers-compatible exports

## P2: Future Nice-To-Haves

- [ ] Add a proper export path to standard Hugging Face Transformers format.
- [ ] Add GGUF export for local llama.cpp-style use.
- [ ] Add a small benchmark comparison doc for:
  - base model
  - SFT model
  - later branded reruns
- [ ] Add release tags or named checkpoints for important public demos.

## Cleanup Notes

- [ ] Remove the legacy repo-root `identity_conversations.jsonl` once we no longer need it as a local migration source.
- [ ] Keep `base_data_climbmix/` out of HF backup defaults.
- [ ] Keep optimizer shards out of inference-focused HF model repos by default.
