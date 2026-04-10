# picoLLM Identity Data

Canonical runtime dataset:

- `identity_conversations.jsonl`

Canonical manifest:

- `identity_conversations.manifest.json`

Schema:

- one JSON array per line
- every array is a single conversation
- messages alternate `user` then `assistant`
- every message object must contain `role` and `content`

Current source and generation process:

- built with `python scripts/build_picollm_identity_dataset.py`
- source input is the repo-root `identity_conversations.jsonl` migration file
- the current version is a rewrite-derived stopgap, not the final first-principles picoLLM-native identity generator

Integrity and hosting:

- the manifest records the canonical row count and SHA-256 checksum
- the intended hosted mirror is `https://assets.montek.dev/identity_conversations.jsonl`
- after upload, verify the hosted copy with `python scripts/verify_identity_asset.py`
