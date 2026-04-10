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
- source prompts and answers are embedded directly in the builder script
- the current version is a fully original picoLLM-native dataset and no longer depends on the repo-root migration file

Integrity and hosting:

- the manifest records the canonical row count and SHA-256 checksum
- the intended hosted mirror is `https://assets.montek.dev/identity_conversations.jsonl`
- after upload, verify the hosted copy with `python scripts/verify_identity_asset.py`
