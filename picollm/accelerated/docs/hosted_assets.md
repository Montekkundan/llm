# Hosted picoLLM Assets

The repo-local copies stay canonical. Hosted mirrors exist only for remote convenience.

Mirror under `assets.montek.dev`:

- `identity_conversations.jsonl`
- `eval_bundle.zip`
- `words_alpha.txt`

Keep upstream-only by design:

- ClimbMix parquet shards

Source of truth:

- `picollm/accelerated/data/identity_conversations.manifest.json`
- `picollm/accelerated/data/public_dependencies.manifest.json`

Workflow:

1. Update the repo-local canonical file.
2. Verify it locally.
3. Upload the exact bytes to `assets.montek.dev`.
4. Re-run the verifier against the hosted URL.

Commands:

```bash
python scripts/verify_identity_asset.py --local-only
python scripts/verify_identity_asset.py
```
