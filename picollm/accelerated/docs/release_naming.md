# Release Naming

Use versioned model names for public demos instead of ad hoc repo ids.

Recommended pattern:

- `picollm-<release-name>-<model-tag>-sft<step>`
- `picollm-<release-name>-<model-tag>-sft<step>-archive`

Examples:

- `picollm-april-h200-run-d24-sft004000`
- `picollm-april-h200-run-d24-sft004000-archive`

Generate them with:

```bash
python scripts/release_picollm_to_hf.py --namespace your-username --release-name april-h200-run --latest-repo-id your-username/picollm-latest
```
