# Branding And Identity

The SFT model learns brand language directly from the supervised conversation mixture.

Important inputs:

- `picollm/accelerated/data/identity_conversations.jsonl`
- `picollm/accelerated/chat/sft.py`
- `python -m picollm.accelerated.chat.identity_smoke`

Rules for identity work:

- keep the canonical dataset in `picollm/accelerated/data/`
- regenerate it with `python scripts/build_picollm_identity_dataset.py`
- verify it with `python scripts/verify_identity_asset.py --local-only`
- run `python -m picollm.accelerated.chat.identity_smoke --dataset-only`
- run the full identity smoke against the latest SFT checkpoint before publishing

Current branding policy:

- the assistant should identify as picoLLM
- runtime-facing answers should not say `nano`, `nanochat`, `Andrej`, or `Karpathy`
- regression checks enforce those forbidden terms in both the dataset and the latest SFT checkpoint output
