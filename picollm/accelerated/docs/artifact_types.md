# Artifact Types

There are four different concepts in the current picoLLM workflow.

## picoLLM-Native Checkpoints

These are the training and inference artifacts written by the repo itself under `PICOLLM_BASE_DIR`.

They load through:

- `picollm.accelerated.checkpoint_manager`

## HF Model Repo Visibility

This is just the Hugging Face hosting surface for the lighter inference bundle.

It does not mean the model is automatically compatible with `AutoModelForCausalLM`.

## Transformers Export

This is a separate compatibility target. It means writing a bundle that standard `transformers` loaders can consume directly.

The export helper now exists:

- `python scripts/export_picollm_to_transformers.py --source sft`

It writes a `trust_remote_code` package with:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `configuration_picollm.py`
- `modeling_picollm.py`

That export path is still separate from the native picoLLM restore path and should be treated as an ecosystem bridge, not the primary runtime.

## GGUF Export

This is a second ecosystem bridge target for packaging the native checkpoint into GGUF format.

The export helper now exists:

- `python scripts/export_picollm_to_gguf.py --source sft`

Important limitation:

- picoLLM is a custom architecture, so stock llama.cpp does not currently include a picoLLM runtime implementation
- the GGUF file is useful as a portable bridge artifact, but it is not a drop-in replacement for the native picoLLM runtime today
