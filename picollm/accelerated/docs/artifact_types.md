# Artifact Types

There are three different concepts in the current picoLLM workflow.

## picoLLM-Native Checkpoints

These are the training and inference artifacts written by the repo itself under `PICOLLM_BASE_DIR`.

They load through:

- `picollm.accelerated.checkpoint_manager`

## HF Model Repo Visibility

This is just the Hugging Face hosting surface for the lighter inference bundle.

It does not mean the model is automatically compatible with `AutoModelForCausalLM`.

## Transformers Export

This is a separate compatibility target. It means writing a bundle that standard `transformers` loaders can consume directly.

That export path is still separate from the native picoLLM restore path and should be treated as an ecosystem bridge, not the primary runtime.
