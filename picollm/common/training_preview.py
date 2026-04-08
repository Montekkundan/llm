from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import TrainerCallback

from .chat import generate_reply


def default_pretrain_preview_items() -> list[dict[str, object]]:
    return [
        {
            "label": "story_seed",
            "prompt": "Once upon a time",
        },
        {
            "label": "ml_seed",
            "prompt": "Machine learning is",
        },
    ]


def default_chat_preview_items() -> list[dict[str, object]]:
    return [
        {
            "label": "hello",
            "messages": [{"role": "user", "content": "hi"}],
        },
        {
            "label": "sky_blue",
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        },
    ]


class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        preview_items: Sequence[dict[str, object]],
        *,
        every_steps: int,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.95,
        label: str = "preview",
    ) -> None:
        self.tokenizer = tokenizer
        self.preview_items = list(preview_items)
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.label = label
        self._last_step = -1

    def on_log(self, args, state, control, model=None, **kwargs):
        if (
            self.every_steps <= 0
            or not state.is_world_process_zero
            or state.global_step <= 0
            or state.global_step % self.every_steps != 0
            or state.global_step == self._last_step
            or model is None
            or not self.preview_items
        ):
            return control

        base_model = getattr(model, "module", model)
        was_training = base_model.training
        self._last_step = state.global_step

        try:
            base_model.eval()
            print(f"\n===== {self.label} sample preview @ step {state.global_step} =====", flush=True)
            for item in self.preview_items:
                label = str(item.get("label", "sample"))
                if "messages" in item:
                    messages = item["messages"]
                    response = generate_reply(
                        base_model,
                        self.tokenizer,
                        messages,  # type: ignore[arg-type]
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    prompt_text = ""
                    if isinstance(messages, Sequence) and messages:
                        last_message = messages[-1]
                        if isinstance(last_message, dict):
                            prompt_text = str(last_message.get("content", "")).strip()
                else:
                    prompt_text = str(item.get("prompt", "")).strip()
                    response = self._generate_plain_completion(base_model, prompt_text)

                print(f"[{label}] prompt: {prompt_text}", flush=True)
                print(f"[{label}] reply: {response or '<empty>'}", flush=True)
            print(f"===== end {self.label} preview =====\n", flush=True)
        finally:
            if was_training:
                base_model.train()

        return control

    @torch.inference_mode()
    def _generate_plain_completion(self, model, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(self.temperature, 1e-5)
            generation_kwargs["top_p"] = self.top_p
        generated = model.generate(**generation_kwargs)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()
