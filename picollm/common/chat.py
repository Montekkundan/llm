from __future__ import annotations

from threading import Thread
from typing import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer


def normalize_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": message["role"].strip(),
                "content": message["content"].strip(),
            }
        )
    return normalized


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    normalized = normalize_messages(messages)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    lines: list[str] = []
    for message in normalized:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n".join(lines)


def decode_assistant_text(full_text: str, prompt_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()


def _prepare_inputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
) -> tuple[str, dict[str, torch.Tensor], int]:
    prompt = build_prompt(tokenizer, messages, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", torch.device("cpu"))
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    return prompt, encoded, prompt_length


@torch.inference_mode()
def generate_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    _, encoded, prompt_length = _prepare_inputs(model, tokenizer, messages)
    do_sample = temperature > 0
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = top_p
    generated = model.generate(**generation_kwargs)
    new_tokens = generated[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def stream_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    _, encoded, _ = _prepare_inputs(model, tokenizer, messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    do_sample = temperature > 0
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = top_p
    worker = Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    for piece in streamer:
        yield piece
    worker.join()
