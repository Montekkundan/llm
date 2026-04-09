from __future__ import annotations

from threading import Thread
from typing import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from .conversation import build_prompt, encode_messages_for_generation, normalize_messages, stop_token_ids


def _prepare_inputs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    max_new_tokens: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    encoded_ids = encode_messages_for_generation(tokenizer, messages)
    encoded = {
        "input_ids": torch.tensor([encoded_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(encoded_ids)), dtype=torch.long),
    }
    max_positions = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    generation_max_new_tokens = max_new_tokens
    if max_positions is not None:
        max_positions = int(max_positions)
        reserve_for_prompt = max(1, max_positions - max_new_tokens)
        if int(encoded["input_ids"].shape[1]) > reserve_for_prompt:
            encoded = {
                key: value[:, -reserve_for_prompt:]
                for key, value in encoded.items()
            }
        prompt_length = int(encoded["input_ids"].shape[1])
        generation_max_new_tokens = max(1, min(max_new_tokens, max_positions - prompt_length))
    device = getattr(model, "device", torch.device("cpu"))
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    return encoded, prompt_length, generation_max_new_tokens


@torch.inference_mode()
def generate_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    encoded, prompt_length, generation_max_new_tokens = _prepare_inputs(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )
    eos_token_ids = stop_token_ids(tokenizer)
    do_sample = temperature > 0
    generation_kwargs = {
        **encoded,
        "max_new_tokens": generation_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if eos_token_ids:
        generation_kwargs["eos_token_id"] = eos_token_ids
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
    encoded, _, generation_max_new_tokens = _prepare_inputs(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    eos_token_ids = stop_token_ids(tokenizer)
    do_sample = temperature > 0
    generation_kwargs = {
        **encoded,
        "max_new_tokens": generation_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if eos_token_ids:
        generation_kwargs["eos_token_id"] = eos_token_ids
    if do_sample:
        generation_kwargs["temperature"] = max(temperature, 1e-5)
        generation_kwargs["top_p"] = top_p
    worker = Thread(target=model.generate, kwargs=generation_kwargs)
    worker.start()
    for piece in streamer:
        yield piece
    worker.join()
