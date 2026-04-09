from __future__ import annotations

from typing import Iterable

from transformers import PreTrainedTokenizerBase


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"

STRUCTURED_ROLE_TOKENS: dict[str, tuple[str, str]] = {
    "system": ("<|system_start|>", "<|system_end|>"),
    "user": ("<|user_start|>", "<|user_end|>"),
    "assistant": ("<|assistant_start|>", "<|assistant_end|>"),
}

LEGACY_ROLE_TOKENS: dict[str, str] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    *[token for pair in STRUCTURED_ROLE_TOKENS.values() for token in pair],
    *LEGACY_ROLE_TOKENS.values(),
]

# GPT-4-style split pattern adapted from nanochat's tokenizer.
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def normalize_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message["role"]).strip().lower()
        content = str(message["content"]).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def merge_leading_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) >= 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
        merged = [message.copy() for message in messages]
        merged[1]["content"] = f"{merged[0]['content']}\n\n{merged[1]['content']}"
        return merged[1:]
    return [message.copy() for message in messages]


def _token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        return None
    return int(token_id)


def _encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def has_structured_chat_tokens(tokenizer: PreTrainedTokenizerBase) -> bool:
    return all(
        _token_id(tokenizer, token) is not None
        for pair in STRUCTURED_ROLE_TOKENS.values()
        for token in pair
    )


def has_legacy_chat_tokens(tokenizer: PreTrainedTokenizerBase) -> bool:
    return all(_token_id(tokenizer, token) is not None for token in LEGACY_ROLE_TOKENS.values())


def build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    normalized = merge_leading_system_into_user(normalize_messages(messages))
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if has_structured_chat_tokens(tokenizer):
        parts: list[str] = []
        if _token_id(tokenizer, BOS_TOKEN) is not None:
            parts.append(BOS_TOKEN)
        for message in normalized:
            start_token, end_token = STRUCTURED_ROLE_TOKENS[message["role"]]
            parts.append(start_token)
            parts.append(message["content"])
            parts.append(end_token)
        if add_generation_prompt:
            parts.append(STRUCTURED_ROLE_TOKENS["assistant"][0])
        return "".join(parts)

    lines: list[str] = []
    for message in normalized:
        token = LEGACY_ROLE_TOKENS.get(message["role"], f"<|{message['role']}|>")
        lines.append(f"{token} {message['content']}")
    if add_generation_prompt:
        lines.append(LEGACY_ROLE_TOKENS["assistant"])
    return "\n".join(lines)


def encode_messages_for_generation(
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    *,
    max_length: int | None = None,
) -> list[int]:
    normalized = merge_leading_system_into_user(normalize_messages(messages))
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(normalized, tokenize=True, add_generation_prompt=True)
        return list(ids[-max_length:]) if max_length is not None else list(ids)

    if has_structured_chat_tokens(tokenizer):
        ids: list[int] = []
        bos_id = _token_id(tokenizer, BOS_TOKEN)
        assistant_start_id = _token_id(tokenizer, STRUCTURED_ROLE_TOKENS["assistant"][0])
        if bos_id is not None:
            ids.append(bos_id)
        for message in normalized:
            start_token, end_token = STRUCTURED_ROLE_TOKENS[message["role"]]
            start_id = _token_id(tokenizer, start_token)
            end_id = _token_id(tokenizer, end_token)
            if start_id is None or end_id is None:
                raise ValueError("Tokenizer is missing one or more structured chat tokens.")
            ids.append(start_id)
            ids.extend(_encode_text(tokenizer, message["content"]))
            ids.append(end_id)
        if assistant_start_id is not None:
            ids.append(assistant_start_id)
        return ids[-max_length:] if max_length is not None else ids

    prompt = build_prompt(tokenizer, normalized, add_generation_prompt=True)
    ids = list(tokenizer(prompt)["input_ids"])
    return ids[-max_length:] if max_length is not None else ids


def encode_messages_for_training(
    tokenizer: PreTrainedTokenizerBase,
    messages: Iterable[dict[str, str]],
    *,
    max_length: int | None = None,
) -> dict[str, list[int]] | None:
    normalized = merge_leading_system_into_user(normalize_messages(messages))
    if not normalized:
        return None

    if has_structured_chat_tokens(tokenizer):
        input_ids: list[int] = []
        labels: list[int] = []

        bos_id = _token_id(tokenizer, BOS_TOKEN)
        if bos_id is not None:
            input_ids.append(bos_id)
            labels.append(-100)

        for message in normalized:
            start_token, end_token = STRUCTURED_ROLE_TOKENS[message["role"]]
            start_id = _token_id(tokenizer, start_token)
            end_id = _token_id(tokenizer, end_token)
            if start_id is None or end_id is None:
                return None

            input_ids.append(start_id)
            labels.append(-100)

            content_ids = _encode_text(tokenizer, message["content"])
            input_ids.extend(content_ids)
            if message["role"] == "assistant":
                labels.extend(content_ids)
            else:
                labels.extend([-100] * len(content_ids))

            input_ids.append(end_id)
            labels.append(end_id if message["role"] == "assistant" else -100)

        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

        if not any(label != -100 for label in labels):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    prompt_lines: list[str] = []
    completion: str | None = None
    for message in normalized:
        if message["role"] == "assistant":
            completion = message["content"]
            break
        role_token = LEGACY_ROLE_TOKENS.get(message["role"], f"<|{message['role']}|>")
        prompt_lines.append(f"{role_token} {message['content']}")
    if completion is None:
        return None
    prompt_lines.append(LEGACY_ROLE_TOKENS["assistant"])
    prompt_ids = _encode_text(tokenizer, "\n".join(prompt_lines))
    completion_ids = _encode_text(tokenizer, f" {completion}")
    if tokenizer.eos_token_id is not None:
        completion_ids.append(int(tokenizer.eos_token_id))
    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + completion_ids[:]
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    if not any(label != -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    assistant_end_id = _token_id(tokenizer, STRUCTURED_ROLE_TOKENS["assistant"][1])
    if assistant_end_id is not None:
        token_ids.append(assistant_end_id)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(token_ids))
