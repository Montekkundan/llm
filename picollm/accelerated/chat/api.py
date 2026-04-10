from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field


MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MIN_P = 0.0
MAX_MIN_P = 1.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096


class ChatValidationError(ValueError):
    pass


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, ge=1, le=MAX_MAX_TOKENS)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=MAX_MAX_TOKENS)
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationDefaults:
    temperature: float
    top_k: int
    top_p: Optional[float]
    min_p: Optional[float]
    max_tokens: int
    seed: Optional[int]


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    top_k: int
    top_p: Optional[float]
    min_p: Optional[float]
    max_tokens: int
    seed: int


def validate_chat_request(request: ChatRequest) -> None:
    if not request.messages:
        raise ChatValidationError("At least one message is required.")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise ChatValidationError(f"Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed.")

    total_length = 0
    for idx, message in enumerate(request.messages):
        if message.role not in {"system", "user", "assistant"}:
            raise ChatValidationError(f"Message {idx} has invalid role '{message.role}'.")
        if not message.content:
            raise ChatValidationError(f"Message {idx} has empty content.")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise ChatValidationError(f"Message {idx} exceeds {MAX_MESSAGE_LENGTH} characters.")
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise ChatValidationError(f"Conversation exceeds {MAX_TOTAL_CONVERSATION_LENGTH} characters.")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise ChatValidationError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}.")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise ChatValidationError(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}.")
    if request.top_p is not None and not (MIN_TOP_P <= request.top_p <= MAX_TOP_P):
        raise ChatValidationError(f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}.")
    if request.min_p is not None and not (MIN_MIN_P <= request.min_p <= MAX_MIN_P):
        raise ChatValidationError(f"min_p must be between {MIN_MIN_P} and {MAX_MIN_P}.")
    if request.seed is not None and request.seed < 0:
        raise ChatValidationError("seed must be non-negative.")


def resolve_generation_settings(
    request: ChatRequest,
    defaults: GenerationDefaults,
    random_seed_fn,
) -> GenerationSettings:
    max_tokens = request.max_completion_tokens or request.max_tokens or defaults.max_tokens
    seed = request.seed
    if seed is None:
        seed = defaults.seed
    if seed is None:
        seed = random_seed_fn()
    return GenerationSettings(
        temperature=request.temperature if request.temperature is not None else defaults.temperature,
        top_k=request.top_k if request.top_k is not None else defaults.top_k,
        top_p=request.top_p if request.top_p is not None else defaults.top_p,
        min_p=request.min_p if request.min_p is not None else defaults.min_p,
        max_tokens=max_tokens,
        seed=seed,
    )


def build_conversation_tokens(tokenizer, messages: List[ChatMessage]) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos]
    for message in messages:
        content = message.content.strip()
        if message.role == "system":
            content = f"System instruction:\n{content}"
            role = "user"
        else:
            role = message.role

        if role == "user":
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(content))
            tokens.append(user_end)
        elif role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(content))
            tokens.append(assistant_end)

    tokens.append(assistant_start)
    return tokens


def validate_generation_budget(
    tokenizer,
    context_window: int,
    request: ChatRequest,
    defaults: GenerationDefaults,
    random_seed_fn,
) -> tuple[list[int], GenerationSettings]:
    validate_chat_request(request)
    settings = resolve_generation_settings(request, defaults, random_seed_fn=random_seed_fn)
    prompt_tokens = build_conversation_tokens(tokenizer, request.messages)
    prompt_length = len(prompt_tokens)
    remaining_tokens = context_window - prompt_length

    if remaining_tokens <= 0:
        raise ChatValidationError(
            f"Conversation is {prompt_length} tokens after formatting, which exceeds the "
            f"{context_window}-token context window. Start a new conversation or shorten the history."
        )

    if settings.max_tokens > remaining_tokens:
        raise ChatValidationError(
            f"Requested max_tokens={settings.max_tokens}, but only {remaining_tokens} tokens remain in the "
            f"{context_window}-token context window. Reduce max_tokens or shorten the conversation."
        )

    return prompt_tokens, settings
