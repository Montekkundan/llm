from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from picollm.common import generate_reply, load_generation_bundle, stream_reply
from picollm.common.loading import metadata_for_bundle


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


MODEL_NAME = os.environ.get("PICOLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("PICOLLM_ADAPTER")
DEVICE = os.environ.get("PICOLLM_DEVICE", "auto")
QUANTIZATION = os.environ.get("PICOLLM_QUANTIZATION", "none")
UI_PATH = Path(__file__).with_name("ui") / "index.html"

app = FastAPI(title="picoLLM Chat App", version="0.1.0")


def _resolve_max_new_tokens(request: ChatRequest) -> int:
    return request.max_completion_tokens or request.max_tokens or request.max_new_tokens


@lru_cache(maxsize=1)
def _get_base_bundle():
    return load_generation_bundle(MODEL_NAME, device=DEVICE, quantization=QUANTIZATION)


@lru_cache(maxsize=1)
def _get_adapter_bundle():
    return load_generation_bundle(MODEL_NAME, adapter_path=ADAPTER, device=DEVICE, quantization=QUANTIZATION)


def _get_bundle():
    return _get_adapter_bundle() if ADAPTER else _get_base_bundle()


def _require_adapter() -> None:
    if not ADAPTER:
        raise HTTPException(status_code=404, detail="No adapter configured for compare mode.")


def _generate_for_bundle(bundle, request: ChatRequest) -> str:
    return generate_reply(
        bundle.model,
        bundle.tokenizer,
        [message.model_dump() for message in request.messages],
        max_new_tokens=_resolve_max_new_tokens(request),
        temperature=request.temperature,
        top_p=request.top_p,
    )


_STREAM_DONE = object()


def _next_stream_piece(iterator):
    return next(iterator, _STREAM_DONE)


async def _stream_text_response(bundle, request: ChatRequest):
    iterator = iter(
        stream_reply(
            bundle.model,
            bundle.tokenizer,
            [message.model_dump() for message in request.messages],
            max_new_tokens=_resolve_max_new_tokens(request),
            temperature=request.temperature,
            top_p=request.top_p,
        )
    )
    while True:
        piece = await asyncio.to_thread(_next_stream_piece, iterator)
        if piece is _STREAM_DONE:
            break
        if not piece:
            continue
        yield f"data: {json.dumps({'token': piece})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_openai_pieces(bundle, request: ChatRequest):
    iterator = iter(
        stream_reply(
            bundle.model,
            bundle.tokenizer,
            [message.model_dump() for message in request.messages],
            max_new_tokens=_resolve_max_new_tokens(request),
            temperature=request.temperature,
            top_p=request.top_p,
        )
    )
    while True:
        piece = await asyncio.to_thread(_next_stream_piece, iterator)
        if piece is _STREAM_DONE:
            break
        if not piece:
            continue
        yield piece


async def _stream_openai_response(request: ChatRequest):
    bundle = _get_bundle()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = request.model or MODEL_NAME
    yield (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        + "\n\n"
    )
    async for piece in _stream_openai_pieces(bundle, request):
        yield (
            "data: "
            + json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
            )
            + "\n\n"
        )
    yield (
        "data: "
        + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        + "\n\n"
    )
    yield "data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(UI_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return metadata_for_bundle(_get_bundle())


@app.get("/ui-config")
def ui_config() -> dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "adapter_path": ADAPTER,
        "compare_available": bool(ADAPTER),
        "base_label": MODEL_NAME,
        "adapter_label": "LoRA Adapter" if ADAPTER else None,
    }


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    bundle = _get_bundle()
    reply = _generate_for_bundle(bundle, request)
    return {"reply": reply}


@app.post("/chat/base")
def chat_base(request: ChatRequest) -> dict[str, object]:
    return {"reply": _generate_for_bundle(_get_base_bundle(), request)}


@app.post("/chat/adapter")
def chat_adapter(request: ChatRequest) -> dict[str, object]:
    _require_adapter()
    return {"reply": _generate_for_bundle(_get_adapter_bundle(), request)}


@app.post("/chat/base/stream")
def chat_base_stream(request: ChatRequest):
    return StreamingResponse(_stream_text_response(_get_base_bundle(), request), media_type="text/event-stream")


@app.post("/chat/adapter/stream")
def chat_adapter_stream(request: ChatRequest):
    _require_adapter()
    return StreamingResponse(_stream_text_response(_get_adapter_bundle(), request), media_type="text/event-stream")


@app.post("/v1/chat/completions")
def openai_chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(_stream_openai_response(request), media_type="text/event-stream")
    bundle = _get_bundle()
    reply = generate_reply(
        bundle.model,
        bundle.tokenizer,
        [message.model_dump() for message in request.messages],
        max_new_tokens=_resolve_max_new_tokens(request),
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return {
        "id": "chatcmpl-picollm",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
    }
