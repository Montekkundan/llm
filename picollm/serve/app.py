from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from picollm.common import generate_reply, load_generation_bundle
from picollm.common.loading import metadata_for_bundle


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


MODEL_NAME = os.environ.get("PICOLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("PICOLLM_ADAPTER")
DEVICE = os.environ.get("PICOLLM_DEVICE", "auto")
QUANTIZATION = os.environ.get("PICOLLM_QUANTIZATION", "none")
UI_PATH = Path(__file__).with_name("ui") / "index.html"

app = FastAPI(title="picoLLM Chat App", version="0.1.0")


@lru_cache(maxsize=1)
def _get_bundle():
    return load_generation_bundle(MODEL_NAME, adapter_path=ADAPTER, device=DEVICE, quantization=QUANTIZATION)


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(UI_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return metadata_for_bundle(_get_bundle())


@app.get("/v1/models")
def models() -> dict[str, list[dict[str, str]]]:
    return {"data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    bundle = _get_bundle()
    reply = generate_reply(
        bundle.model,
        bundle.tokenizer,
        [message.model_dump() for message in request.messages],
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    return {"reply": reply}


async def _stream_response(reply: str):
    for chunk in reply.split():
        yield f"data: {json.dumps({'delta': chunk + ' '})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def openai_chat(request: ChatRequest):
    bundle = _get_bundle()
    reply = generate_reply(
        bundle.model,
        bundle.tokenizer,
        [message.model_dump() for message in request.messages],
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    if request.stream:
        return StreamingResponse(_stream_response(reply), media_type="text/event-stream")
    return {
        "id": "chatcmpl-picollm",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
    }
