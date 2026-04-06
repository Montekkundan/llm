from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from course_tools import available_checkpoints, default_artifact_dir, ensure_demo_checkpoint, format_messages, generate_text, stream_text


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=8, ge=1, le=64)
    stream: bool = False


app = FastAPI(title='LLM Course Chat App', version='0.1.0')
UI_PATH = Path(__file__).with_name('ui') / 'index.html'
BUNDLE = ensure_demo_checkpoint(steps=40)
MODEL = BUNDLE['model']
TOKENIZER = BUNDLE['tokenizer']


def fallback_response(messages: list[dict[str, str]]) -> str:
    last = messages[-1]['content'].lower()
    if 'self-attention' in last:
        return 'Self-attention lets each token score other tokens and mix information from the most relevant ones.'
    if 'tokenization' in last:
        return 'Tokenization turns raw text into discrete IDs before the model reads it.'
    if 'embedding' in last:
        return 'The embedding layer maps token IDs to learned vectors.'
    return f"Demo reply: {messages[-1]['content'][:140]}"


def run_chat(messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_k: int | None) -> str:
    prompt = format_messages(messages, add_assistant_prompt=True)
    text = generate_text(MODEL, TOKENIZER, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k).strip()
    return text or fallback_response(messages)


@app.get('/', response_class=HTMLResponse)
def index():
    return FileResponse(UI_PATH)


@app.get('/health')
def health():
    return {'status': 'ok', 'checkpoint': BUNDLE['path']}


@app.get('/stats')
def stats():
    return {
        'parameters': sum(p.numel() for p in MODEL.parameters()),
        'vocab_size': len(TOKENIZER.stoi),
        'artifact_dir': str(default_artifact_dir()),
    }


@app.get('/metadata')
def metadata():
    return {
        'config': BUNDLE['config'].__dict__,
        'metadata': BUNDLE['metadata'],
        'history': BUNDLE['history'][-5:],
    }


@app.get('/checkpoints')
def checkpoints():
    return {'items': available_checkpoints()}


@app.get('/v1/models')
def models():
    return {'data': [{'id': 'tiny-transformer-demo', 'object': 'model'}]}


@app.post('/chat')
def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    return {
        'reply': run_chat(messages, request.max_new_tokens, request.temperature, request.top_k),
        'usage': {'prompt_messages': len(messages)},
    }


async def event_stream(messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_k: int | None) -> AsyncIterator[str]:
    prompt = format_messages(messages, add_assistant_prompt=True)
    streamed_any = False
    for piece in stream_text(MODEL, TOKENIZER, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k):
        streamed_any = True
        yield f"data: {json.dumps({'delta': piece})}\n\n"
    if not streamed_any:
        fallback = fallback_response(messages)
        for ch in fallback:
            yield f"data: {json.dumps({'delta': ch})}\n\n"
    yield 'data: [DONE]\n\n'


@app.post('/chat/completions')
def chat_completions(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    if request.stream:
        return StreamingResponse(event_stream(messages, request.max_new_tokens, request.temperature, request.top_k), media_type='text/event-stream')
    return {'completion': run_chat(messages, request.max_new_tokens, request.temperature, request.top_k)}


@app.post('/v1/chat/completions')
def openai_chat_completions(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    if request.stream:
        return StreamingResponse(event_stream(messages, request.max_new_tokens, request.temperature, request.top_k), media_type='text/event-stream')
    content = run_chat(messages, request.max_new_tokens, request.temperature, request.top_k)
    return {
        'id': 'chatcmpl-demo',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            }
        ],
    }
