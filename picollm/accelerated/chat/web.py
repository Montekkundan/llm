#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from picollm.accelerated.checkpoint_manager import load_model
from picollm.accelerated.common import autodetect_device_type, compute_init, get_assets_dir
from picollm.accelerated.engine import Engine

MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description="picoLLM web server")
parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("-i", "--source", type=str, default="sft", help="Model source: base|sft|rl")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
parser.add_argument("-p", "--port", type=int, default=8008, help="Port to run the server on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
parser.add_argument("--model-id", type=str, default="picollm-chat", help="Stable model id exposed by the OpenAI-compatible API")
parser.add_argument("--temperature", type=float, default=0.8, help="Default sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Default top-k sampling parameter")
parser.add_argument("--max-tokens", type=int, default=512, help="Default max generation length")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type: cuda|cpu|mps")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object


class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        logger.info("Initializing picoLLM worker pool with %s device(s)", self.num_gpus)
        if self.num_gpus > 1:
            assert device_type == "cuda", "Multiple workers require CUDA."

        for gpu_id in range(self.num_gpus):
            worker_device = torch.device(f"cuda:{gpu_id}") if device_type == "cuda" else torch.device(device_type)
            logger.info("Loading %s model on %s", source, worker_device)
            model, tokenizer, _ = load_model(source, worker_device, phase="eval", model_tag=model_tag, step=step)
            worker = Worker(gpu_id=gpu_id, device=worker_device, engine=Engine(model, tokenizer), tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        logger.info("Loaded %s picoLLM worker(s)", len(self.workers))

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


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


def validate_chat_request(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed.")

    total_length = 0
    for idx, message in enumerate(request.messages):
        if message.role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Message {idx} has invalid role '{message.role}'.")
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {idx} has empty content.")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {idx} exceeds {MAX_MESSAGE_LENGTH} characters.")
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Conversation exceeds {MAX_TOTAL_CONVERSATION_LENGTH} characters.")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}.")

    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}.")


def resolve_max_tokens(request: ChatRequest) -> int:
    return request.max_completion_tokens or request.max_tokens or args.max_tokens


def request_temperature(request: ChatRequest) -> float:
    return request.temperature if request.temperature is not None else args.temperature


def request_top_k(request: ChatRequest) -> int:
    return request.top_k if request.top_k is not None else args.top_k


def build_conversation_tokens(worker: Worker, messages: List[ChatMessage]) -> list[int]:
    bos = worker.tokenizer.get_bos_token_id()
    user_start = worker.tokenizer.encode_special("<|user_start|>")
    user_end = worker.tokenizer.encode_special("<|user_end|>")
    assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

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
            tokens.extend(worker.tokenizer.encode(content))
            tokens.append(user_end)
        elif role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(worker.tokenizer.encode(content))
            tokens.append(assistant_end)

    tokens.append(assistant_start)
    return tokens


async def generate_text_stream(worker: Worker, request: ChatRequest) -> AsyncGenerator[str, None]:
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    prompt_tokens = build_conversation_tokens(worker, request.messages)

    accumulated_tokens: list[int] = []
    last_clean_text = ""
    for token_column, token_masks in worker.engine.generate(
        prompt_tokens,
        num_samples=1,
        max_tokens=resolve_max_tokens(request),
        temperature=request_temperature(request),
        top_k=request_top_k(request),
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token in {assistant_end, bos}:
            break
        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if current_text.endswith("�"):
            continue
        new_text = current_text[len(last_clean_text):]
        if new_text:
            last_clean_text = current_text
            yield new_text


async def collect_response(worker: Worker, request: ChatRequest) -> str:
    chunks: list[str] = []
    async for piece in generate_text_stream(worker, request):
        chunks.append(piece)
    return "".join(chunks)


def openai_chunk_payload(completion_id: str, model_id: str, created: int, delta: dict[str, str], finish_reason: Optional[str]):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading picoLLM models across devices")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    logger.info("Server ready at http://%s:%s", args.host, args.port)
    yield


app = FastAPI(title="picoLLM API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    ui_html_path = get_assets_dir() / "ui.html"
    html_content = ui_html_path.read_text(encoding="utf-8")
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';",
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    return FileResponse(Path(get_assets_dir() / "logo.svg"), media_type="image/svg+xml")


@app.get("/health")
async def health():
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "model_id": args.model_id,
        "source": args.source,
        "num_workers": len(worker_pool.workers) if worker_pool else 0,
    }


@app.get("/stats")
async def stats():
    worker_pool = app.state.worker_pool
    return {
        "model_id": args.model_id,
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [{"gpu_id": w.gpu_id, "device": str(w.device)} for w in worker_pool.workers],
    }


@app.get("/v1/models")
async def models():
    return {"data": [{"id": args.model_id, "object": "model"}]}


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()
    response_chunks: list[str] = []

    async def stream_and_release():
        try:
            async for piece in generate_text_stream(worker, request):
                response_chunks.append(piece)
                yield f"data: {json.dumps({'token': piece, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        finally:
            logger.info("[gpu %s] %s", worker.gpu_id, "".join(response_chunks))
            await worker_pool.release_worker(worker)

    return StreamingResponse(stream_and_release(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
    validate_chat_request(request)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_id = request.model or args.model_id

    if request.stream:
        async def openai_stream_and_release():
            pieces: list[str] = []
            try:
                yield "data: " + json.dumps(
                    openai_chunk_payload(completion_id, model_id, created, {"role": "assistant"}, None),
                    ensure_ascii=False,
                ) + "\n\n"
                async for piece in generate_text_stream(worker, request):
                    pieces.append(piece)
                    yield "data: " + json.dumps(
                        openai_chunk_payload(completion_id, model_id, created, {"content": piece}, None),
                        ensure_ascii=False,
                    ) + "\n\n"
                yield "data: " + json.dumps(
                    openai_chunk_payload(completion_id, model_id, created, {}, "stop"),
                    ensure_ascii=False,
                ) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                logger.info("[gpu %s] %s", worker.gpu_id, "".join(pieces))
                await worker_pool.release_worker(worker)

        return StreamingResponse(openai_stream_and_release(), media_type="text/event-stream")

    try:
        reply = await collect_response(worker, request)
        logger.info("[gpu %s] %s", worker.gpu_id, reply)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        }
    finally:
        await worker_pool.release_worker(worker)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting picoLLM web server")
    logger.info("Temperature=%s top_k=%s max_tokens=%s", args.temperature, args.top_k, args.max_tokens)
    uvicorn.run(app, host=args.host, port=args.port)
