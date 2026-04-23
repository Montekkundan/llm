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
from typing import AsyncGenerator, Callable, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from picollm.accelerated.chat.api import (
    ChatRequest,
    ChatValidationError,
    GenerationDefaults,
    GenerationSettings,
    validate_generation_budget,
)
from picollm.accelerated.checkpoint_manager import load_model
from picollm.accelerated.common import autodetect_device_type, compute_init, get_assets_dir
from picollm.accelerated.engine import Engine


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="picoLLM web server")
    parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("-i", "--source", type=str, default="sft", choices=["base", "sft"], help="Model source: base|sft")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("-p", "--port", type=int, default=8008, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--model-id", type=str, default="picollm-chat", help="Stable model id exposed by the OpenAI-compatible API")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Default top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=None, help="Default top-p sampling parameter")
    parser.add_argument("--min-p", type=float, default=None, help="Default min-p cutoff relative to the most likely token")
    parser.add_argument("--max-tokens", type=int, default=512, help="Default max generation length")
    parser.add_argument("--seed", type=int, default=None, help="Default seed for generation. empty => random per request")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type: cuda|cpu|mps")
    return parser


def generation_defaults_from_args(args) -> GenerationDefaults:
    return GenerationDefaults(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def random_seed() -> int:
    return random.randint(0, 2**31 - 1)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object


class WorkerPool:
    def __init__(
        self,
        args,
        device_type: str,
        *,
        load_model_fn=load_model,
        engine_cls=Engine,
    ):
        if args.num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        else:
            num_gpus = args.num_gpus
        self.args = args
        self.device_type = device_type
        self.num_gpus = num_gpus
        self.load_model_fn = load_model_fn
        self.engine_cls = engine_cls
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()
        self.model_info: dict[str, object] = {}

    async def initialize(self) -> None:
        logger.info("Initializing picoLLM worker pool with %s device(s)", self.num_gpus)
        if self.num_gpus > 1:
            assert self.device_type == "cuda", "Multiple workers require CUDA."

        for gpu_id in range(self.num_gpus):
            worker_device = torch.device(f"cuda:{gpu_id}") if self.device_type == "cuda" else torch.device(self.device_type)
            logger.info("Loading %s model on %s", self.args.source, worker_device)
            model, tokenizer, meta = self.load_model_fn(
                self.args.source,
                worker_device,
                phase="eval",
                model_tag=self.args.model_tag,
                step=self.args.step,
            )
            worker = Worker(
                gpu_id=gpu_id,
                device=worker_device,
                engine=self.engine_cls(model, tokenizer),
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)
            checkpoint = meta.get("_checkpoint", {})
            if not self.model_info:
                self.model_info = {
                    "source": self.args.source,
                    "model_tag": checkpoint.get("model_tag"),
                    "step": checkpoint.get("step"),
                    "device": str(worker_device),
                    "device_type": self.device_type,
                }

        logger.info("Loaded %s picoLLM worker(s)", len(self.workers))

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker) -> None:
        await self.available_workers.put(worker)


def openai_chunk_payload(completion_id: str, model_id: str, created: int, delta: dict[str, str], finish_reason: Optional[str]):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def create_app(
    args,
    *,
    load_model_fn=load_model,
    engine_cls=Engine,
    compute_init_fn=compute_init,
    autodetect_device_type_fn=autodetect_device_type,
) -> FastAPI:
    device_type = autodetect_device_type_fn() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init_fn(device_type)
    defaults = generation_defaults_from_args(args)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading picoLLM models across devices")
        app.state.worker_pool = WorkerPool(
            args,
            device_type,
            load_model_fn=load_model_fn,
            engine_cls=engine_cls,
        )
        await app.state.worker_pool.initialize()
        worker_pool = app.state.worker_pool
        logger.info(
            "Loaded checkpoint: source=%s tag=%s step=%s device=%s",
            worker_pool.model_info.get("source"),
            worker_pool.model_info.get("model_tag"),
            worker_pool.model_info.get("step"),
            worker_pool.model_info.get("device"),
        )
        logger.info("Open picoLLM web chat at http://127.0.0.1:%s (bound host=%s)", args.port, args.host)
        yield

    app = FastAPI(title="picoLLM API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def prepare_generation(worker: Worker, request: ChatRequest) -> tuple[list[int], GenerationSettings]:
        try:
            return validate_generation_budget(
                worker.tokenizer,
                worker.engine.model.config.sequence_len,
                request,
                defaults,
                random_seed_fn=random_seed,
            )
        except ChatValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def generate_text_stream(
        worker: Worker,
        prompt_tokens: list[int],
        settings: GenerationSettings,
    ) -> AsyncGenerator[str, None]:
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
        bos = worker.tokenizer.get_bos_token_id()
        accumulated_tokens: list[int] = []
        last_clean_text = ""
        for token_column, token_masks in worker.engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            min_p=settings.min_p,
            seed=settings.seed,
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

    async def collect_response(worker: Worker, prompt_tokens: list[int], settings: GenerationSettings) -> str:
        chunks: list[str] = []
        async for piece in generate_text_stream(worker, prompt_tokens, settings):
            chunks.append(piece)
        return "".join(chunks)

    async def is_client_disconnected(http_request: Request) -> bool:
        try:
            return await http_request.is_disconnected()
        except RuntimeError:
            return True

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
            "model_tag": worker_pool.model_info.get("model_tag") if worker_pool else args.model_tag,
            "step": worker_pool.model_info.get("step") if worker_pool else args.step,
            "device": worker_pool.model_info.get("device") if worker_pool else str(device),
            "device_type": worker_pool.model_info.get("device_type") if worker_pool else device_type,
            "default_generation": {
                "temperature": defaults.temperature,
                "top_k": defaults.top_k,
                "top_p": defaults.top_p,
                "min_p": defaults.min_p,
                "max_tokens": defaults.max_tokens,
                "seed": defaults.seed,
            },
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
    async def chat_completions(request: ChatRequest, http_request: Request):
        worker_pool = app.state.worker_pool
        worker = await worker_pool.acquire_worker()
        response_chunks: list[str] = []
        try:
            prompt_tokens, settings = prepare_generation(worker, request)
        except Exception:
            await worker_pool.release_worker(worker)
            raise

        async def stream_and_release():
            try:
                if await is_client_disconnected(http_request):
                    return
                async for piece in generate_text_stream(worker, prompt_tokens, settings):
                    if await is_client_disconnected(http_request):
                        logger.info("[gpu %s] client disconnected during stream", worker.gpu_id)
                        break
                    response_chunks.append(piece)
                    yield f"data: {json.dumps({'token': piece, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                if not await is_client_disconnected(http_request):
                    yield f"data: {json.dumps({'done': True})}\n\n"
            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                logger.info("[gpu %s] stream cancelled by client", worker.gpu_id)
            finally:
                logger.info("[gpu %s] %s", worker.gpu_id, "".join(response_chunks))
                await worker_pool.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: ChatRequest, http_request: Request):
        worker_pool = app.state.worker_pool
        worker = await worker_pool.acquire_worker()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_id = request.model or args.model_id
        try:
            prompt_tokens, settings = prepare_generation(worker, request)
        except Exception:
            await worker_pool.release_worker(worker)
            raise

        if request.stream:
            async def openai_stream_and_release():
                pieces: list[str] = []
                try:
                    if await is_client_disconnected(http_request):
                        return
                    yield "data: " + json.dumps(
                        openai_chunk_payload(completion_id, model_id, created, {"role": "assistant"}, None),
                        ensure_ascii=False,
                    ) + "\n\n"
                    async for piece in generate_text_stream(worker, prompt_tokens, settings):
                        if await is_client_disconnected(http_request):
                            logger.info("[gpu %s] client disconnected during OpenAI stream", worker.gpu_id)
                            break
                        pieces.append(piece)
                        yield "data: " + json.dumps(
                            openai_chunk_payload(completion_id, model_id, created, {"content": piece}, None),
                            ensure_ascii=False,
                        ) + "\n\n"
                    if not await is_client_disconnected(http_request):
                        yield "data: " + json.dumps(
                            openai_chunk_payload(completion_id, model_id, created, {}, "stop"),
                            ensure_ascii=False,
                        ) + "\n\n"
                        yield "data: [DONE]\n\n"
                except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                    logger.info("[gpu %s] OpenAI stream cancelled by client", worker.gpu_id)
                finally:
                    logger.info("[gpu %s] %s", worker.gpu_id, "".join(pieces))
                    await worker_pool.release_worker(worker)

            return StreamingResponse(openai_stream_and_release(), media_type="text/event-stream")

        try:
            reply = await collect_response(worker, prompt_tokens, settings)
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

    return app


def main(argv=None, *, create_app_fn=create_app, uvicorn_run=None) -> int:
    args = build_parser().parse_args(argv)
    logger.info("Starting picoLLM web server")
    logger.info(
        "Default generation: temperature=%s top_k=%s top_p=%s min_p=%s max_tokens=%s seed=%s",
        args.temperature,
        args.top_k,
        args.top_p,
        args.min_p,
        args.max_tokens,
        args.seed,
    )
    app = create_app_fn(args)
    if uvicorn_run is None:
        import uvicorn

        uvicorn_run = uvicorn.run
    uvicorn_run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
