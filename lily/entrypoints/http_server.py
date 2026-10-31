"""[OI]-compatible HTTP server. Mirrors sglang/srt/entrypoints/http_server.py + openai/.

Routes: /generate, /v1/completions, /v1/chat/completions (SSE), /health,
/get_server_info, /flush_cache. Streaming requests abort the scheduler request
when the client disconnects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from lily.scheduler.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


# ---- request models --------------------------------------------------------

class _SamplingRequest(BaseModel):
    """Fields shared by the OpenAI completion endpoints."""

    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    stream: bool = False
    n: int = 1
    seed: Optional[int] = None


class CompletionRequest(_SamplingRequest):
    prompt: Union[str, List[str]] = ""
    logprobs: Optional[int] = None
    echo: bool = False


class ChatCompletionRequest(_SamplingRequest):
    messages: List[Dict[str, Any]]
    max_completion_tokens: Optional[int] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    stream: bool = False
    rid: Optional[str] = None


# ---- helpers ---------------------------------------------------------------

def _finish_reason(reason: Optional[dict]) -> Optional[str]:
    if reason is None:
        return None
    return "length" if reason.get("type") == "length" else "stop"


def _sampling_params(body: Any) -> Dict[str, Any]:
    max_new = getattr(body, "max_tokens", None) or getattr(body, "max_completion_tokens", None)
    params: Dict[str, Any] = {
        "max_new_tokens": max_new or 128,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "min_p": body.min_p,
        "repetition_penalty": body.repetition_penalty,
        "frequency_penalty": body.frequency_penalty,
        "presence_penalty": body.presence_penalty,
        "stop": body.stop,
        "stop_token_ids": body.stop_token_ids or [],
        "ignore_eos": body.ignore_eos,
        "n": body.n,
        "seed": body.seed,
    }
    if getattr(body, "logprobs", None):
        params["logprobs"] = True
        if getattr(body, "top_logprobs", None):
            params["top_logprobs"] = body.top_logprobs
    return params


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage(meta: Dict[str, Any]) -> Dict[str, int]:
    prompt = meta.get("prompt_tokens", 0)
    completion = meta.get("completion_tokens", 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def _text_choice(text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": text, "index": 0, "logprobs": None,
        "finish_reason": _finish_reason(meta.get("finish_reason")),
    }


# ---- app -------------------------------------------------------------------

def create_app(frontend, server_args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        frontend.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            frontend.stop()

    app = FastAPI(title="lily", lifespan=lifespan)
    model_name = server_args.served_model_name or server_args.model_path

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/get_server_info")
    async def get_server_info():
        return {
            "model_path": server_args.model_path,
            "served_model_name": model_name,
            "context_length": server_args.context_length,
            "attention_backend": server_args.attention_backend,
            "schedule_policy": server_args.schedule_policy,
            "reasoning_parser": server_args.reasoning_parser,
            "speculative_algorithm": server_args.speculative_algorithm,
        }

    @app.post("/flush_cache")
    async def flush_cache():
        frontend.flush_cache()
        return {"status": "ok"}

    # ---- native /generate --------------------------------------------------

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request):
        obj = GenerateReqInput(
            text=body.text, input_ids=body.input_ids,
            sampling_params=body.sampling_params, stream=body.stream, rid=body.rid,
        )
        if not body.stream:
            async for chunk in frontend.generate_request(obj):
                return JSONResponse({
                    "text": chunk["text"],
                    "reasoning_content": chunk["reasoning_content"] or None,
                    "meta_info": chunk["meta_info"],
                })

        async def stream():
            async for chunk in frontend.generate_request(obj):
                if await request.is_disconnected():
                    frontend.abort(obj.rid)
                    return
                yield _sse(chunk)
        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---- /v1/completions ---------------------------------------------------

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        if isinstance(body.prompt, list):
            return JSONResponse({"error": "batch prompts are not supported"}, status_code=400)
        rid = f"cmpl-{uuid.uuid4().hex}"
        obj = GenerateReqInput(
            text=body.prompt, sampling_params=_sampling_params(body), stream=body.stream, rid=rid,
        )
        created = int(time.time())

        if not body.stream:
            async for chunk in frontend.generate_request(obj):
                meta = chunk["meta_info"]
                return JSONResponse({
                    "id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [_text_choice(chunk["text"], meta)],
                    "usage": _usage(meta),
                })

        async def stream():
            async for chunk in frontend.generate_request(obj):
                if await request.is_disconnected():
                    frontend.abort(rid)
                    return
                meta = chunk["meta_info"]
                yield _sse({
                    "id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [_text_choice(chunk["text"], meta)],
                })
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---- /v1/chat/completions ---------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        obj = GenerateReqInput(
            messages=body.messages, is_chat=True, chat_template_kwargs=body.chat_template_kwargs,
            sampling_params=_sampling_params(body), stream=body.stream, rid=rid,
        )
        created = int(time.time())

        if not body.stream:
            async for chunk in frontend.generate_request(obj):
                meta = chunk["meta_info"]
                message: Dict[str, Any] = {"role": "assistant", "content": chunk["text"]}
                if chunk["reasoning_content"]:
                    message["reasoning_content"] = chunk["reasoning_content"]
                return JSONResponse({
                    "id": rid, "object": "chat.completion", "created": created, "model": model_name,
                    "choices": [{
                        "index": 0, "message": message,
                        "finish_reason": _finish_reason(meta.get("finish_reason")),
                    }],
                    "usage": _usage(meta),
                })

        async def stream():
            first = True
            async for chunk in frontend.generate_request(obj):
                if await request.is_disconnected():
                    frontend.abort(rid)
                    return
                meta = chunk["meta_info"]
                delta: Dict[str, Any] = {}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if chunk["reasoning_content"]:
                    delta["reasoning_content"] = chunk["reasoning_content"]
                if chunk["text"]:
                    delta["content"] = chunk["text"]
                yield _sse({
                    "id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{
                        "index": 0, "delta": delta,
                        "finish_reason": _finish_reason(meta.get("finish_reason")),
                    }],
                })
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
