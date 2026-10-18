"""Tokenizer frontend: the serving-side front door. Mirrors sglang/srt/managers/tokenizer_manager.py.

Lives in the HTTP process. Tokenizes (including the chat template), assigns a
rid, sends `TokenizedGenerateReqInput` to the scheduler over ZMQ, and turns the
detokenizer's `BatchStrOut` stream back into per-request async iterators,
splitting `<think>` reasoning content on the way out.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Dict, List, Optional

import zmq

from lula.scheduler.io_struct import (
    AbortReq,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from lula.parsers.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


class TokenizerFrontend:
    def __init__(self, tokenizer: Any, server_args: Any, req_endpoint: str, str_endpoint: str):
        self.tokenizer = tokenizer
        self.server_args = server_args
        self.model_name = server_args.served_model_name or server_args.model_path

        context = zmq.Context.instance()
        self.to_scheduler = context.socket(zmq.PUSH)
        self.to_scheduler.connect(f"ipc://{req_endpoint}")
        self.from_detokenizer = context.socket(zmq.PULL)
        self.from_detokenizer.bind(f"ipc://{str_endpoint}")

        self.rid_to_queue: Dict[str, "asyncio.Queue[Dict[str, Any]]"] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()

    # ---- lifecycle ---------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _next_rid(self) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"{self.server_args.served_model_name or 'lula'}-{self._counter}"

    # ---- receive path ------------------------------------------------------

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self.from_detokenizer, zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(timeout=100)).get(self.from_detokenizer) != zmq.POLLIN:
                continue
            while True:
                try:
                    msg = self.from_detokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    break
                for i, rid in enumerate(msg.rids):
                    queue = self.rid_to_queue.get(rid)
                    if queue is None or self._loop is None:
                        continue
                    item = {"text": msg.output_strs[i], "meta_info": dict(msg.meta_info[i])}
                    self._loop.call_soon_threadsafe(queue.put_nowait, item)

    # ---- request path ------------------------------------------------------

    def tokenize(self, obj: GenerateReqInput) -> List[int]:
        if obj.input_ids is not None:
            return list(obj.input_ids)
        if obj.is_chat:
            if obj.messages is None:
                raise ValueError("chat request requires messages")
            kwargs = dict(obj.chat_template_kwargs or {})
            ids = self.tokenizer.apply_chat_template(
                obj.messages, add_generation_prompt=True, tokenize=True, **kwargs
            )
            # transformers may return a plain list or a BatchEncoding.
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            return [int(i) for i in ids]
        if obj.text is None:
            raise ValueError("request requires text or input_ids")
        return list(self.tokenizer.encode(obj.text))

    def _reasoning_parser(self, obj: GenerateReqInput) -> ReasoningParser:
        thinking = True
        if obj.chat_template_kwargs and "enable_thinking" in obj.chat_template_kwargs:
            thinking = bool(obj.chat_template_kwargs["enable_thinking"])
        if self.server_args.reasoning_parser is None:
            thinking = False
        return ReasoningParser(thinking=thinking)

    def _send(self, rid: str, obj: GenerateReqInput) -> None:
        input_ids = self.tokenize(obj)
        self.to_scheduler.send_pyobj(
            TokenizedGenerateReqInput(rid, input_ids, obj.to_sampling_params(), obj.stream)
        )

    async def generate_request(self, obj: GenerateReqInput) -> AsyncIterator[Dict[str, Any]]:
        """Yield `{"text", "reasoning_content", "meta_info"}` chunks for one request."""
        rid = obj.rid or self._next_rid()
        queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.rid_to_queue[rid] = queue
        try:
            self._send(rid, obj)
        except Exception:
            self.rid_to_queue.pop(rid, None)
            raise

        parser = self._reasoning_parser(obj)
        try:
            if obj.stream:
                async for chunk in self._stream(rid, queue, parser):
                    yield chunk
            else:
                yield await self._collect(rid, queue, parser)
        finally:
            self.rid_to_queue.pop(rid, None)

    async def _stream(self, rid, queue, parser) -> AsyncIterator[Dict[str, Any]]:
        while True:
            item = await queue.get()
            reasoning, content = parser.parse_stream_chunk(item["text"])
            finished = "finish_reason" in item["meta_info"]
            if finished:
                r, c = parser.flush()
                reasoning += r
                content += c
            if reasoning or content or finished:
                yield {"text": content, "reasoning_content": reasoning, "meta_info": item["meta_info"]}
            if finished:
                return

    async def _collect(self, rid, queue, parser) -> Dict[str, Any]:
        reasoning = ""
        content = ""
        meta_info: Dict[str, Any] = {}
        while True:
            item = await queue.get()
            r, c = parser.parse_stream_chunk(item["text"])
            reasoning += r
            content += c
            meta_info = item["meta_info"]
            if "finish_reason" in meta_info:
                r, c = parser.flush()
                reasoning += r
                content += c
                break
        return {"text": content, "reasoning_content": reasoning, "meta_info": meta_info}

    # ---- control -----------------------------------------------------------

    def abort(self, rid: str) -> None:
        self.to_scheduler.send_pyobj(AbortReq(rid))

    def flush_cache(self) -> None:
        self.to_scheduler.send_pyobj(FlushCacheReq())
