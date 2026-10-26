"""Launcher: starts the three processes and serves HTTP.

    tokenizer/HTTP process ──▶ scheduler process ──▶ detokenizer process ──▶ HTTP

With `--tp-size N` there are N scheduler processes (one per GPU); only rank 0
is connected to the tokenizer and detokenizer, the rest follow its broadcasts.
All inter-process links are ZMQ ipc sockets. Mirrors sglang/srt/entrypoints/engine.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import uvicorn
from transformers import AutoTokenizer

from lily.models.model_config import ModelConfig
from lily.entrypoints.http_server import create_app
from lily.scheduler.detokenizer import run_detokenizer_process
from lily.scheduler.scheduler import run_scheduler_process
from lily.scheduler.tokenizer import TokenizerFrontend
from lily.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class Endpoints:
    req: str  # tokenizer -> scheduler
    token: str  # scheduler -> detokenizer
    text: str  # detokenizer -> tokenizer


def make_endpoints(prefix: Optional[str] = None) -> Endpoints:
    base = prefix or f"/tmp/lily_{uuid.uuid4().hex[:10]}"
    return Endpoints(req=f"{base}_req", token=f"{base}_token", text=f"{base}_text")


def launch_scheduler_and_detokenizer(model_config, server_args, endpoints: Endpoints):
    """Start the scheduler (one per TP rank) and detokenizer processes; returns them for supervision/shutdown."""
    ctx = mp.get_context("spawn")
    schedulers = [
        ctx.Process(
            target=run_scheduler_process,
            args=(model_config, server_args, endpoints.req, endpoints.token, None, tp_rank),
            daemon=True,
            name=f"lily-scheduler-tp{tp_rank}",
        )
        for tp_rank in range(server_args.tp_size)
    ]
    detokenizer = ctx.Process(
        target=run_detokenizer_process,
        args=(server_args.tokenizer_path, endpoints.token, endpoints.text),
        daemon=True,
        name="lily-detokenizer",
    )
    for p in schedulers:
        p.start()
    detokenizer.start()
    return (*schedulers, detokenizer)


def serve(model_path: str, server_args=None, host: Optional[str] = None,
          port: Optional[int] = None) -> None:
    """Load config, start the workers, and serve the [OI] API (blocking)."""
    server_args = server_args or ServerArgs(model_path=model_path)
    if server_args.model_path == "":
        server_args.model_path = model_path
    if server_args.tokenizer_path is None:
        server_args.tokenizer_path = model_path

    model_config = ModelConfig.from_pretrained(
        model_path, dtype=server_args.dtype, context_length=server_args.context_length,
        revision=server_args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(server_args.tokenizer_path, trust_remote_code=server_args.trust_remote_code)

    endpoints = make_endpoints()
    processes = launch_scheduler_and_detokenizer(model_config, server_args, endpoints)

    frontend = TokenizerFrontend(tokenizer, server_args, endpoints.req, endpoints.text)
    app = create_app(frontend, server_args)

    try:
        uvicorn.run(app, host=host or server_args.host, port=port or server_args.port, log_level=server_args.log_level)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        frontend.stop()


def main(argv=None) -> None:
    """CLI entry point: `lily-serve --model-path ...` (or `python -m lily.entrypoints.launch`)."""
    parser = argparse.ArgumentParser("lily-serve")
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(args)
    if not server_args.model_path:
        parser.error("--model-path is required")
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve(server_args.model_path, server_args)


if __name__ == "__main__":
    main()
