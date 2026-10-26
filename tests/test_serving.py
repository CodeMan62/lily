"""Phase 6: serving layer.

Three tests:
  1. detokenizer (incremental, multibyte-safe, stop strings) and reasoning parser;
  2. [OI] HTTP client, non-streaming completions and chat;
  3. streaming equals non-streaming for both endpoints.

Tests 2 and 3 share one running server (scheduler + detokenizer + tokenizer
manager threads over ZMQ ipc sockets, served by uvicorn) and drive it with the
[OI] Python client.
"""

import threading
import time
import uuid

import openai
import pytest
import uvicorn
from transformers import AutoTokenizer

from lily.models.model_config import ModelConfig
from lily.entrypoints.http_server import create_app
from lily.entrypoints.launch import make_endpoints
from lily.scheduler.detokenizer import (
    DecodeStatus,
    DetokenizerFrontend,
    run_detokenizer_process,
)
from lily.scheduler.io_struct import BatchTokenIDOut
from lily.scheduler.scheduler import run_scheduler_process
from lily.scheduler.tokenizer import TokenizerFrontend
from lily.parsers.reasoning_parser import ReasoningParser
from lily.server_args import ServerArgs
from tests.models import SMALL_LLAMA
from tests.util import free_cuda

MODEL = SMALL_LLAMA


class _FakeTokenizer:
    def __init__(self, table):
        self.table = table

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table[i] for i in ids)


def test_detokenizer_and_reasoning_parser():
    # Incremental decode of a byte-level tokenizer never emits a replacement char.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer.encode("你好世界🙂")
    status = DecodeStatus()
    emitted = ""
    for token in ids:
        text = status.decode(tokenizer, [token])
        assert "\ufffd" not in text[len(emitted):]
        emitted = text
    assert emitted == "你好世界🙂"

    # Stop-string truncation, and holding back a partial stop string.
    manager = DetokenizerFrontend(_FakeTokenizer({0: "Hel", 1: "lo ST", 2: "OP", 3: "!"}))
    msg = BatchTokenIDOut(rids=["a"], output_ids=[[0, 1]], finished_reasons=[None],
                          prompt_tokens=[1], completion_tokens=[2], stop_strs=[["STOP"]])
    assert manager.handle_batch(msg).output_strs == ["Hello "]  # "ST" held back
    msg = BatchTokenIDOut(rids=["a"], output_ids=[[2, 3]], finished_reasons=[None],
                          prompt_tokens=[1], completion_tokens=[4], stop_strs=[["STOP"]])
    out = manager.handle_batch(msg)
    assert out.output_strs == [""]
    assert out.meta_info[0]["finish_reason"]["type"] == "stop"

    # Reasoning: non-streaming split and streaming across a split delimiter.
    parser = ReasoningParser()
    reasoning, content = parser.parse_non_stream("<think>\nLet me think.\n</think>\n\nAnswer: 4")
    assert reasoning == "\nLet me think.\n" and content == "\n\nAnswer: 4"

    parser = ReasoningParser(thinking=True)
    reasoning = content = ""
    for chunk in ["step 1\n", "step 2\n</th", "ink>final"]:
        r, c = parser.parse_stream_chunk(chunk)
        reasoning += r
        content += c
    r, c = parser.flush()
    reasoning += r
    content += c
    assert reasoning == "step 1\nstep 2\n" and content == "final"


@pytest.fixture(scope="module")
def client():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    cfg = ModelConfig.from_pretrained(MODEL, dtype="float32", context_length=1024)
    sa = ServerArgs(
        model_path=MODEL, dtype="float32", attention_backend="triton", context_length=1024,
        max_running_requests=8, max_total_tokens=4096, chunked_prefill_size=2048,
        served_model_name="lily-test",
    )

    endpoints = make_endpoints(prefix=f"/tmp/lily_test_{uuid.uuid4().hex[:8]}")
    stop = threading.Event()
    workers = [
        threading.Thread(target=run_scheduler_process,
                         args=(cfg, sa, endpoints.req, endpoints.token, stop), daemon=True),
        threading.Thread(target=run_detokenizer,
                         args=(tokenizer, endpoints.token, endpoints.text, stop), daemon=True),
    ]
    for w in workers:
        w.start()

    frontend = TokenizerFrontend(tokenizer, sa, endpoints.req, endpoints.text)
    app = create_app(frontend, sa)
    port = 38000 + (uuid.uuid4().int % 2000)
    uv_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    uv_thread = threading.Thread(target=uv_server.run, daemon=True)
    uv_thread.start()

    deadline = time.time() + 60
    while not uv_server.started and time.time() < deadline:
        time.sleep(0.05)
    assert uv_server.started, "uvicorn did not start"

    yield openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="none", timeout=60.0)

    uv_server.should_exit = True
    uv_thread.join(timeout=10)
    stop.set()
    for w in workers:
        w.join(timeout=5)
    free_cuda()


def run_detokenizer(tokenizer, recv, send, stop):
    run_detokenizer_process(tokenizer, recv, send, stop)


@pytest.mark.timeout(300)
def test_http_client_non_streaming(client):
    resp = client.completions.create(
        model="lily-test", prompt="The capital of France is", max_tokens=8, temperature=0.0,
    )
    assert isinstance(resp.choices[0].text, str) and resp.choices[0].text
    assert resp.usage.completion_tokens > 0

    chat = client.chat.completions.create(
        model="lily-test", messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8, temperature=0.0,
    )
    content = chat.choices[0].message.content
    assert isinstance(content, str) and content
    assert chat.choices[0].message.role == "assistant"


@pytest.mark.timeout(300)
def test_http_streaming_equals_non_streaming(client):
    prompt = "The quick brown fox"
    non_stream = client.completions.create(
        model="lily-test", prompt=prompt, max_tokens=16, temperature=0.0,
    ).choices[0].text
    streamed = "".join(
        chunk.choices[0].text or ""
        for chunk in client.completions.create(
            model="lily-test", prompt=prompt, max_tokens=16, temperature=0.0, stream=True,
        )
    )
    assert streamed == non_stream

    messages = [{"role": "user", "content": "Say hello."}]
    non_stream = client.chat.completions.create(
        model="lily-test", messages=messages, max_tokens=16, temperature=0.0,
    ).choices[0].message.content
    streamed = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in client.chat.completions.create(
            model="lily-test", messages=messages, max_tokens=16, temperature=0.0, stream=True,
        )
    )
    assert streamed == non_stream
