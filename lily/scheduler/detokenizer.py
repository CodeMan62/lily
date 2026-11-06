"""Incremental detokenizer frontend. Mirrors sglang/srt/managers/detokenizer_manager.py.

`DecodeStatus` keeps the ids decoded so far and the text already returned, so a
request can stream token by token. Two subtleties are handled:

* partial multibyte characters: a byte-level BPE decoder renders an incomplete
  UTF-8 sequence as U+FFFD, so a trailing replacement char is held back until
  the sequence completes (the full re-decode then yields the real character);
* stop strings: text that could be the beginning of a stop string is held back,
  and once a stop string appears the output is truncated at it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import zmq
from transformers import AutoTokenizer

from lily.scheduler.io_struct import BatchStrOut, BatchTokenIDOut
from lily.utils import longest_partial_suffix

logger = logging.getLogger(__name__)

_REPLACEMENT = "\ufffd"


class DecodeStatus:
    """Tracks ids decoded so far and the text already returned to the client.

    `decode` only updates `decoded_text` (the full re-decode of all ids, with a
    trailing replacement char held back); advancing `sent_text` is the caller's
    job so stop-string handling stays in one place.
    """

    def __init__(self) -> None:
        self.ids: List[int] = []
        self.decoded_text: str = ""
        self.sent_text: str = ""
        self.sent_offset: int = 0

    def decode(self, tokenizer, new_ids: List[int]) -> str:
        self.ids.extend(new_ids)
        full = tokenizer.decode(self.ids, skip_special_tokens=True)
        if not full.startswith(self.sent_text):
            common = 0
            while common < min(len(full), len(self.sent_text)) and full[common] == self.sent_text[common]:
                common += 1
            self.sent_text = self.sent_text[:common]
        # Hold back an incomplete trailing multibyte character.
        self.decoded_text = full.rstrip(_REPLACEMENT)
        self.sent_offset = len(self.sent_text)
        return self.decoded_text


class DetokenizerFrontend:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.status: Dict[str, DecodeStatus] = {}

    def handle_batch(self, msg: BatchTokenIDOut) -> BatchStrOut:
        output_strs: List[str] = []
        meta_info: List[Dict[str, Any]] = []

        for i, rid in enumerate(msg.rids):
            status = self.status.setdefault(rid, DecodeStatus())
            status.decode(self.tokenizer, msg.output_ids[i])

            stop_strs = msg.stop_strs[i] if i < len(msg.stop_strs) else []
            finish_reason = msg.finished_reasons[i]

            text = status.decoded_text
            old_sent = status.sent_text if text.startswith(status.sent_text) else ""

            stop_at = None
            stop_match = None
            for stop in stop_strs:
                index = text.find(stop)
                if index >= 0 and (stop_at is None or index < stop_at):
                    stop_at, stop_match = index, stop
            if stop_at is not None:
                text = text[:stop_at]
                emit_to = len(text)
                if finish_reason is None:
                    finish_reason = {"type": "stop", "value": stop_match}
            elif finish_reason is not None:
                emit_to = len(text)
            else:
                hold = max((longest_partial_suffix(text, s) for s in stop_strs), default=0)
                emit_to = len(text) - hold

            new_text = text[len(old_sent):emit_to]
            status.sent_text = text[:emit_to]
            output_strs.append(new_text)

            info: Dict[str, Any] = {
                "prompt_tokens": msg.prompt_tokens[i],
                "completion_tokens": msg.completion_tokens[i],
                "output_ids": msg.output_ids[i],      # this chunk's new tokens
            }
            if msg.logprobs is not None and msg.logprobs[i] is not None:
                info["output_token_logprobs"] = msg.logprobs[i]
            if finish_reason is not None:
                info["finish_reason"] = finish_reason
            meta_info.append(info)

            if finish_reason is not None:
                self.status.pop(rid, None)

        return BatchStrOut(msg.rids, output_strs, msg.finished_reasons, meta_info)


def run_detokenizer_process(tokenizer: Any, recv_endpoint: str, send_endpoint: str,
                            stop_event: Optional[threading.Event] = None) -> None:
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    context = zmq.Context()
    recv = context.socket(zmq.PULL)
    recv.bind(f"ipc://{recv_endpoint}")
    send = context.socket(zmq.PUSH)
    send.connect(f"ipc://{send_endpoint}")

    manager = DetokenizerFrontend(tokenizer)
    poller = zmq.Poller()
    poller.register(recv, zmq.POLLIN)
    logger.info("detokenizer ready")
    try:
        while stop_event is None or not stop_event.is_set():
            if dict(poller.poll(timeout=100)).get(recv) != zmq.POLLIN:
                continue
            while True:
                try:
                    msg = recv.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    break
                send.send_pyobj(manager.handle_batch(msg))
    finally:
        recv.close(linger=0)
        send.close(linger=0)
        context.term()
