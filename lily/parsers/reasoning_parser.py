"""Reasoning-content split. Mirrors sglang/srt/parsers/reasoning_parser.py (qwen3 style).

Qwen3 thinking mode wraps its chain of thought in `<think>...</think>`. When
thinking is enabled the `<think>` opener is usually part of the prompt, so the
generated text only carries the closing `</think>`. Both cases are handled.

Streaming is stateful: delimiters may be split across chunks, so a partial
delimiter is held back until it is complete.
"""

from __future__ import annotations

from typing import Tuple

from lily.utils import longest_partial_suffix

THINK_START = "<think>"
THINK_END = "</think>"


class ReasoningParser:
    def __init__(self, think_start: str = THINK_START, think_end: str = THINK_END,
                 thinking: bool = True):
        self.think_start = think_start
        self.think_end = think_end
        self.thinking = thinking
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._in_reasoning = self.thinking
        self._at_start = True

    # ---- non-streaming -----------------------------------------------------

    def parse_non_stream(self, text: str) -> Tuple[str, str]:
        if self.think_start in text:
            rest = text.split(self.think_start, 1)[1]
            if self.think_end in rest:
                reasoning, content = rest.split(self.think_end, 1)
            else:
                reasoning, content = rest, ""
        elif self.think_end in text:
            reasoning, content = text.split(self.think_end, 1)
        else:
            reasoning, content = "", text
        return reasoning, content

    # ---- streaming ---------------------------------------------------------

    def parse_stream_chunk(self, chunk: str) -> Tuple[str, str]:
        self._buffer += chunk
        reasoning_out = ""
        content_out = ""
        while True:
            if self._at_start:
                # Consume a leading `<think>` (it may arrive split across chunks).
                if self._buffer.startswith(self.think_start):
                    self._buffer = self._buffer[len(self.think_start):]
                    self._in_reasoning = True
                    self._at_start = False
                    continue
                if self._buffer and self.think_start.startswith(self._buffer):
                    break  # partial opener: wait for more
                self._at_start = False
                continue
            if self._in_reasoning:
                index = self._buffer.find(self.think_end)
                if index >= 0:
                    reasoning_out += self._buffer[:index]
                    self._buffer = self._buffer[index + len(self.think_end):]
                    self._in_reasoning = False
                    continue
                hold = longest_partial_suffix(self._buffer, self.think_end)
                if hold:
                    reasoning_out += self._buffer[: len(self._buffer) - hold]
                    self._buffer = self._buffer[len(self._buffer) - hold:]
                else:
                    reasoning_out += self._buffer
                    self._buffer = ""
                break
            else:
                index = self._buffer.find(self.think_start)
                if index >= 0:
                    content_out += self._buffer[:index]
                    self._buffer = self._buffer[index + len(self.think_start):]
                    self._in_reasoning = True
                    continue
                hold = longest_partial_suffix(self._buffer, self.think_start)
                if hold:
                    content_out += self._buffer[: len(self._buffer) - hold]
                    self._buffer = self._buffer[len(self._buffer) - hold:]
                else:
                    content_out += self._buffer
                    self._buffer = ""
                break
        return reasoning_out, content_out

    def flush(self) -> Tuple[str, str]:
        """Emit whatever is still buffered (call once the request finishes)."""
        pending, self._buffer = self._buffer, ""
        if self._in_reasoning:
            return pending, ""
        return "", pending

