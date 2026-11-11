"""Tool-call extraction. Mirrors sglang/srt/function_call/ (hermes, qwen3_coder, glm4_moe detectors).

All three formats wrap each call in `<tool_call>...</tool_call>`; only the body differs:
  hermes      (Qwen3):        {"name": "f", "arguments": {"x": 1}}
  qwen3_coder (Qwen3.5/3.8):  <function=f><parameter=x>1</parameter></function>
  glm         (GLM-4.5+/5.x): f<arg_key>x</arg_key><arg_value>1</arg_value>

Streaming holds back text from `<tool_call>` until the matching `</tool_call>`, then
emits the call whole (name + full arguments), which OpenAI clients accept as one delta.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from lily.utils import longest_partial_suffix

TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"

_QWEN_FUNC = re.compile(r"<function=(.*?)>(.*?)(?:</function>|$)", re.DOTALL)
_QWEN_PARAM = re.compile(r"<parameter=(.*?)>\n?(.*?)\n?</parameter>", re.DOTALL)
_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def detect_tool_format(chat_template: Optional[str]) -> Optional[str]:
    """Pick the parser from the chat template's own tool-call instructions."""
    if not chat_template or TOOL_START not in chat_template:
        return None
    if "<arg_key>" in chat_template:
        return "glm"
    if "<function=" in chat_template:
        return "qwen3_coder"
    return "hermes"


def convert_value(value: str, json_type: Optional[str]) -> Any:
    """XML formats write string args raw and everything else as JSON."""
    if json_type == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


class ToolParser:
    def __init__(self, fmt: str, tools: List[Dict[str, Any]]):
        self.fmt = fmt
        # name -> {param: json type}, for the XML formats' value conversion.
        self.param_types: Dict[str, Dict[str, Optional[str]]] = {}
        for tool in tools:
            fn = tool.get("function", tool)
            props = (fn.get("parameters") or {}).get("properties") or {}
            self.param_types[fn.get("name", "")] = {k: v.get("type") for k, v in props.items()}
        self._buffer = ""
        self._in_call = False

    def parse_call(self, body: str) -> Optional[Dict[str, str]]:
        """One `<tool_call>` body -> {"name", "arguments" (JSON string)}, None if malformed."""
        if self.fmt == "hermes":
            try:
                call = json.loads(body)
                name, args = call["name"], call.get("arguments", {})
            except (ValueError, KeyError, TypeError):
                return None
        else:
            if self.fmt == "qwen3_coder":
                match = _QWEN_FUNC.search(body)
                if match is None:
                    return None
                name, pairs = match.group(1), _QWEN_PARAM.findall(match.group(2))
            else:
                name, pairs = body.split("<arg_key>", 1)[0], _GLM_ARG.findall(body)
            name = name.strip()
            if not name:
                return None
            types = self.param_types.get(name, {})
            args = {k.strip(): convert_value(v, types.get(k.strip())) for k, v in pairs}
        if isinstance(args, str):  # some models emit the arguments already JSON-encoded
            return {"name": name, "arguments": args}
        return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}

    def parse_stream_chunk(self, chunk: str) -> Tuple[str, List[Dict[str, str]]]:
        self._buffer += chunk
        content, calls = "", []
        while True:
            if self._in_call:
                index = self._buffer.find(TOOL_END)
                if index < 0:
                    break
                body = self._buffer[:index]
                self._buffer = self._buffer[index + len(TOOL_END):]
                self._in_call = False
                call = self.parse_call(body.strip())
                if call is None:
                    content += TOOL_START + body + TOOL_END
                else:
                    calls.append(call)
                continue
            index = self._buffer.find(TOOL_START)
            if index >= 0:
                content += self._buffer[:index]
                self._buffer = self._buffer[index + len(TOOL_START):]
                self._in_call = True
                continue
            hold = longest_partial_suffix(self._buffer, TOOL_START)
            content += self._buffer[: len(self._buffer) - hold]
            self._buffer = self._buffer[len(self._buffer) - hold:]
            break
        # Whitespace between or around calls is formatting, not content.
        return (content if content.strip() or not calls else ""), calls

    def flush(self) -> Tuple[str, List[Dict[str, str]]]:
        """End of request: a call cut off before `</tool_call>` is parsed if it can be, else returned as text."""
        pending, self._buffer = self._buffer, ""
        if not self._in_call:
            return pending, []
        self._in_call = False
        call = self.parse_call(pending.strip())
        return ("", [call]) if call is not None else (TOOL_START + pending, [])
