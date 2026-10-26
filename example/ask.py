"""Ask the lily server a single question and print the answer.

Start the server first:

    uv run lily-serve --model-path Qwen/Qwen3-0.6B --dtype bfloat16

Then:

    uv run python example/ask.py "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import sys

import openai


def main() -> None:
    parser = argparse.ArgumentParser("lily ask example")
    parser.add_argument("question", nargs="?", help="the question to ask")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="served model name")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    question = args.question or input("question> ")
    if not question.strip():
        sys.exit("no question given")

    client = openai.Client(base_url=args.base_url, api_key="none")
    reply = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": question}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(reply.choices[0].message.content)


if __name__ == "__main__":
    main()
