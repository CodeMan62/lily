"""Interactive chat with a lula server.

Start the server in one terminal:

    uv run lula-serve --model-path Qwen/Qwen3-0.6B --dtype bfloat16

Then run this file in another:

    uv run python example/chat.py --model Qwen/Qwen3-0.6B

Type a question and press Enter. Commands: /reset, /exit.
"""

from __future__ import annotations

import argparse

import openai


def main() -> None:
    parser = argparse.ArgumentParser("lula chat example")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="served model name")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true", help="wait for the whole answer")
    args = parser.parse_args()

    client = openai.Client(base_url=args.base_url, api_key="none")
    messages: list[dict] = []

    print("lula chat. /reset to clear the history, /exit to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            messages.clear()
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": question})
        print("bot> ", end="", flush=True)

        if args.no_stream:
            reply = client.chat.completions.create(
                model=args.model, messages=messages,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            answer = reply.choices[0].message.content
            print(answer)
        else:
            stream = client.chat.completions.create(
                model=args.model, messages=messages,
                max_tokens=args.max_tokens, temperature=args.temperature, stream=True,
            )
            answer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    answer += delta.content
                    print(delta.content, end="", flush=True)
            print()

        messages.append({"role": "assistant", "content": answer})
        print()


if __name__ == "__main__":
    main()
