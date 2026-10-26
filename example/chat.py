"""Interactive chat with a lily server.

Start the server in one terminal:

    uv run lily-serve --model-path Qwen/Qwen3-0.6B --dtype bfloat16

Then run this file in another:

    uv run python example/chat.py --model Qwen/Qwen3-0.6B

Type a question and press Enter. Commands: /reset, /exit.
"""

from __future__ import annotations

import argparse

import openai


def main() -> None:
    parser = argparse.ArgumentParser("lily chat example")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="served model name")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true", help="wait for the whole answer")
    parser.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking mode")
    args = parser.parse_args()

    client = openai.Client(base_url=args.base_url, api_key="none")
    messages: list[dict] = []

    print("lily chat. /reset to clear the history, /exit to quit.\n")
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
        extra: dict = {}
        if args.no_think:
            extra["chat_template_kwargs"] = {"enable_thinking": False}

        if args.no_stream:
            reply = client.chat.completions.create(
                model=args.model, messages=messages,
                max_tokens=args.max_tokens, temperature=args.temperature, **extra,
            )
            message = reply.choices[0].message
            if getattr(message, "reasoning_content", None):
                print(f"\033[2m(thinking) {message.reasoning_content}\033[0m")
            answer = message.content
            print(answer)
        else:
            stream = client.chat.completions.create(
                model=args.model, messages=messages,
                max_tokens=args.max_tokens, temperature=args.temperature, stream=True, **extra,
            )
            answer = ""
            reasoning = ""
            shown_thinking = False
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    if reasoning and not shown_thinking:
                        print(f"\033[2m(thinking) {reasoning}\033[0m", end="\n\n")
                        shown_thinking = True
                    answer += delta.content
                    print(delta.content, end="", flush=True)
                elif getattr(delta, "reasoning_content", None):
                    reasoning += delta.reasoning_content
            print()

        messages.append({"role": "assistant", "content": answer})
        print()


if __name__ == "__main__":
    main()
