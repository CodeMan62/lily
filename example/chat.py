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
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="-1 disables")
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
        extra: dict = {"top_p": args.top_p, "extra_body": {"top_k": args.top_k}}
        if args.no_think:
            extra["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}

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
            finish_reason = None
            thinking_open = False
            for chunk in stream:
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if getattr(delta, "reasoning_content", None):
                    # stream the thinking as it arrives (dim), instead of holding it back
                    if not thinking_open:
                        print("\033[2m(thinking) ", end="", flush=True)
                        thinking_open = True
                    reasoning += delta.reasoning_content
                    print(delta.reasoning_content, end="", flush=True)
                if delta.content:
                    if thinking_open:
                        print("\033[0m\n", flush=True)
                        thinking_open = False
                    answer += delta.content
                    print(delta.content, end="", flush=True)
            if thinking_open:
                print("\033[0m")
            if not answer:
                print(f"(no answer: finish_reason={finish_reason}, {len(reasoning)} chars of thinking"
                      f"{'; raise --max-tokens' if finish_reason == 'length' else ''})")
            print()

        messages.append({"role": "assistant", "content": answer})
        print()


if __name__ == "__main__":
    main()
