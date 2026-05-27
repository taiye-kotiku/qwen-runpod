#!/usr/bin/env python3
"""
Test script for the Dolphin-Mixtral Modal endpoint.

Usage
-----
  # Non-streaming
  python test_modal.py --url https://<your-app>--serve.modal.run

  # Streaming
  python test_modal.py --url https://<your-app>--serve.modal.run --stream

  # Custom prompt
  python test_modal.py --url <url> --prompt "Explain quantum entanglement simply."
"""

import argparse
import json
import sys
import urllib.request


def post(url: str, payload: dict, stream: bool) -> None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        if stream:
            print("[streaming]\n", flush=True)
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        print(text, end="", flush=True)
            print()  # final newline
        else:
            body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            print("[response]\n")
            print(content)


def health_check(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            print(f"Health: {data}")
            return data.get("status") == "ok"
    except Exception as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Modal Dolphin API")
    parser.add_argument("--url", required=True, help="Base URL of the Modal endpoint")
    parser.add_argument("--prompt", default="Who are you and what can you do?")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--system",
        default="You are Dolphin, an uncensored and unbiased AI assistant.",
    )
    args = parser.parse_args()

    base = args.url.rstrip("/")

    print(f"Endpoint : {base}")
    print(f"Prompt   : {args.prompt}\n")

    if not health_check(base):
        sys.exit(1)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    payload = {
        "messages": messages,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    post(f"{base}/chat", payload, stream=args.stream)


if __name__ == "__main__":
    main()
