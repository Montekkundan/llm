from __future__ import annotations

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the accelerated picoLLM chat server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008", help="Base URL for the accelerated picoLLM server")
    parser.add_argument("--model", default="picollm-chat", help="Model id for the OpenAI-compatible endpoint")
    return parser.parse_args()


def stream_chat_completion(base_url: str, payload: dict) -> str:
    response = requests.post(f"{base_url}/chat/completions", json=payload, timeout=30, stream=True)
    response.raise_for_status()
    pieces: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        event = json.loads(raw_line[6:])
        if event.get("token"):
            pieces.append(event["token"])
        if event.get("done"):
            break
    return "".join(pieces)


def run_openai_completion(base_url: str, payload: dict) -> str:
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain tokenization in one short paragraph."},
    ]

    print("health:", requests.get(f"{base_url}/health", timeout=10).json())

    accelerated_payload = {
        "messages": messages,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 40,
        "seed": 42,
    }
    print("chat/completions:", stream_chat_completion(base_url, accelerated_payload))

    openai_payload = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 40,
        "seed": 42,
    }
    print("v1/chat/completions:", run_openai_completion(base_url, openai_payload))
