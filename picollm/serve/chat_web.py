from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the picoLLM FastAPI chat app.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    os.environ["PICOLLM_MODEL"] = args.model
    if args.adapter:
        os.environ["PICOLLM_ADAPTER"] = args.adapter
    os.environ["PICOLLM_DEVICE"] = args.device
    os.environ["PICOLLM_QUANTIZATION"] = args.quantization
    uvicorn.run("picollm.serve.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
