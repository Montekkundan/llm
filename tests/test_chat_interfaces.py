import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from picollm.accelerated.chat import cli, web
from picollm.accelerated.chat.api import (
    ChatRequest,
    ChatValidationError,
    GenerationDefaults,
    resolve_generation_settings,
    validate_generation_budget,
)


class _FakeTokenizer:
    SPECIAL = {
        "<|user_start|>": 10,
        "<|user_end|>": 11,
        "<|assistant_start|>": 12,
        "<|assistant_end|>": 13,
    }

    def get_bos_token_id(self):
        return 1

    def encode_special(self, token: str):
        return self.SPECIAL[token]

    def encode(self, text: str):
        count = max(1, len(text.split()))
        return [20] * count

    def decode(self, tokens):
        pieces = {
            200: "Hello",
            201: " world",
            202: "!",
        }
        return "".join(pieces.get(token, "") for token in tokens)


class _FakeEngine:
    last_call = None

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt_tokens, **kwargs):
        _FakeEngine.last_call = {
            "prompt_tokens": list(prompt_tokens),
            "kwargs": dict(kwargs),
        }
        yield ([200], None)
        yield ([201], None)
        yield ([13], None)


def _fake_load_model(source, device, phase, model_tag=None, step=None):
    model = SimpleNamespace(config=SimpleNamespace(sequence_len=24))
    tokenizer = _FakeTokenizer()
    meta = {
        "_checkpoint": {
            "model_tag": model_tag or "d2",
            "step": step or 7,
        }
    }
    return model, tokenizer, meta


def _fake_compute_init(device_type):
    return False, 0, 0, 1, torch.device(device_type)


class ChatInterfaceTests(unittest.TestCase):
    def test_resolve_generation_settings_prefers_request_values(self):
        request = ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            top_k=7,
            top_p=0.9,
            min_p=0.05,
            max_tokens=12,
            seed=123,
        )
        defaults = GenerationDefaults(
            temperature=0.8,
            top_k=50,
            top_p=None,
            min_p=None,
            max_tokens=64,
            seed=None,
        )

        settings = resolve_generation_settings(request, defaults, random_seed_fn=lambda: 99)

        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.top_k, 7)
        self.assertEqual(settings.top_p, 0.9)
        self.assertEqual(settings.min_p, 0.05)
        self.assertEqual(settings.max_tokens, 12)
        self.assertEqual(settings.seed, 123)

    def test_validate_generation_budget_rejects_overlong_request(self):
        request = ChatRequest(messages=[{"role": "user", "content": "hello there"}], max_tokens=10)
        defaults = GenerationDefaults(temperature=0.8, top_k=50, top_p=None, min_p=None, max_tokens=10, seed=1)

        with self.assertRaises(ChatValidationError):
            validate_generation_budget(
                _FakeTokenizer(),
                context_window=4,
                request=request,
                defaults=defaults,
                random_seed_fn=lambda: 1,
            )

    def test_cli_main_runs_single_prompt_with_requested_controls(self):
        _FakeEngine.last_call = None
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(
                [
                    "-p",
                    "Who are you?",
                    "--top-p",
                    "0.9",
                    "--min-p",
                    "0.05",
                    "--seed",
                    "7",
                    "--max-tokens",
                    "12",
                ],
                load_model_fn=_fake_load_model,
                engine_cls=_FakeEngine,
                compute_init_fn=_fake_compute_init,
                autodetect_device_type_fn=lambda: "cpu",
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Assistant: Hello world", stdout.getvalue())
        self.assertEqual(_FakeEngine.last_call["kwargs"]["top_p"], 0.9)
        self.assertEqual(_FakeEngine.last_call["kwargs"]["min_p"], 0.05)
        self.assertEqual(_FakeEngine.last_call["kwargs"]["seed"], 7)
        self.assertEqual(_FakeEngine.last_call["kwargs"]["max_tokens"], 12)

    def test_web_app_exposes_health_and_chat_endpoints(self):
        _FakeEngine.last_call = None
        args = web.build_parser().parse_args([])
        app = web.create_app(
            args,
            load_model_fn=_fake_load_model,
            engine_cls=_FakeEngine,
            compute_init_fn=_fake_compute_init,
            autodetect_device_type_fn=lambda: "cpu",
        )

        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["ready"])

            payload = {
                "model": "picollm-chat",
                "messages": [{"role": "user", "content": "Say hi"}],
                "stream": False,
                "top_p": 0.91,
                "min_p": 0.04,
                "seed": 55,
                "max_tokens": 8,
            }
            response = client.post("/v1/chat/completions", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["choices"][0]["message"]["content"],
                "Hello world",
            )
            self.assertEqual(_FakeEngine.last_call["kwargs"]["top_p"], 0.91)
            self.assertEqual(_FakeEngine.last_call["kwargs"]["min_p"], 0.04)
            self.assertEqual(_FakeEngine.last_call["kwargs"]["seed"], 55)

            with client.stream("POST", "/chat/completions", json=payload) as stream_response:
                body = "\n".join(stream_response.iter_lines())
            self.assertEqual(stream_response.status_code, 200)
            self.assertIn('"token": "Hello"', body)
            self.assertIn('"done": true', body)

            too_long = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello there"}],
                    "max_tokens": 30,
                },
            )
            self.assertEqual(too_long.status_code, 400)
            self.assertIn("context window", too_long.json()["detail"])

    def test_web_main_invokes_uvicorn_with_created_app(self):
        calls = {}

        def fake_create_app(args):
            calls["args"] = args
            return "fake-app"

        def fake_uvicorn_run(app, host, port):
            calls["app"] = app
            calls["host"] = host
            calls["port"] = port

        exit_code = web.main(
            ["--host", "127.0.0.1", "--port", "8123"],
            create_app_fn=fake_create_app,
            uvicorn_run=fake_uvicorn_run,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls["app"], "fake-app")
        self.assertEqual(calls["host"], "127.0.0.1")
        self.assertEqual(calls["port"], 8123)


if __name__ == "__main__":
    unittest.main()
