from __future__ import annotations

import json
import os
from typing import Any

import requests

BASE_URL = "https://console.vast.ai/api/v0"


def require_api_key() -> str:
    token = os.environ.get("VAST_API_KEY", "").strip()
    if not token:
        raise SystemExit("Set VAST_API_KEY before using the Vast.ai helper scripts.")
    return token


def request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    token = require_api_key()
    response = requests.request(
        method=method.upper(),
        url=f"{BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload) if payload is not None else None,
        timeout=60,
    )
    if response.status_code >= 400:
        raise SystemExit(f"Vast.ai API request failed ({response.status_code}): {response.text}")
    return response.json()
