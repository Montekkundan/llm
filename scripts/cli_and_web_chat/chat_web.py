from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fastapi_chat_app.app import app


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001, reload=False)
