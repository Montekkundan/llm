FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY course_tools ./course_tools
COPY scripts ./scripts

RUN pip install --no-cache-dir uv && uv pip install --system .
RUN python scripts/deployment/build_demo_checkpoint.py

EXPOSE 8000

CMD ["python", "scripts/fastapi_chat_app/serve.py"]
