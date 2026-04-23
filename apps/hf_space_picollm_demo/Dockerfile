FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PICOLLM_BASE_DIR=/data/picollm \
    PICOLLM_DTYPE=float32 \
    PYTHONPATH=/app/llm \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN git clone https://github.com/Montekkundan/llm.git /app/llm

WORKDIR /app/llm
RUN git fetch origin main \
    && git checkout c2ef010dc4296f59ea370253f3d49f4c1192f2e7 \
    && pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

EXPOSE 7860

CMD ["python3", "-m", "picollm.accelerated.chat.web", "-i", "sft", "--device-type", "cpu", "--host", "0.0.0.0", "--port", "7860", "--max-tokens", "256"]
