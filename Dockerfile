FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 nexus && \
    adduser --system --uid 1001 --ingroup nexus nexus

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/nexus/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/nexus/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/nexus/.cache/huggingface

WORKDIR /app

COPY --chown=nexus:nexus . .

RUN mkdir -p /app/instance /app/static/uploads /home/nexus/.cache/huggingface && \
    chown -R nexus:nexus /app/instance /app/static/uploads /home/nexus/.cache

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER nexus

ENTRYPOINT ["/entrypoint.sh"]
