FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY docs/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 nexus && \
    adduser --system --uid 1001 --ingroup nexus nexus

COPY --from=builder /root/.local /home/nexus/.local
ENV PATH=/home/nexus/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/nexus/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/nexus/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/nexus/.cache/huggingface

WORKDIR /app

COPY --chown=nexus:nexus . .

RUN mkdir -p /app/instance /app/static/uploads && chown -R nexus:nexus /app/instance /app/static/uploads

RUN mkdir -p /home/nexus/.cache && chown -R nexus:nexus /home/nexus/.cache

EXPOSE 8000

USER nexus

COPY --chown=nexus:nexus entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
