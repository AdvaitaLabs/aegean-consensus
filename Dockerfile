# aegean-consensus service Dockerfile.
# Builds the multi-agent consensus engine and serves on port 8000.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements*.txt ./
RUN pip install --user --no-cache-dir \
    fastapi uvicorn httpx pydantic requests \
    openai anthropic

# ----------------------------- runtime -----------------------------

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app/src

WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY src /app/src
COPY main.py /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fs http://localhost:8000/api/v1/health || exit 1

CMD ["python", "main.py", "--port", "8000"]
