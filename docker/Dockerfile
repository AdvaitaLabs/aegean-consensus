FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install LLM provider SDKs
RUN pip install --no-cache-dir openai

# Copy source
COPY src/ ./src/
COPY main.py .
COPY pyproject.toml .
COPY scripts/ ./scripts/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]

