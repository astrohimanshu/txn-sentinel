FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PORT=8000

# LightGBM links against libgomp.so.1, absent from python:3.11-slim. Without this
# the image builds cleanly and then fails at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
# Trained model, operating threshold and replay sample. Committed rather than fetched
# at runtime so the image is reproducible from a tagged release.
COPY models ./models

RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn txn_sentinel.api:app --host 0.0.0.0 --port ${PORT}"]
