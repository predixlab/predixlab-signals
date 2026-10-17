# Use Python 3.12 slim image as base
FROM python:3.12-slim

LABEL maintainer="predixlab-signals"
LABEL description="Swing-point (local high/low) prediction producing daily buy/sell advice"
LABEL version="0.1.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# LightGBM needs libgomp at runtime; the compilers are only needed while wheels
# without a musl/slim build are compiled, so they are dropped again afterwards.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency metadata first so the layer caches across source-only changes.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN pip install uv==0.13.1 \
    && uv sync --frozen --no-dev --extra job

COPY . .

# The shared PredixLab data share is mounted here by the Container Apps job; the
# collector init container writes it and this process only reads it.
ENV PREDIXLAB_DATA=/data
# Rebuildable datasets and research outputs are container-local scratch.
ENV PIVOT_ARTIFACTS=artifacts

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

# Default command: the scheduled pipeline.
CMD ["python", "-m", "predixlab_signals", "daily"]
