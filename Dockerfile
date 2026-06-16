# =============================================================================
# Adaptive Speculative Decoding — Multi-stage Dockerfile
# =============================================================================
#
# Usage:
#   # Build GPU image
#   docker build -t adaptive-speculative-decoding:latest -f Dockerfile .
#
#   # Run experiments
#   docker run --rm -it --gpus all -v $(pwd):/app adaptive-speculative-decoding:latest \
#     python src/main.py --smoke
#
#   # Run with MLflow tracking
#   docker-compose up -d
#   docker run --rm -it --gpus all -v $(pwd):/app adaptive-speculative-decoding:latest \
#     python src/main.py --mlflow-tracking-uri=http://host.docker.internal:5000
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy only pyproject.toml + lockfile for dependency install
COPY pyproject.toml ./

# Install dependencies (fast: cache layer, only installs what changed)
RUN uv pip install --system --compile-bytecode -e ".[dev]"

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal CUDA-enabled image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Metadata
LABEL org.opencontainers.image.title="Adaptive Speculative Decoding"
LABEL org.opencontainers.image.description="Adaptive Speculative Decoding Framework"
LABEL org.opencontainers.image.source="https://github.com/Andrchest/Adaptive-speculative-decoding"
LABEL org.opencontainers.image.licenses="MIT"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 from deadsnakes (matches builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# Copy application
COPY pyproject.toml .
COPY src/ src/

# Default command
CMD ["python", "src/main.py", "--help"]

# Expose ports (MLflow + API)
EXPOSE 5000 8000
