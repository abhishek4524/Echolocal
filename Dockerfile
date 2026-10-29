# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# EchoLocal — GPU-accelerated AI Video Dubbing Pipeline
# Base image: PyTorch 2.6 + CUDA 12.4 (matches local dev environment)
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="Abhishek <https://github.com/abhishek4524>"
LABEL description="EchoLocal: Local AI Video Dubbing & Voice Cloning Pipeline"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pytest pytest-mock

# ── Application code ───────────────────────────────────────────────────────────
COPY src/       ./src/
COPY api/       ./api/
COPY tests/     ./tests/
COPY app.py     ./app.py
COPY run_pipeline.py ./run_pipeline.py
COPY benchmark.py    ./benchmark.py

# ── Create runtime directories ─────────────────────────────────────────────────
RUN mkdir -p input output models logs jobs temp

# ── Model cache (mount a volume in production) ────────────────────────────────
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# ── Expose ports ───────────────────────────────────────────────────────────────
#   8000 → FastAPI REST API
#   8501 → Streamlit GUI
EXPOSE 8000 8501

# ── Default: start the REST API ────────────────────────────────────────────────
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
