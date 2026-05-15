FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ───────────────────────────
COPY requirements.txt .

# ── Install Python deps ───────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────
COPY . .

# ── Create output/cache directories ──────────────────────────────────
RUN mkdir -p \
    outputs/models \
    outputs/reports \
    outputs/plots \
    outputs/logs \
    outputs/benchmarks \
    outputs/ablation \
    cache \
    data/raw \
    data/processed \
    data/sample_datasets

# ── Non-root user for security ────────────────────────────────────────
RUN useradd -m -u 1000 emmds \
    && chown -R emmds:emmds /app
USER emmds

# ── Environment ───────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose ports ──────────────────────────────────────────────────────
# 8000 = FastAPI backend
# 8501 = Streamlit frontend
EXPOSE 8000 8501

# ── Default: run the API ──────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
