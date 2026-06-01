# ============================================================
# ARA-1 Docker Deployment
# ============================================================
# Multi-stage build for production deployment.
#
# Build: docker build -t ara-1:latest .
# Run:   docker run -p 8000:8000 --env-file .env ara-1:latest
# ============================================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Production stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/chroma data/episodic data/checkpoints data/evaluations \
    data/telemetry data/benchmarks reports

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Expose FastAPI port
EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
