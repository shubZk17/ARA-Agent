# ============================================================
# ARA-1 — single image: FastAPI backend + static web UI
# ============================================================
# Works on Hugging Face Docker Spaces (runs as uid 1000, no root writes)
# and on any container host. Port: $PORT if set (Render), else 7860 (HF).
#
#   docker build -t ara-1 .
#   docker run -p 7860:7860 --env-file .env ara-1
# ============================================================

FROM python:3.12-slim

# HF Spaces run the container as this non-root user; match it so pip's
# --user installs and data/ writes land somewhere writable.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

RUN mkdir -p logs data/chroma data/episodic data/evaluations \
    data/telemetry data/sec reports

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
