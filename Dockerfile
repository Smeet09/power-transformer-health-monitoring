FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=4

# libgomp1: OpenMP runtime needed by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY Dataset/ Dataset/
COPY src/ src/
COPY api/ api/
COPY tests/ tests/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8000/healthz || exit 1

# One command end to end: train + evaluate + reports, then serve the API
CMD ["sh", "-c", "python -m src.pipeline && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
