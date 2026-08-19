FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Render's free tier (and other small/cgroup-limited containers) can trip a
# joblib/loky bug that miscounts physical CPU cores as 0 and crashes before
# training starts -- fixed at the code level too (app/risk.py), set here as
# well so it's explicit at the container level.
ENV LOKY_MAX_CPU_COUNT=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

# Train the risk model and generate the sample invoice at build time so the
# container starts ready to serve, without a training step on first request.
RUN python scripts/generate_synthetic_claims.py && python scripts/generate_sample_invoice.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
