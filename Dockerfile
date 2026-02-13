# syntax=docker/dockerfile:1

# Multi-stage build for optimized production deployment
ARG PYTHON_VERSION=3.10.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies — use PyTorch CPU-only index as extra source
# so torch resolves to CPU wheel while other deps resolve from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the source code into the container.
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/temp_data /app/static && \
    chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Environment variables
ENV PYTHONPATH="/app" \
    GRADIO_SERVER_NAME=0.0.0.0

# Railway sets PORT env var. The Gradio app reads it via os.environ.get("PORT").
CMD python app.py
