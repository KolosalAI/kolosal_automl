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
RUN apt-get update && apt-get install -y \
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

# Download dependencies as a separate step to take advantage of Docker's caching.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/temp_data /app/static && \
    chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Environment variables with production defaults
ENV PYTHONPATH="/app" \
    API_ENV=production \
    API_DEBUG=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4 \
    REQUIRE_API_KEY=true \
    ENABLE_MONITORING=true \
    BATCH_MAX_SIZE=128 \
    BATCH_TIMEOUT=0.02 \
    RATE_LIMIT_REQUESTS=1000 \
    RATE_LIMIT_WINDOW=60

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application with proper signal handling
CMD ["python", "-m", "modules.api.app"]
