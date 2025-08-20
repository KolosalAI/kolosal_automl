# syntax=docker/dockerfile:1.4

# Multi-stage build for optimized production deployment
ARG PYTHON_VERSION=3.10.11

#======================================
# Build stage - compile dependencies
#======================================
FROM python:${PYTHON_VERSION}-slim as builder

# Build arguments for customization
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies with caching and optimization
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel setuptools && \
    pip install --no-deps -r requirements.txt && \
    pip install -e .[api,performance] --no-deps && \
    pip check

#======================================
# Runtime stage - minimal production image
#======================================
FROM python:${PYTHON_VERSION}-slim as runtime

# Add labels for better container management
LABEL maintainer="Genta Technology <contact@genta.tech>"
LABEL version=${VERSION:-"0.2.0"}
LABEL description="Production-ready AutoML API with advanced optimization"
LABEL build-date=${BUILD_DATE}
LABEL vcs-ref=${VCS_REF}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Create non-privileged user for security
ARG UID=10001
ARG GID=10001
RUN groupadd -g ${GID} appgroup && \
    adduser \
    --uid ${UID} \
    --gid ${GID} \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    appuser

# Create application directories with proper permissions
RUN mkdir -p /app/logs /app/models /app/temp_data /app/static /app/checkpoints /app/configs && \
    chown -R appuser:appgroup /app && \
    chmod -R 755 /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Install the package in development mode for modules access
RUN pip install -e . --no-deps

# Compile Python code to bytecode for better performance (optional)
RUN python -m compileall -b /app/modules/ || true

# Switch to non-privileged user
USER appuser

# Environment variables for production
ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONOPTIMIZE=1

# API Configuration
ENV API_ENV=production \
    API_DEBUG=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4

# Security Configuration  
ENV SECURITY_ENV=production \
    SECURITY_ENFORCE_HTTPS=false \
    SECURITY_REQUIRE_API_KEY=true \
    SECURITY_ENABLE_RATE_LIMITING=true \
    SECURITY_RATE_LIMIT_REQUESTS=100 \
    SECURITY_RATE_LIMIT_WINDOW=60 \
    SECURITY_ENABLE_JWT=true \
    SECURITY_ENABLE_AUDIT_LOGGING=true

# Performance Configuration
ENV BATCH_MAX_SIZE=128 \
    BATCH_TIMEOUT=0.02 \
    ENABLE_MONITORING=true \
    MAX_REQUEST_SIZE=10485760

# Logging Configuration
ENV LOG_LEVEL=INFO \
    ENABLE_AUDIT_LOGGING=true

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Add startup script for better initialization
COPY --chown=appuser:appgroup <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

echo "Starting kolosal AutoML API..."
echo "Environment: \${API_ENV}"
echo "Debug: \${API_DEBUG}"
echo "Host: \${API_HOST}"
echo "Port: \${API_PORT}"
echo "Security Environment: \${SECURITY_ENV}"

# Validate critical directories exist
mkdir -p /app/logs /app/models /app/temp_data

# Start the application
exec python -m uvicorn modules.api.app:app \
    --host \${API_HOST:-0.0.0.0} \
    --port \${API_PORT:-8000} \
    --workers \${API_WORKERS:-1} \
    --access-log \
    --loop uvloop
EOF

RUN chmod +x /app/entrypoint.sh

# Use entrypoint script for better initialization
ENTRYPOINT ["/app/entrypoint.sh"]

# Alternative CMD for direct Python execution (if needed)
# CMD ["python", "-m", "modules.api.app"]
