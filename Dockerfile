# Multi-stage build for optimized image size
FROM python:3.11-slim as builder

WORKDIR /app

# Copy requirements and install dependencies to a temporary location
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Final stage
FROM python:3.11-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Set the working directory in the container
WORKDIR /app

# Create logs directory
RUN mkdir -p /app/logs

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Only download the neural model if explicitly requested (saves ~400MB for LSA-only builds)
ARG EMBEDDING_TYPE=lsa
RUN if [ "$EMBEDDING_TYPE" = "neural" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Start the application using Gunicorn and Uvicorn workers for production
CMD exec gunicorn -k uvicorn.workers.UvicornWorker -w ${WORKERS:-4} --preload -b 0.0.0.0:${PORT:-8000} --access-logfile - --error-logfile - --log-level ${LOG_LEVEL:-info} main:app

