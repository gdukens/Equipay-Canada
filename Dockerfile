# EquiPay Canada - Docker Image
# Multi-stage build for optimized production image

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies stage ----
FROM base as dependencies

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Production stage ----
FROM base as production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY config.yaml .
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Create logs directory
RUN mkdir -p logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports for API and Dashboard
EXPOSE 8000 8501

# Default command: run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Development stage ----
FROM dependencies as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 isort jupyter

# Copy all files for development
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash devuser || true
RUN chown -R devuser:devuser /app 2>/dev/null || true

# Keep container running for development
CMD ["bash"]
