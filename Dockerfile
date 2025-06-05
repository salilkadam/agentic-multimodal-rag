# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model during build
# This avoids runtime download issues in environments without internet access
ENV HF_HOME=/app/models/huggingface
ENV TRANSFORMERS_CACHE=/app/models/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/huggingface/sentence_transformers
RUN mkdir -p /app/models/huggingface/transformers /app/models/huggingface/sentence_transformers

# Copy and run model download script
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create directories for models and temp files
RUN mkdir -p /app/models /tmp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app/main.py"]
