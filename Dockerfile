# ── SpeakSense Docker Container ───────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3

# Set working directory
WORKDIR /app

# Install system dependencies for audio decoding & libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application directories & configuration
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY features/ ./features/
COPY reports/ ./reports/
COPY .streamlit/ ./.streamlit/
COPY packages.txt .
COPY README.md .

# Expose Streamlit default port
EXPOSE 8501

# Health check to monitor container status
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit application
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
