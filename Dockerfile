# ── Single Dockerfile — runs both FastAPI and Streamlit ──────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY start.sh ./start.sh

# Models directory (persisted via volume)
RUN mkdir -p /app/models

# Make src importable by both services
ENV PYTHONPATH="/app/src"

# Make startup script executable
RUN chmod +x /app/start.sh

EXPOSE 8000 8501

CMD ["/app/start.sh"]
