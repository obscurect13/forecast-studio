#!/bin/bash
set -e

echo "========================================"
echo "  Starting Forecast Studio"
echo "========================================"

# Start FastAPI in background
echo "[1/2] Starting FastAPI on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 2

# Check if FastAPI is still running using /proc (available in all Linux)
if [ -d "/proc/$FASTAPI_PID" ]; then
    echo "✓ FastAPI started (PID: $FASTAPI_PID)"
else
    echo "✗ FastAPI failed to start"
    exit 1
fi

# Start Streamlit in foreground (keeps container alive)
echo "[2/2] Starting Streamlit on port 8501..."
streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false
