import sys
import os
import tempfile
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import numpy as np
import pandas as pd
import joblib

# ── Project Directory Setup ──────────────────────────────────────────────────
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR     = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)    # Ensures Python finds models.py, prepare_data.py, etc.

from models import get_model, WINDOW
from prepare_data import prepare_data
from train_select_best_model import run_training
from special_models import train_prophet_model, train_arima_model, predict_prophet, predict_arima
from logger_config import setup_logger

logger = setup_logger(__name__)

MODELS_DIR  = os.path.join(ROOT_DIR, "models")

# ── Filter noisy endpoints from uvicorn access logs ───────────────────────────
import logging
class _AccessFilter(logging.Filter):
    _SKIP = ("/health", "/job-status")
    def filter(self, record):
        return not any(skip in record.getMessage() for skip in self._SKIP)
logging.getLogger("uvicorn.access").addFilter(_AccessFilter())

app = FastAPI(title="Forecast API - Async Job System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for jobs (For production, use Redis or a persistent DB)
jobs: Dict[str, dict] = {}

# ── Background Training Logic ────────────────────────────────────────────────

def background_prepare_and_train(job_id: str, content: bytes, target_col: Optional[str] = None):
    """Function executed in the background to handle data preparation and model training."""
    try:
        jobs[job_id]["status"] = "preparing"
        logger.info(f"Job {job_id}: Starting data preparation")

        # Write uploaded bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Prepare data (this can be slow for large files)
            X, X_lstm, y, scaler = prepare_data(tmp_path, target_col=target_col or None)
            logger.info(f"Job {job_id}: Data preparation complete - {len(y)} samples")

            # Update status to training
            jobs[job_id]["status"] = "training"
            logger.info(f"Job {job_id}: Starting model training")

            # run_training saves the best model to MODELS_DIR automatically
            best, all_results = run_training(X, X_lstm, y, models_dir=MODELS_DIR)
            logger.info(f"Job {job_id}: Training complete - best model: {best['model']}")

            # Group scores by model for final report
            scores_by_model = {}
            for r in all_results:
                name = r["model"]
                if name not in scores_by_model or r["mse"] < scores_by_model[name]["mse"]:
                    scores_by_model[name] = {
                        "mse":         float(r["mse"]),
                        "rmse":        float(np.sqrt(r["mse"])),
                        "mae":         float(r["mae"]),
                        "r2":          float(r["r2"]),
                        "best_params": r["params"],
                    }

            # Update job status with results
            jobs[job_id].update({
                "status": "completed",
                "result": {
                    "results":                 scores_by_model,
                    "best_model":              best["model"],
                    "best_params":             best["params"],
                    "best_score":              float(best["mse"]),
                    "samples_after_windowing": int(len(y)),
                },
                "completed_at": datetime.now().isoformat()
            })
            logger.info(f"Job {job_id}: Job completed successfully")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Job {job_id}: Failed with error: {str(e)}", exc_info=True)
        jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compare-models")
async def compare_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_col: Optional[str] = None,
):
    """
    Triggers model training in the background and returns a job_id immediately.
    Prevents HTTP timeout during long training processes.
    Data preparation is also done in the background to avoid blocking on large files.
    """
    # Read file content immediately (this is fast)
    content = await file.read()

    # Validate file size (optional - prevent extremely large uploads)
    file_size = len(content)
    if file_size > 100 * 1024 * 1024:  # 100 MB limit
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum size is 100 MB."
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "file_size": file_size
    }

    # Dispatch the entire pipeline (data prep + training) to the background
    background_tasks.add_task(background_prepare_and_train, job_id, content, target_col)

    return {"job_id": job_id, "status": "started", "file_size": file_size}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Endpoint for the client (Streamlit) to poll for training results."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/predict-best")
async def predict_best(
    file: UploadFile = File(...),
    target_col: Optional[str] = None,
    n_steps: int = 20,
):
    """
    Loads saved best_config.pkl and the corresponding model file
    to return predictions on new data + n_steps future forecast.
    """
    config_path = os.path.join(MODELS_DIR, "best_config.pkl")
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=404,
            detail="best_config.pkl not found. Please run training via /compare-models first."
        )

    best       = joblib.load(config_path)
    model_name = best["model"]
    params     = best["params"]

    content = await file.read()
    try:
        # Write uploaded bytes to a temporary file, then call prepare_data()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            X, X_lstm, y, scaler = prepare_data(tmp_path, target_col=target_col or None)
        finally:
            os.remove(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prediction logic based on model type
    if model_name == "lstm":
        from tensorflow.keras.models import load_model
        model_path = os.path.join(MODELS_DIR, "best_model.h5")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="best_model.h5 not found.")
        model        = load_model(model_path)
        preds_scaled = model.predict(X_lstm).flatten()

    elif model_name == "prophet":
        # Prophet requires special handling
        model_path = os.path.join(MODELS_DIR, "best_model_prophet.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="best_model_prophet.pkl not found.")

        model = joblib.load(model_path)

        # Prepare Prophet data (use scaled y — Prophet was trained on scaled values)
        dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(y), freq='D')
        prophet_df = pd.DataFrame({'ds': dates, 'y': y})
        # Required for logistic growth
        if params.get('growth') == 'logistic':
            prophet_df['cap'] = float(y.max()) * 1.2

        # Get in-sample predictions and future forecast
        predictions_scaled, future_scaled = predict_prophet(model, prophet_df, n_steps)

        # Align lengths (Prophet may return all history)
        min_len = min(len(y), len(predictions_scaled))
        predictions_scaled = predictions_scaled[-min_len:]
        y_aligned = y[-min_len:]

        # Inverse transform to original scale
        preds   = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(np.array(y_aligned).reshape(-1, 1)).flatten()
        future  = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten().tolist()

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        r2  = r2_score(actuals, preds)

        return {
            "model":  model_name,
            "params": params,
            "metrics": {
                "mse":  float(mse),
                "mae":  float(mae),
                "rmse": float(np.sqrt(mse)),
                "r2":   float(r2),
            },
            "predictions": preds.tolist()[-50:],
            "actuals":     actuals.tolist()[-50:],
            "future":      future,
            "n_steps":     n_steps,
        }

    elif model_name == "arima":
        # ARIMA requires special handling
        model_path = os.path.join(MODELS_DIR, "best_model_arima.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="best_model_arima.pkl not found.")

        model = joblib.load(model_path)

        # Get in-sample fitted values and future forecast (both in scaled space)
        predictions_scaled, future_scaled = predict_arima(model, n_steps)

        # Align lengths
        min_len = min(len(y), len(predictions_scaled))
        predictions_scaled = np.array(predictions_scaled[-min_len:])
        y_aligned = y[-min_len:]

        # Inverse transform to original scale
        preds   = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(np.array(y_aligned).reshape(-1, 1)).flatten()
        future_arr = np.array(future_scaled) if isinstance(future_scaled, (list, np.ndarray)) else np.array([future_scaled])
        future  = scaler.inverse_transform(future_arr.reshape(-1, 1)).flatten().tolist()

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        r2  = r2_score(actuals, preds)

        return {
            "model":  model_name,
            "params": params,
            "metrics": {
                "mse":  float(mse),
                "mae":  float(mae),
                "rmse": float(np.sqrt(mse)),
                "r2":   float(r2),
            },
            "predictions": preds.tolist()[-50:],
            "actuals":     actuals.tolist()[-50:],
            "future":      future,
            "n_steps":     n_steps,
        }

    else:
        # Standard sklearn-compatible models
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="best_model.pkl not found.")
        model        = joblib.load(model_path)
        preds_scaled = model.predict(X.reshape(X.shape[0], -1))

    # Revert scaling for real-world values
    preds   = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    r2  = r2_score(actuals, preds)

    # ── Future forecast (n_steps ahead) ──────────────────────────────────────
    # Seed with the last WINDOW scaled values then predict step by step
    last_window = y[-WINDOW:].tolist()  # already scaled
    future_scaled = []

    if model_name == "lstm":
        # Use model.__call__ instead of model.predict to avoid tf.function
        # retracing on every step when calling predict() in a loop
        for _ in range(n_steps):
            x_in     = np.array(last_window[-WINDOW:], dtype=np.float32).reshape(1, WINDOW, 1)
            next_val = float(model(x_in, training=False).numpy().flatten()[0])
            future_scaled.append(next_val)
            last_window.append(next_val)
    else:
        # Standard sklearn-compatible models (Prophet/ARIMA already returned above)
        for _ in range(n_steps):
            x_in     = np.array(last_window[-WINDOW:], dtype=np.float32).reshape(1, -1)
            next_val = float(model.predict(x_in)[0])
            future_scaled.append(next_val)
            last_window.append(next_val)

    future = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten().tolist()

    return {
        "model":  model_name,
        "params": params,
        "metrics": {
            "mse":  float(mse),
            "mae":  float(mae),
            "rmse": float(np.sqrt(mse)),
            "r2":   float(r2),
        },
        "predictions": preds.tolist()[-50:],
        "actuals":     actuals.tolist()[-50:],
        "future":      future,
        "n_steps":     n_steps,
    }