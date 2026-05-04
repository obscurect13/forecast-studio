from models import get_model
from param_grid_config import param_grid
from rolling_walk_forward_cv import rolling_cv
from prepare_data import prepare_data
from special_models import prophet_rolling_cv, arima_rolling_cv, train_prophet_model, train_arima_model
from logger_config import setup_logger

import os
import joblib
import pandas as pd
import numpy as np

logger = setup_logger(__name__)


def run_training(X, X_lstm, y, models_dir="../models"):
    """
    Iterates over param_grid, evaluates each config via rolling_cv,
    saves the best model and returns the `best` dict.
    """
    results = []
    total = sum(len(configs) for configs in param_grid.values())
    current = 0

    # Try to extract dates from original data if available
    dates = None
    try:
        # This assumes prepare_data might have stored date information
        # For now, we'll create sequential dates for Prophet
        dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=len(y), freq='D')
    except:
        dates = None

    for model_name, configs in param_grid.items():
        for config_index, params in enumerate(configs):
            current += 1

            try:
                # Handle different model types
                if model_name == "lstm":
                    model_fn = lambda **p: get_model(model_name, **p)
                    metrics = rolling_cv(X_lstm, y, model_fn, params, lstm=True)
                elif model_name == "prophet":
                    # Use special Prophet rolling CV with config_index
                    metrics = prophet_rolling_cv(y, dates, n_splits=3, config_index=config_index)
                elif model_name == "arima":
                    # Use special ARIMA rolling CV with config_index
                    metrics = arima_rolling_cv(y, n_splits=3, config_index=config_index)
                else:
                    # Standard sklearn-compatible models
                    model_fn = lambda **p: get_model(model_name, **p)
                    metrics = rolling_cv(X, y, model_fn, params, lstm=False)
            except Exception as e:
                logger.warning(f"[{current}/{total}] {model_name} skipped | params={params} — {e}")
                continue

            logger.info(
                f"[{current}/{total}] {model_name} | "
                f"MSE={metrics['mse']:.6f} | MAE={metrics['mae']:.6f} | R²={metrics['r2']:.4f} | "
                f"params={params}"
            )

            results.append({
                "model":  model_name,
                "params": params,
                "mse":    metrics["mse"],
                "mae":    metrics["mae"],
                "r2":     metrics["r2"],
            })

    if not results:
        raise RuntimeError("All model configurations failed during training.")

    best = sorted(results, key=lambda x: x["mse"])[0]
    logger.info(
        f"Best model: {best['model'].upper()} | "
        f"MSE={best['mse']:.6f} | MAE={best['mae']:.6f} | R²={best['r2']:.4f}"
    )

    # Train and save the best model
    os.makedirs(models_dir, exist_ok=True)

    if best["model"] == "lstm":
        logger.info("Fitting best LSTM model on full dataset...")
        best_model = get_model(best["model"], **best["params"])
        best_model.fit(X_lstm, y, epochs=10, verbose=0)
        model_path = os.path.join(models_dir, "best_model.h5")
        best_model.save(model_path)
        logger.info(f"LSTM model saved → {model_path}")

    elif best["model"] == "prophet":
        logger.info("Fitting best Prophet model on full dataset...")
        # Find the config_index for the best params
        config_index = param_grid["prophet"].index(best["params"])
        best_model, prophet_df, _ = train_prophet_model(y, dates, config_index)
        model_path = os.path.join(models_dir, "best_model_prophet.pkl")
        joblib.dump(best_model, model_path)
        # Also save the training data format for future predictions
        joblib.dump(prophet_df, os.path.join(models_dir, "prophet_training_data.pkl"))
        logger.info(f"Prophet model saved → {model_path}")

    elif best["model"] == "arima":
        logger.info("Fitting best ARIMA model on full dataset...")
        # Find the config_index for the best params
        config_index = param_grid["arima"].index(best["params"])
        best_model, _ = train_arima_model(y, config_index)
        model_path = os.path.join(models_dir, "best_model_arima.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"ARIMA model saved → {model_path}")

    else:
        logger.info(f"Fitting best {best['model']} model on full dataset...")
        best_model = get_model(best["model"], **best["params"])
        best_model.fit(X.reshape(X.shape[0], -1), y)
        model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Model saved → {model_path}")

    config_path = os.path.join(models_dir, "best_config.pkl")
    joblib.dump(best, config_path)
    logger.info(f"Config saved → {config_path}")

    return best, results


# ── Direct execution (standalone mode, independent from Streamlit/FastAPI) ────
if __name__ == "__main__":
    logger.info("Starting standalone training...")
    X, X_lstm, y, scaler = prepare_data("../data/timeseries.csv")
    best, results = run_training(X, X_lstm, y)
    logger.info(f"Training complete. Best: {best}")
