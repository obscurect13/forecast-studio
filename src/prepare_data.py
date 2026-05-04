import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from logger_config import setup_logger

logger = setup_logger(__name__)

WINDOW = 20  # must match WINDOW in models.py


def prepare_data(filepath: str, target_col: str = None, date_col: str = None):
    """
    Load a CSV time series and return windowed arrays ready for training.

    Parameters
    ----------
    filepath   : path to the CSV file
    target_col : name of the numeric column to forecast.
                 If None, the first numeric column is used automatically.
    date_col   : name of the date/datetime column to use as index.
                 If None, auto-detected or ignored if no date column exists.

    Returns
    -------
    X       : (n_samples, WINDOW)    — flat windows for RF / XGB / Linear
    X_lstm  : (n_samples, WINDOW, 1) — reshaped windows for LSTM
    y       : (n_samples,)           — targets (scaled)
    scaler  : fitted MinMaxScaler
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"CSV loaded — {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    # ── Resolve date column ───────────────────────────────────────────────────
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format="mixed")
        df = df.set_index(date_col).sort_index()
        logger.info(f"Using specified date column: '{date_col}'")
    else:
        date_candidates = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ("date", "time", "timestamp", "period"))
        ]
        if date_candidates:
            df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]], dayfirst=True, format="mixed")
            df = df.set_index(date_candidates[0]).sort_index()
            logger.info(f"Auto-detected date column: '{date_candidates[0]}'")
        else:
            logger.info("No date column found — using default integer index")

    # ── Resolve target column ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric column found in the CSV.")

    if target_col and target_col in df.columns:
        series = df[target_col].dropna().values.astype(float)
        logger.info(f"Using specified target column: '{target_col}'")
    else:
        chosen = numeric_cols[0]
        series = df[chosen].dropna().values.astype(float)
        logger.info(f"No target column specified — using first numeric column: '{chosen}'")

    if len(series) < WINDOW + 1:
        raise ValueError(
            f"Not enough data: need at least {WINDOW + 1} points, got {len(series)}."
        )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    logger.info(f"Series scaled | min={series.min():.4f} | max={series.max():.4f}")

    # ── Sliding windows ───────────────────────────────────────────────────────
    X, y = [], []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i - WINDOW:i])
        y.append(scaled[i])

    X      = np.array(X)
    y      = np.array(y)
    X_lstm = X.reshape(X.shape[0], X.shape[1], 1)

    logger.info(f"Windowing complete | {len(series)} points → {len(y)} samples (WINDOW={WINDOW})")
    return X, X_lstm, y, scaler
