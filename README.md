# Forecast Studio

A modular time-series forecasting project that compares multiple ML and statistical models via rolling walk-forward cross-validation, selects the best one, and exposes it through a FastAPI backend and a Streamlit frontend.

---

## Project Structure

```
forecast-project/
│
├── api/
│   └── main.py                     # FastAPI backend (async job system)
│
├── app/
│   └── streamlit_app.py            # Streamlit frontend (2 tabs)
│
├── src/
│   ├── logger_config.py            # Shared logger — imported by all modules
│   ├── models.py                   # Model definitions (all 9 models)
│   ├── special_models.py           # Prophet & ARIMA training/prediction helpers
│   ├── param_grid_config.py        # Hyperparameter search space
│   ├── prepare_data.py             # CSV loading, scaling, sliding windows
│   ├── rolling_walk_forward_cv.py  # Walk-forward cross-validation
│   └── train_select_best_model.py  # Training loop + best model selection
│
├── data/
│   └── timeseries.csv              # Your input CSV file
│
├── models/                         # Saved models (auto-created after training)
│   ├── best_model.pkl              # Best sklearn/XGB/LGBM/CatBoost/SVR/KNN model
│   ├── best_model.h5               # Best LSTM model (if applicable)
│   ├── best_model_prophet.pkl      # Best Prophet model (if applicable)
│   ├── best_model_arima.pkl        # Best ARIMA model (if applicable)
│   └── best_config.pkl             # Best model name + params
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── start.sh
└── README.md
```

---

## Models

| Model | Type | Library | Notes |
|---|---|---|---|
| `linear` | ML | scikit-learn | `LinearRegression` |
| `rf` | ML | scikit-learn | `RandomForestRegressor` |
| `xgb` | ML | XGBoost | `XGBRegressor` |
| `lgbm` | ML | LightGBM | `LGBMRegressor` — fast gradient boosting |
| `catboost` | ML | CatBoost | `CatBoostRegressor` — strong out-of-the-box |
| `svr` | ML | scikit-learn | `SVR` — kernel-based regression |
| `knn` | ML | scikit-learn | `KNeighborsRegressor` — local pattern matching |
| `lstm` | Deep Learning | TensorFlow/Keras | Bidirectional LSTM with Dropout |
| `prophet` | Statistical | Prophet | Facebook's decomposable trend/seasonality model |
| `arima` | Statistical | statsmodels | Classic ARIMA with ADF stationarity check |

> **Note:** Prophet and ARIMA use custom training and prediction pipelines defined in `src/special_models.py`, separate from the sliding-window approach used by the ML models.

---

## How It Works

1. **`prepare_data.py`** loads the CSV, auto-detects the date and target columns, scales the series with `MinMaxScaler`, and builds sliding windows of size `WINDOW=20`
2. **`rolling_walk_forward_cv.py`** evaluates each model config with expanding-window CV (MSE, MAE, and R² averaged across folds)
3. **`train_select_best_model.py`** iterates over `param_grid`, finds the best config by MSE, retrains on the full dataset, and saves the model to `models/`
4. **`main.py`** exposes three endpoints — training runs in a background task to avoid HTTP timeouts, and the client polls for results
5. **`streamlit_app.py`** provides a two-tab UI on top of the API

```
Upload CSV
    └── prepare_data()
            └── rolling_cv()          ← all configs in param_grid
                    └── get_model()   ← dispatches to ML or special models
            └── best model selected (lowest MSE)
            └── saved to models/
```

---

## Metrics

| Metric | Description | Best value |
|---|---|---|
| MSE | Mean Squared Error — used to select the best model | → 0 |
| RMSE | Root MSE — same scale as the original series | → 0 |
| MAE | Mean Absolute Error — robust to outliers | → 0 |
| R² | Proportion of variance explained by the model | → 1 |

> **Note on R²** in time series: a high R² can be misleading on trended or seasonal series. Always cross-check with RMSE.

---

## Logging

All modules share a unified logger defined in `src/logger_config.py`.

**Log format:**
```
2026-04-29 01:12:43 | INFO     | train_select_best_model | [3/11] Testing rf | params={...}
2026-04-29 01:12:45 | INFO     | train_select_best_model | Best model: LINEAR | MSE=0.087 | R²=0.91
2026-04-29 01:12:45 | INFO     | main                    | Job a3f1bc... completed — best model: linear
```

**Log levels used:**

| Level | Where | What |
|---|---|---|
| `INFO` | All modules | Job lifecycle, model selection, file paths, data shape |
| `DEBUG` | `rolling_cv`, `models` | Per-fold scores, model instantiation details |
| `WARNING` | `train_select_best_model`, `special_models` | Skipped configs, stationarity test failures |
| `ERROR` | `main`, `models` | Job failures, unknown model names |

**`/health` and `/job-status` endpoints** are filtered from uvicorn access logs to avoid noise in the terminal.

**To enable DEBUG logs**, change the level in `logger_config.py`:
```python
def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/compare-models` | Start background training job, returns `job_id` immediately |
| `GET` | `/job-status/{job_id}` | Poll training job — status: `pending` / `training` / `completed` / `failed` |
| `POST` | `/predict-best` | Run inference with the saved best model |

Both `POST` endpoints accept:
- `file` — CSV file (multipart/form-data)
- `target_col` *(optional)* — name of the column to forecast

`/predict-best` also accepts:
- `n_steps` *(optional, default 20)* — number of future steps to forecast

### Why async jobs?

Training multiple models over rolling CV can take several minutes. A synchronous HTTP request would time out. The solution: `/compare-models` launches the training in a background task and returns a `job_id` immediately. Streamlit polls `/job-status/{job_id}` every few seconds until the job completes.

> **Production note:** Job state is stored in-memory. For multi-worker or persistent deployments, replace the `jobs` dict with Redis or a database.

---

## Installation

### Option 1 — Docker (recommended)

```bash
docker-compose up --build
```

- Streamlit → `http://localhost:8501`
- FastAPI docs → `http://localhost:8000/docs`

### Option 2 — Local

```bash
pip install -r requirements.txt
```

**Start the API:**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**Start Streamlit** (new terminal):
```bash
cd app
streamlit run streamlit_app.py
```

**Standalone training** (no API):
```bash
cd src
python train_select_best_model.py
```

---

## Deployment (AWS EC2)

### Prerequisites
- An AWS account
- A key pair (`.pem` file) created in the EC2 console

### 1. Launch an EC2 instance
- AMI: **Ubuntu 24.04 LTS**
- Instance type: **t3.micro** (free tier eligible)
- Storage: **20 GiB**
- Security group — open the following ports:

| Type | Port | Source |
|---|---|---|
| SSH | 22 | `0.0.0.0/0` |
| Custom TCP | 8000 | `0.0.0.0/0` |
| Custom TCP | 8501 | `0.0.0.0/0` |

- Paste the following in the **User data** field to auto-install Docker on boot:

```bash
#!/bin/bash
apt update -y
apt install -y docker.io docker-compose
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker
```

### 2. Upload the project

From your local machine:

```bash
scp -i "your-key.pem" -r ./forecast-project ubuntu@<your-ec2-public-ip>:~/
```

### 3. Build and run

SSH into the instance (or use EC2 Instance Connect in the AWS console), then:

```bash
cd ~/forecast-project
docker-compose up --build -d
```

The `-d` flag runs the containers in the background — the app stays alive even after closing the terminal.

### 4. Access the app

- Streamlit → `http://<your-ec2-public-ip>:8501`
- FastAPI docs → `http://<your-ec2-public-ip>:8000/docs`

> **Note:** The public IP changes every time you stop and restart the instance. To keep a fixed IP, allocate an **Elastic IP** from the EC2 console and attach it to your instance — it's free as long as the instance is running.

---

## Input CSV Format

The CSV must contain at least one numeric column. A date column is optional but recommended for proper time ordering.

```
date,value
19-02-2010,123.4
20-02-2010,125.1
...
```

Supported date formats: `DD-MM-YYYY`, `YYYY-MM-DD`, `MM/DD/YYYY`, ISO8601, and mixed formats.

If no `target_col` is specified, the first numeric column is used automatically.

Minimum required rows: **40** (default `WINDOW=20`).

---

## Configuration

### Change the window size
Edit `WINDOW` in `src/models.py` and `src/prepare_data.py` (must match).

### Add or modify hyperparameters
Edit `src/param_grid_config.py`:

```python
param_grid = {
    "rf": [
        {"n_estimators": 100, "max_depth": 10},
    ],
    "lgbm": [
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    ],
    "prophet": [
        {"growth": "linear", "changepoint_prior_scale": 0.05},
    ],
    "arima": [
        {"p": 2, "d": 1, "q": 2},
    ],
}
```

### Add a new model
1. Add its definition in `src/models.py` under `get_model()` and define a `_PARAMS` set for it
2. If it requires custom training/prediction logic (like Prophet or ARIMA), add handlers in `src/special_models.py`
3. Add its configs in `src/param_grid_config.py`
4. Handle model-specific prediction in `api/main.py` under the `/predict-best` endpoint if needed

---

## API Usage Example

```python
import requests
import time

# Step 1 — start training job
with open("data/timeseries.csv", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/compare-models",
        files={"file": f},
        params={"target_col": "value"}
    )
job_id = resp.json()["job_id"]

# Step 2 — poll until done
while True:
    status = requests.get(f"http://localhost:8000/job-status/{job_id}").json()
    if status["status"] == "completed":
        print(status["result"])
        break
    elif status["status"] == "failed":
        print("Error:", status["error"])
        break
    time.sleep(5)

# Step 3 — predict with best model (20 steps ahead by default)
with open("data/timeseries.csv", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict-best",
        files={"file": f},
        params={"target_col": "value", "n_steps": 30}
    )
result = resp.json()
print(result["model"])    # e.g. "lgbm"
print(result["metrics"])  # mse, mae, rmse, r2
print(result["future"])   # list of n_steps forecasted values
```
