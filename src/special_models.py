"""
Special model handling for Prophet and ARIMA models.
These models require custom data processing and training pipelines.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from param_grid_config import param_grid
from logger_config import setup_logger
import warnings

warnings.filterwarnings('ignore')
logger = setup_logger(__name__)


def prepare_prophet_data(y, dates=None):
    """
    Prepare data for Prophet model.

    Args:
        y: Target values (numpy array)
        dates: Optional datetime index (if None, creates sequential dates)

    Returns:
        DataFrame with 'ds' and 'y' columns for Prophet
    """
    if dates is None:
        # Create sequential dates starting from today
        start_date = pd.Timestamp.now().normalize()
        dates = pd.date_range(start=start_date, periods=len(y), freq='D')

    df = pd.DataFrame({
        'ds': dates,
        'y': y
    })
    return df


def check_stationarity(y, significance_level=0.05):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.

    Args:
        y: Time series values
        significance_level: Threshold for p-value (default 0.05)

    Returns:
        is_stationary: Boolean indicating if series is stationary
        p_value: P-value from ADF test
        d: Suggested differencing order (0 if stationary, 1 if not)
    """
    try:
        result = adfuller(y)
        p_value = result[1]
        is_stationary = p_value < significance_level

        # Suggest differencing order
        d = 0 if is_stationary else 1

        logger.info(f"Stationarity test: p-value={p_value:.4f}, stationary={is_stationary}, suggested_d={d}")

        return is_stationary, p_value, d
    except Exception as e:
        logger.warning(f"Stationarity test failed: {e}. Assuming non-stationary.")
        return False, 1.0, 1


def difference_series(y, d=1):
    """
    Apply differencing to make series stationary.

    Args:
        y: Time series values
        d: Order of differencing

    Returns:
        differenced: Differenced series
        last_values: Values needed for inverse differencing
    """
    differenced = y.copy()
    last_values = []

    for i in range(d):
        last_values.append(differenced[-1])
        differenced = np.diff(differenced)

    return differenced, last_values


def inverse_difference(forecast, last_values):
    """
    Inverse differencing to get back to original scale.

    Args:
        forecast: Forecasted differenced values
        last_values: Last values from differencing (in reverse order)

    Returns:
        restored: Values in original scale
    """
    restored = forecast.copy()

    for last_val in reversed(last_values):
        restored = np.concatenate([[last_val], restored])
        restored = np.cumsum(restored)

    return restored


def train_prophet_model(y, dates=None, config_index=0):
    """
    Train Prophet model on time series data using param_grid config.

    Args:
        y: Target values
        dates: Optional datetime index
        config_index: Index of Prophet config to use from param_grid

    Returns:
        model: Trained Prophet model
        prophet_df: DataFrame used for training
        params: Parameters used
    """
    prophet_df = prepare_prophet_data(y, dates)

    # Get parameters from param_grid
    if config_index >= len(param_grid["prophet"]):
        raise ValueError(f"config_index {config_index} out of range for Prophet configs")

    params = param_grid["prophet"][config_index]

    # Extract Prophet-specific parameters
    prophet_params = {
        'growth': params.get('growth', 'linear'),
        'changepoint_prior_scale': params.get('changepoint_prior_scale', 0.05),
        'seasonality_prior_scale': params.get('seasonality_prior_scale', 10),
        'yearly_seasonality': params.get('yearly_seasonality', 'auto'),
        'weekly_seasonality': params.get('weekly_seasonality', 'auto'),
        'daily_seasonality': params.get('daily_seasonality', 'auto'),
    }

    # Remove None values
    prophet_params = {k: v for k, v in prophet_params.items() if v is not None}

    logger.info(f"Training Prophet with params: {prophet_params}")

    # Logistic growth requires a cap column
    if prophet_params.get('growth') == 'logistic':
        prophet_df['cap'] = prophet_df['y'].max() * 1.2  # 20% above max

    model = Prophet(**prophet_params)
    model.fit(prophet_df)

    return model, prophet_df, params


def predict_prophet(model, prophet_df, n_steps=1):
    """
    Make predictions with Prophet model.

    Args:
        model: Trained Prophet model
        prophet_df: Training DataFrame
        n_steps: Number of steps to forecast

    Returns:
        predictions: In-sample predictions
        future_forecast: Future forecast values
    """
    # Make in-sample predictions
    predictions = model.predict(prophet_df)['yhat'].values

    # Make future predictions
    future_dates = model.make_future_dataframe(periods=n_steps, include_history=False)
    future_forecast = model.predict(future_dates)['yhat'].values

    return predictions, future_forecast


def train_arima_model(y, config_index=0):
    """
    Train ARIMA model on time series data using param_grid config.

    Args:
        y: Target values
        config_index: Index of ARIMA config to use from param_grid

    Returns:
        model: Trained ARIMA model
        params: Parameters used
    """
    # Get parameters from param_grid
    if config_index >= len(param_grid["arima"]):
        raise ValueError(f"config_index {config_index} out of range for ARIMA configs")

    params = param_grid["arima"][config_index]

    # Get parameters with defaults
    p = params.get('p', 1)
    d = params.get('d', 1)
    q = params.get('q', 1)

    # Check stationarity and adjust d if needed
    is_stationary, p_value, suggested_d = check_stationarity(y)

    # Use suggested d if not explicitly provided or if current d won't work
    if d == 1 and not is_stationary:
        d = suggested_d
        logger.info(f"Using suggested differencing order d={d} based on stationarity test")

    # Apply differencing if needed
    if d > 0:
        y_diff, last_values = difference_series(y, d)
        logger.info(f"Applied {d}-order differencing")
    else:
        y_diff = y
        last_values = []

    logger.info(f"Training ARIMA({p},{d},{q}) on {len(y_diff)} samples")

    # Train ARIMA model
    try:
        model = ARIMA(y_diff, order=(p, d, q))
        fitted_model = model.fit()

        # Store metadata for inverse operations
        fitted_model.d_used = d
        fitted_model.last_values = last_values
        fitted_model.original_length = len(y)
        fitted_model.params_used = params

        return fitted_model, params

    except Exception as e:
        logger.error(f"ARIMA training failed: {e}")
        # Try with simpler parameters
        logger.info("Retrying with ARIMA(1,1,1)")
        model = ARIMA(y_diff, order=(1, 1, 1))
        fitted_model = model.fit()
        fitted_model.d_used = 1
        fitted_model.last_values = last_values if d == 1 else []
        fitted_model.original_length = len(y)
        fitted_model.params_used = {'p': 1, 'd': 1, 'q': 1}
        return fitted_model, {'p': 1, 'd': 1, 'q': 1}


def predict_arima(model, n_steps=1):
    """
    Make predictions with ARIMA model.

    Args:
        model: Trained ARIMA model
        n_steps: Number of steps to forecast

    Returns:
        predictions: In-sample predictions
        future_forecast: Future forecast values (in original scale)
    """
    # Get future forecast (used for both CV test preds and future horizon)
    forecast_result = model.forecast(steps=n_steps)
    future_forecast = (
        forecast_result if isinstance(forecast_result, np.ndarray)
        else forecast_result.values
    )

    # Apply inverse differencing if needed
    if hasattr(model, 'd_used') and model.d_used > 0 and len(model.last_values) > 0:
        future_forecast = inverse_difference(future_forecast, model.last_values)

    # In-sample predictions (fitted values, already in differenced scale)
    predictions = model.fittedvalues

    return predictions, future_forecast


def prophet_rolling_cv(y, dates=None, n_splits=5, config_index=0):
    """
    Perform rolling cross-validation for Prophet model using param_grid config.

    Args:
        y: Target values
        dates: Optional datetime index
        n_splits: Number of CV splits
        config_index: Index of Prophet config to use from param_grid

    Returns:
        metrics: Dictionary with MSE, MAE, R2 scores
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Get parameters from param_grid
    if config_index >= len(param_grid["prophet"]):
        raise ValueError(f"config_index {config_index} out of range for Prophet configs")

    params = param_grid["prophet"][config_index]

    min_train_size = max(20, len(y) // (n_splits + 1))
    metrics_list = []

    for i in range(n_splits):
        # Calculate split point
        split_point = min_train_size + i * ((len(y) - min_train_size) // n_splits)

        if split_point >= len(y):
            break

        # Split data
        y_train = y[:split_point]
        y_test = y[split_point:split_point + 20]  # Predict next 20 points

        if len(y_test) == 0:
            break

        # Get corresponding dates
        if dates is not None:
            dates_train = dates[:split_point]
            dates_test = dates[split_point:split_point + 20]
        else:
            dates_train = None
            dates_test = None

        try:
            # Train model
            model, prophet_df, _ = train_prophet_model(y_train, dates_train, config_index)

            # Make predictions for test period
            if dates_test is not None:
                test_dates = pd.date_range(start=dates_test[0], periods=len(y_test), freq='D')
                test_df = pd.DataFrame({'ds': test_dates})
            else:
                start_date = pd.Timestamp.now().normalize()
                test_dates = pd.date_range(start=start_date, periods=len(y_test), freq='D')
                test_df = pd.DataFrame({'ds': test_dates})

            # Required for logistic growth
            if params.get('growth') == 'logistic':
                test_df['cap'] = float(prophet_df['cap'].max()) if 'cap' in prophet_df.columns else 1.0

            forecast = model.predict(test_df)
            predictions = forecast['yhat'].values

            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            metrics_list.append({'mse': mse, 'mae': mae, 'r2': r2})

        except Exception as e:
            logger.warning(f"Prophet CV split {i+1} failed: {e}")
            continue

    if not metrics_list:
        raise RuntimeError("All Prophet CV splits failed")

    # Average metrics across splits
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'mae': np.mean([m['mae'] for m in metrics_list]),
        'r2': np.mean([m['r2'] for m in metrics_list])
    }

    return avg_metrics


def arima_rolling_cv(y, n_splits=5, config_index=0):
    """
    Perform rolling cross-validation for ARIMA model using param_grid config.

    Args:
        y: Target values
        n_splits: Number of CV splits
        config_index: Index of ARIMA config to use from param_grid

    Returns:
        metrics: Dictionary with MSE, MAE, R2 scores
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Get parameters from param_grid
    if config_index >= len(param_grid["arima"]):
        raise ValueError(f"config_index {config_index} out of range for ARIMA configs")

    params = param_grid["arima"][config_index]

    min_train_size = max(20, len(y) // (n_splits + 1))
    metrics_list = []

    for i in range(n_splits):
        # Calculate split point
        split_point = min_train_size + i * ((len(y) - min_train_size) // n_splits)

        if split_point >= len(y):
            break

        # Split data
        y_train = y[:split_point]
        y_test = y[split_point:split_point + 20]  # Predict next 20 points

        if len(y_test) == 0:
            break

        try:
            # Train model
            model, _ = train_arima_model(y_train, config_index)

            # Use future forecast (not in-sample fittedvalues) for fair CV comparison
            _, predictions = predict_arima(model, n_steps=len(y_test))

            # Align predictions with test data
            min_len = min(len(y_test), len(predictions))
            if min_len < 5:  # Need minimum samples for meaningful metrics
                continue

            y_test_aligned = y_test[:min_len]
            predictions_aligned = predictions[:min_len]

            # Calculate metrics
            mse = mean_squared_error(y_test_aligned, predictions_aligned)
            mae = mean_absolute_error(y_test_aligned, predictions_aligned)
            r2 = r2_score(y_test_aligned, predictions_aligned)

            metrics_list.append({'mse': mse, 'mae': mae, 'r2': r2})

        except Exception as e:
            logger.warning(f"ARIMA CV split {i+1} failed: {e}")
            continue

    if not metrics_list:
        raise RuntimeError("All ARIMA CV splits failed")

    # Average metrics across splits
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in metrics_list]),
        'mae': np.mean([m['mae'] for m in metrics_list]),
        'r2': np.mean([m['r2'] for m in metrics_list])
    }

    return avg_metrics