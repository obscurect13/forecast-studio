"""
Shared fixtures and configuration for pytest tests.
"""
import pytest
import numpy as np
import tempfile
import os
import sys

# Add src directory to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_points = 100
    t = np.linspace(0, 10, n_points)
    # Create a time series with trend and seasonality
    trend = 2 * t
    seasonality = 5 * np.sin(2 * np.pi * t)
    noise = np.random.normal(0, 0.5, n_points)
    data = trend + seasonality + noise + 100
    return data


@pytest.fixture
def sample_windowed_data():
    """Generate sample windowed data for ML models."""
    np.random.seed(42)
    n_samples = 50
    window_size = 20
    X = np.random.randn(n_samples, window_size)
    y = np.random.randn(n_samples)
    X_lstm = X.reshape(n_samples, window_size, 1)
    return X, X_lstm, y


@pytest.fixture
def temp_csv_file(sample_time_series_data):
    """Create a temporary CSV file with sample data."""
    import pandas as pd

    dates = pd.date_range(start='2020-01-01', periods=len(sample_time_series_data))
    df = pd.DataFrame({
        'date': dates.strftime('%d-%m-%Y'),
        'value': sample_time_series_data
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def mock_models_dir(tmp_path):
    """Create a temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        "model": "linear",
        "params": {"fit_intercept": True},
        "mse": 0.087,
        "r2": 0.91
    }


@pytest.fixture
def sample_training_results():
    """Sample training results for testing."""
    return [
        {
            "model": "linear",
            "params": {"fit_intercept": True},
            "mse": 0.087,
            "mae": 0.234,
            "r2": 0.91
        },
        {
            "model": "rf",
            "params": {"n_estimators": 10, "max_depth": 5},
            "mse": 0.092,
            "mae": 0.245,
            "r2": 0.90
        },
        {
            "model": "xgb",
            "params": {"n_estimators": 10, "max_depth": 3},
            "mse": 0.089,
            "mae": 0.238,
            "r2": 0.91
        }
    ]


# Pytest hooks for test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "api: API tests"
    )
    config.addinivalue_line(
        "markers", "ml: ML model tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add markers based on function names
        if "api" in str(item.fspath) or "api" in item.name.lower():
            item.add_marker(pytest.mark.api)
        if "model" in item.name.lower() or "prediction" in item.name.lower():
            item.add_marker(pytest.mark.ml)