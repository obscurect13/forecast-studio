"""
Unit tests for model creation and configuration.
"""
import pytest
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    get_model,
    get_lgbm_model,
    get_catboost_model,
    get_svr_model,
    get_knn_model,
    get_prophet_model,
    get_arima_model,
    WINDOW
)


@pytest.mark.unit
class TestModelCreation:
    """Test suite for model creation functions."""

    def test_get_model_linear(self):
        """Test LinearRegression model creation."""
        model = get_model("linear", fit_intercept=True)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_random_forest(self):
        """Test RandomForest model creation."""
        model = get_model("rf", n_estimators=10, max_depth=5)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_xgboost(self):
        """Test XGBoost model creation."""
        model = get_model("xgb", n_estimators=10, max_depth=3)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_lightgbm(self):
        """Test LightGBM model creation."""
        model = get_model("lgbm", n_estimators=10, max_depth=3)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_catboost(self):
        """Test CatBoost model creation."""
        model = get_model("catboost", iterations=10, depth=3)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_svr(self):
        """Test SVR model creation."""
        model = get_model("svr", kernel='rbf', C=1.0)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_knn(self):
        """Test KNN model creation."""
        model = get_model("knn", n_neighbors=5)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_lstm(self):
        """Test LSTM model creation.
        Fix: Keras always includes the batch dimension in input_shape,
        so the correct expected shape is (None, WINDOW, 1) not (WINDOW, 1).
        """
        model = get_model("lstm", units=32, units2=16, dropout=0.1)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        # Keras input_shape includes the batch dimension as None
        assert model.input_shape == (None, WINDOW, 1)

    @pytest.mark.skip(
        reason="Prophet is disabled in Docker/Linux deployment due to Stan backend "
               "incompatibility with Python 3.12. Removed from param_grid_config.py "
               "for cloud deployment. Works locally via conda."
    )
    def test_get_model_prophet(self):
        """Test Prophet model creation."""
        pass

    @pytest.mark.skip(
        reason="statsmodels ARIMA requires endog data at instantiation — "
               "cannot be constructed without a time series. "
               "ARIMA is tested end-to-end in integration tests instead."
    )
    def test_get_model_arima(self):
        """Test ARIMA model creation."""
        pass

    def test_get_model_unknown(self):
        """Test error handling for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model")

    def test_get_model_filters_params(self):
        """Test that irrelevant parameters are filtered out."""
        model = get_model("linear", fit_intercept=True, irrelevant_param=123)
        assert model is not None

    def test_get_lgbm_model(self):
        """Test LightGBM model helper function."""
        model = get_lgbm_model(n_estimators=10, max_depth=3)
        assert model is not None
        assert hasattr(model, 'fit')

    def test_get_catboost_model(self):
        """Test CatBoost model helper function."""
        model = get_catboost_model(iterations=10, depth=3)
        assert model is not None
        assert hasattr(model, 'fit')

    def test_get_svr_model(self):
        """Test SVR model helper function."""
        model = get_svr_model(kernel='rbf', C=1.0)
        assert model is not None
        assert hasattr(model, 'fit')

    def test_get_knn_model(self):
        """Test KNN model helper function."""
        model = get_knn_model(n_neighbors=5)
        assert model is not None
        assert hasattr(model, 'fit')

    @pytest.mark.skip(
        reason="Prophet is disabled in Docker/Linux deployment due to Stan backend "
               "incompatibility with Python 3.12. Removed from param_grid_config.py "
               "for cloud deployment. Works locally via conda."
    )
    def test_get_prophet_model(self):
        """Test Prophet model helper function."""
        pass

    @pytest.mark.skip(
        reason="statsmodels ARIMA requires endog data at instantiation — "
               "cannot be constructed without a time series. "
               "ARIMA is tested end-to-end in integration tests instead."
    )
    def test_get_arima_model(self):
        """Test ARIMA model helper function."""
        pass


@pytest.mark.unit
@pytest.mark.ml
class TestModelPrediction:
    """Test suite for model prediction capabilities."""

    def test_linear_model_prediction(self):
        """Test that LinearRegression can make predictions."""
        model = get_model("linear", fit_intercept=True)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_rf_model_prediction(self):
        """Test that RandomForest can make predictions."""
        model = get_model("rf", n_estimators=10, max_depth=3)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_xgb_model_prediction(self):
        """Test that XGBoost can make predictions."""
        model = get_model("xgb", n_estimators=10, max_depth=3)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_lgbm_model_prediction(self):
        """Test that LightGBM can make predictions."""
        model = get_model("lgbm", n_estimators=10, max_depth=3)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_catboost_model_prediction(self):
        """Test that CatBoost can make predictions."""
        model = get_model("catboost", iterations=10, depth=3)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_svr_model_prediction(self):
        """Test that SVR can make predictions."""
        model = get_model("svr", kernel='rbf', C=1.0)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_knn_model_prediction(self):
        """Test that KNN can make predictions."""
        model = get_model("knn", n_neighbors=5)

        X_train = np.random.randn(50, WINDOW)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train)
        X_test = np.random.randn(5, WINDOW)
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_lstm_model_prediction(self):
        """Test that LSTM can make predictions."""
        model = get_model("lstm", units=16, units2=8, dropout=0.1)

        X_train = np.random.randn(50, WINDOW, 1)
        y_train = np.random.randn(50)

        model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)

        X_test = np.random.randn(5, WINDOW, 1)
        predictions = model.predict(X_test, verbose=0)

        assert predictions is not None
        assert len(predictions) == 5
