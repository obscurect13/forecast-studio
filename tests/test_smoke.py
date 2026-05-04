"""
Smoke tests to verify test infrastructure is working correctly.
"""
import pytest
import sys
import os


@pytest.mark.unit
class TestSmokeTests:
    """Basic smoke tests to verify test infrastructure."""

    def test_python_imports(self):
        """Test that basic Python imports work."""
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        assert np is not None
        assert pd is not None
        assert MinMaxScaler is not None

    def test_project_structure(self):
        """Test that project structure is correct."""
        # Check that key directories exist
        assert os.path.exists("src"), "src directory should exist"
        assert os.path.exists("api"), "api directory should exist"
        assert os.path.exists("app"), "app directory should exist"
        assert os.path.exists("tests"), "tests directory should exist"

    def test_src_modules_importable(self):
        """Test that src modules can be imported."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        from logger_config import setup_logger
        from models import get_model, WINDOW

        assert setup_logger is not None
        assert get_model is not None
        assert WINDOW == 20

    def test_test_fixtures_exist(self):
        """Test that test fixtures directory exists."""
        fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
        assert os.path.exists(fixtures_dir), "fixtures directory should exist"

    def test_sample_data_exists(self):
        """Test that sample test data exists."""
        sample_csv = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_timeseries.csv')
        assert os.path.exists(sample_csv), "sample_timeseries.csv should exist"

    def test_basic_model_creation(self):
        """Test that we can create a basic model."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        from models import get_model

        model = get_model("linear")
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_logger_creation(self):
        """Test that we can create a logger."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        from logger_config import setup_logger

        logger = setup_logger("smoke_test")
        assert logger is not None
        assert logger.name == "smoke_test"

    def test_numpy_operations(self):
        """Test basic numpy operations work."""
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.sum() == 15
        assert arr.mean() == 3.0

    def test_pandas_operations(self):
        """Test basic pandas operations work."""
        import pandas as pd

        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        assert len(df) == 3
        assert list(df.columns) == ['a', 'b']

    def test_pytest_configuration(self):
        """Test that pytest is properly configured."""
        # This test will run if pytest is properly configured
        assert True


@pytest.mark.integration
class TestSmokeIntegration:
    """Basic smoke tests for integration test infrastructure."""

    def test_fastapi_importable(self):
        """Test that FastAPI can be imported."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        assert FastAPI is not None
        assert TestClient is not None

    def test_test_client_creation(self):
        """Test that we can create a test client."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        try:
            from api.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            assert client is not None

            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        except ImportError:
            pytest.skip("API module not fully importable in test environment")