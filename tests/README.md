# Test Suite for Forecast Project

This directory contains comprehensive tests for the forecast-project CI/CD pipeline.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and pytest configuration
├── fixtures/                   # Test data and fixtures
│   └── sample_timeseries.csv  # Sample time series data
├── unit/                       # Unit tests
│   ├── test_prepare_data.py   # Data preparation tests
│   ├── test_models.py         # Model creation and prediction tests
│   └── test_logger_config.py  # Logger configuration tests
└── integration/                # Integration tests
    └── test_api.py            # FastAPI endpoint tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **test_prepare_data.py**: Tests for data loading, scaling, and windowing
- **test_models.py**: Tests for model creation, configuration, and prediction
- **test_logger_config.py**: Tests for logging configuration

### Integration Tests (`tests/integration/`)
- **test_api.py**: Tests for FastAPI endpoints and complete workflows

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest tests/unit/ -v
```

### Run Only Integration Tests
```bash
pytest tests/integration/ -v
```

### Run Specific Test File
```bash
pytest tests/unit/test_models.py -v
```

### Run Specific Test Function
```bash
pytest tests/unit/test_models.py::TestModelCreation::test_get_model_linear -v
```

### Run with Coverage Report
```bash
pytest --cov=src --cov=api --cov-report=html
```

### Run by Markers
```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only API tests
pytest -m api -v

# Run only ML model tests
pytest -m ml -v

# Skip slow tests
pytest -m "not slow" -v
```

## Test Configuration

### pytest.ini
The `pytest.ini` file in the project root contains:
- Test paths and patterns
- Coverage configuration
- Custom markers
- Warning settings

### Requirements
Install test dependencies:
```bash
pip install -r requirements-test.txt
```

## CI/CD Integration

The tests are automatically run in GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

1. **Test Job**: Runs unit and integration tests across multiple Python versions
2. **Build Job**: Builds Docker image and runs basic smoke tests
3. **Security Scan Job**: Runs security scans using Bandit
4. **Deploy Job**: Deploys to production (only on main branch)

## Test Fixtures

Available fixtures in `conftest.py`:
- `sample_time_series_data`: Generates synthetic time series data
- `sample_windowed_data`: Creates windowed data for ML models
- `temp_csv_file`: Creates temporary CSV files
- `mock_models_dir`: Creates temporary models directory
- `sample_model_config`: Sample model configuration
- `sample_training_results`: Sample training results

## Writing New Tests

### Unit Test Example
```python
import pytest
from mymodule import my_function

@pytest.mark.unit
class TestMyFunction:
    def test_basic_case(self):
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Integration Test Example
```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoint:
    def test_endpoint(self):
        client = TestClient(app)
        response = client.get("/endpoint")
        assert response.status_code == 200
```

## Test Data

Sample test data is available in `tests/fixtures/`:
- `sample_timeseries.csv`: 40 rows of time series data for basic testing

## Coverage Goals

- **Overall Coverage**: Target > 80%
- **Critical Paths**: Aim for > 90% coverage
- **API Endpoints**: Ensure all endpoints are tested

## Troubleshooting

### Import Errors
If you encounter import errors, ensure the `src` directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Docker Tests
For Docker-related tests, ensure Docker is running:
```bash
docker ps
```

### Slow Tests
Some tests are marked as `@pytest.mark.slow`. Skip them with:
```bash
pytest -m "not slow"
```

## Continuous Improvement

To improve the test suite:
1. Add tests for new features
2. Increase coverage for critical paths
3. Add more edge case tests
4. Improve test data quality
5. Add performance benchmarks for critical operations