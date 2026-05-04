# Quick Start Guide for Testing

This guide will help you get started with running tests for the forecast-project.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker for integration tests

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd forecast-project
   ```

2. **Install dependencies**:
   ```bash
   # Install main dependencies
   pip install -r requirements.txt

   # Install test dependencies
   pip install -r requirements-test.txt
   ```

## Running Tests

### Quick Start
```bash
# Run all tests
pytest

# Or use the convenience script
./run_tests.sh
```

### Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only API tests
pytest -m api

# Run only ML model tests
pytest -m ml
```

### With Coverage

```bash
# Generate coverage report
pytest --cov=src --cov=api --cov-report=html

# Open the coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Verbose Output

```bash
# Detailed test output
pytest -v

# Very detailed output (shows print statements)
pytest -vv -s
```

### Run Specific Tests

```bash
# Run specific test file
pytest tests/unit/test_models.py

# Run specific test class
pytest tests/unit/test_models.py::TestModelCreation

# Run specific test function
pytest tests/unit/test_models.py::TestModelCreation::test_get_model_linear
```

## Test Scripts

### Using the Convenience Script

The `run_tests.sh` script provides an easy interface:

```bash
# Show help
./run_tests.sh --help

# Run unit tests with coverage
./run_tests.sh --unit --coverage

# Run API tests with verbose output
./run_tests.sh --api --verbose

# Run all tests with coverage
./run_tests.sh --coverage
```

## Understanding Test Results

### Success Output
```
tests/unit/test_models.py::TestModelCreation::test_get_model_linear PASSED
tests/unit/test_models.py::TestModelCreation::test_get_model_random_forest PASSED
...
======================== 42 passed in 2.34s =========================
```

### Failure Output
```
tests/unit/test_models.py::TestModelCreation::test_get_model_linear FAILED
...
======================== 1 failed, 41 passed in 2.45s =========================
```

### Coverage Output
```
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
src/models.py               154      8    95%   23-45, 67-89
src/prepare_data.py          45      2    96%   12-13
api/main.py                 201     15    93%   45-67, 123-145
-------------------------------------------------------
TOTAL                        400     25    94%
```

## CI/CD Pipeline

The tests are automatically run in GitHub Actions:

1. **On Push**: Tests run on every push to `main` or `develop` branches
2. **On Pull Request**: Tests run on every PR
3. **Manual**: You can trigger the workflow manually from GitHub Actions tab

### Workflow Stages

1. **Test**: Runs unit and integration tests across Python 3.9, 3.10, 3.11
2. **Build**: Builds Docker image and runs smoke tests
3. **Security Scan**: Runs security checks with Bandit
4. **Deploy**: Deploys to production (main branch only)

## Troubleshooting

### Import Errors
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run from project root
cd /path/to/forecast-project
pytest
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Docker Issues
```bash
# Check if Docker is running
docker ps

# Restart Docker daemon
sudo systemctl restart docker  # Linux
# Or restart Docker Desktop  # Windows/macOS
```

### Slow Tests
Some tests are marked as slow. Skip them:
```bash
pytest -m "not slow"
```

## Writing New Tests

### Basic Test Structure

```python
import pytest
from mymodule import my_function

@pytest.mark.unit
class TestMyFunction:
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge cases."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Using Fixtures

```python
def test_with_fixture(sample_time_series_data):
    """Test using shared fixture."""
    result = process_data(sample_time_series_data)
    assert result is not None
```

## Best Practices

1. **Write descriptive test names**: `test_get_model_linear` is better than `test_1`
2. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`
3. **Keep tests independent**: Each test should run in isolation
4. **Test edge cases**: Don't just test the happy path
5. **Use fixtures**: Avoid code duplication with shared fixtures
6. **Keep tests fast**: Avoid slow operations in unit tests

## Next Steps

- Explore the test suite in `tests/` directory
- Read the detailed test documentation in `tests/README.md`
- Check out the pytest configuration in `pytest.ini`
- Review the CI/CD workflow in `.github/workflows/ci-cd.yml`

## Getting Help

If you encounter issues:

1. Check the test output for error messages
2. Review the troubleshooting section above
3. Check the GitHub Actions logs for CI/CD issues
4. Open an issue with detailed error information

Happy testing! 🧪