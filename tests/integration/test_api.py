"""
Integration tests for the FastAPI application.
"""
import pytest
import tempfile
import os
import sys
from unittest.mock import patch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing (41 rows — above WINDOW=20 minimum)."""
    csv_content = """date,value
01-01-2020,100.0
02-01-2020,102.5
03-01-2020,101.8
04-01-2020,103.2
05-01-2020,104.1
06-01-2020,103.8
07-01-2020,105.5
08-01-2020,106.2
09-01-2020,105.9
10-01-2020,107.1
11-01-2020,108.3
12-01-2020,107.8
13-01-2020,109.2
14-01-2020,110.5
15-01-2020,109.8
16-01-2020,111.2
17-01-2020,112.5
18-01-2020,111.8
19-01-2020,113.2
20-01-2020,114.5
21-01-2020,113.8
22-01-2020,115.2
23-01-2020,116.5
24-01-2020,115.8
25-01-2020,117.2
26-01-2020,118.5
27-01-2020,117.8
28-01-2020,119.2
29-01-2020,120.5
30-01-2020,119.8
31-01-2020,121.2
01-02-2020,122.5
02-02-2020,121.8
03-02-2020,123.2
04-02-2020,124.5
05-02-2020,123.8
06-02-2020,125.2
07-02-2020,126.5
08-02-2020,125.8
09-02-2020,127.2
10-02-2020,128.5
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_compare_models_endpoint(self, client, sample_csv_file):
        """Test the compare-models endpoint."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/compare-models",
                files={"file": ("test.csv", f, "text/csv")},
                params={"target_col": "value"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "started"

    def test_compare_models_with_target_col(self, client, sample_csv_file):
        """Test compare-models with explicit target column."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/compare-models",
                files={"file": ("test.csv", f, "text/csv")},
                params={"target_col": "value"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_compare_models_without_target_col(self, client, sample_csv_file):
        """Test compare-models without target column (auto-detect)."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/compare-models",
                files={"file": ("test.csv", f, "text/csv")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_job_status_endpoint(self, client, sample_csv_file):
        """Test the job-status endpoint.
        Fix: use sample_csv_file (41 rows) instead of a 2-row inline CSV
        that was below the WINDOW=20 minimum and caused a 400 error.
        """
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/compare-models",
                files={"file": ("test.csv", f, "text/csv")},
                params={"target_col": "value"}
            )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Check job status
        response = client.get(f"/job-status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["pending", "training", "completed", "failed"]

    def test_job_status_not_found(self, client):
        """Test job-status with non-existent job ID."""
        response = client.get("/job-status/non-existent-job-id")
        assert response.status_code == 404

    def test_predict_best_without_training(self, client, sample_csv_file, tmp_path):
        """Test predict-best endpoint without prior training (should fail).
        Fix: patch MODELS_DIR to an empty tmp_path so the test is not affected
        by any best_config.pkl already present in the real models/ folder.
        """
        with patch("api.main.MODELS_DIR", str(tmp_path)):
            with open(sample_csv_file, 'rb') as f:
                response = client.post(
                    "/predict-best",
                    files={"file": ("test.csv", f, "text/csv")},
                    params={"target_col": "value", "n_steps": 10}
                )

        assert response.status_code == 404

    def test_compare_models_invalid_file(self, client):
        """Test compare-models with invalid file.
        Fix: /compare-models now always returns 200 immediately (async job system).
        The validation error happens in the background task — check job status for 'failed'.
        """
        invalid_content = b"not a valid csv file"
        response = client.post(
            "/compare-models",
            files={"file": ("test.txt", invalid_content, "text/plain")}
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll until job reaches a terminal state
        import time
        for _ in range(15):
            status_resp = client.get(f"/job-status/{job_id}")
            job_data = status_resp.json()
            if job_data["status"] in ["completed", "failed"]:
                break
            time.sleep(1)

        assert job_data["status"] == "failed"

    def test_compare_models_insufficient_data(self, client):
        """Test compare-models with insufficient data.
        Fix: /compare-models now always returns 200 immediately (async job system).
        The validation error happens in the background task — check job status for 'failed'.
        """
        insufficient_csv = b"date,value\n01-01-2020,100.0\n02-01-2020,102.5\n"
        response = client.post(
            "/compare-models",
            files={"file": ("test.csv", insufficient_csv, "text/csv")}
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll until job reaches a terminal state
        import time
        for _ in range(15):
            status_resp = client.get(f"/job-status/{job_id}")
            job_data = status_resp.json()
            if job_data["status"] in ["completed", "failed"]:
                break
            time.sleep(1)

        assert job_data["status"] == "failed"

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set.
        Fix: CORS middleware only injects headers when the request includes
        an Origin header — a plain GET without it won't trigger CORS.
        """
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:8501"}
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestAPIWorkflow:
    """Test suite for complete API workflow."""

    def test_complete_training_workflow(self, client, sample_csv_file):
        """Test complete workflow: submit job, check status, get results."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/compare-models",
                files={"file": ("test.csv", f, "text/csv")},
                params={"target_col": "value"}
            )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        response = client.get(f"/job-status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_multiple_jobs(self, client, sample_csv_file):
        """Test handling multiple concurrent jobs."""
        job_ids = []

        for i in range(3):
            with open(sample_csv_file, 'rb') as f:
                response = client.post(
                    "/compare-models",
                    files={"file": (f"test{i}.csv", f, "text/csv")},
                    params={"target_col": "value"}
                )
            job_ids.append(response.json()["job_id"])

        assert len(job_ids) == 3
        assert len(set(job_ids)) == 3  # All job IDs should be unique

        for job_id in job_ids:
            response = client.get(f"/job-status/{job_id}")
            assert response.status_code == 200
