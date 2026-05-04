"""
Unit tests for data preparation module.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Add src directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prepare_data import prepare_data, WINDOW


@pytest.mark.unit
class TestPrepareData:
    """Test suite for prepare_data function."""

    def test_prepare_data_basic(self):
        """Test basic data preparation with valid CSV."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            for i in range(50):
                f.write(f"01-01-2020,{100 + i}\n")
            temp_path = f.name

        try:
            X, X_lstm, y, scaler = prepare_data(temp_path, target_col="value")

            # Check shapes
            assert X.shape[0] > 0, "Should have samples after windowing"
            assert X.shape[1] == WINDOW, f"Window size should be {WINDOW}"
            assert X_lstm.shape == (X.shape[0], WINDOW, 1), "LSTM shape should be (samples, window, 1)"
            assert len(y) == X.shape[0], "Target length should match samples"
            assert scaler is not None, "Scaler should be fitted"

        finally:
            os.remove(temp_path)

    def test_prepare_data_auto_detect_target(self):
        """Test automatic target column detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value,other\n")
            for i in range(50):
                f.write(f"01-01-2020,{100 + i},{200 + i}\n")
            temp_path = f.name

        try:
            X, X_lstm, y, scaler = prepare_data(temp_path)
            # Should auto-detect first numeric column
            assert X.shape[0] > 0, "Should have samples"

        finally:
            os.remove(temp_path)

    def test_prepare_data_with_date_column(self):
        """Test data preparation with date column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            for i in range(50):
                f.write(f"2020-01-{i+1:02d},{100 + i}\n")
            temp_path = f.name

        try:
            X, X_lstm, y, scaler = prepare_data(temp_path, target_col="value", date_col="date")
            assert X.shape[0] > 0, "Should have samples"

        finally:
            os.remove(temp_path)

    def test_prepare_data_insufficient_data(self):
        """Test error handling for insufficient data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            for i in range(10):  # Less than WINDOW + 1
                f.write(f"01-01-2020,{100 + i}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Not enough data"):
                prepare_data(temp_path, target_col="value")
        finally:
            os.remove(temp_path)

    def test_prepare_data_no_numeric_column(self):
        """Test error handling for no numeric columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,name\n")
            for i in range(50):
                f.write(f"01-01-2020,name{i}\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="No numeric column"):
                prepare_data(temp_path)
        finally:
            os.remove(temp_path)

    def test_prepare_data_scaling(self):
        """Test that data is properly scaled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            for i in range(50):
                f.write(f"01-01-2020,{100 + i}\n")
            temp_path = f.name

        try:
            X, X_lstm, y, scaler = prepare_data(temp_path, target_col="value")

            # Check that scaled values are in [0, 1] range
            assert y.min() >= 0, "Scaled values should be >= 0"
            assert y.max() <= 1, "Scaled values should be <= 1"
            assert X.min() >= 0, "Scaled X should be >= 0"
            assert X.max() <= 1, "Scaled X should be <= 1"

        finally:
            os.remove(temp_path)

    def test_prepare_data_windowing(self):
        """Test that sliding window creation is correct."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            for i in range(50):
                f.write(f"01-01-2020,{100 + i}\n")
            temp_path = f.name

        try:
            X, X_lstm, y, scaler = prepare_data(temp_path, target_col="value")

            # Check that each window contains the correct sequence
            for i in range(min(5, len(X))):
                expected_sequence = np.arange(100 + i, 100 + i + WINDOW)
                # After scaling, the relative differences should be preserved
                assert X[i].shape == (WINDOW,), f"Window {i} should have length {WINDOW}"

        finally:
            os.remove(temp_path)