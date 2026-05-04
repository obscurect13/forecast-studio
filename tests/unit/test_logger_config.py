"""
Unit tests for logger configuration.
"""
import pytest
import logging
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logger_config import setup_logger


@pytest.mark.unit
class TestLoggerConfig:
    """Test suite for logger configuration."""

    def test_setup_logger_basic(self):
        """Test basic logger creation."""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO  # Default level

    def test_setup_logger_custom_level(self):
        """Test logger creation with custom level."""
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert logger is not None
        assert logger.name == "test_logger_debug"
        assert logger.level == logging.DEBUG

    def test_setup_logger_handlers(self):
        """Test that logger has appropriate handlers."""
        logger = setup_logger("test_logger_handlers")
        assert logger is not None
        # Should have at least a stream handler
        assert len(logger.handlers) > 0

    def test_setup_logger_formatter(self):
        """Test that logger has proper formatter."""
        logger = setup_logger("test_logger_formatter")
        assert logger is not None

        # Check if any handler has a formatter
        has_formatter = any(handler.formatter is not None for handler in logger.handlers)
        assert has_formatter, "Logger should have a formatter"

    def test_logger_info_message(self):
        """Test that logger can log info messages."""
        logger = setup_logger("test_logger_info")
        # This should not raise an exception
        logger.info("Test info message")

    def test_logger_debug_message(self):
        """Test that logger can log debug messages."""
        logger = setup_logger("test_logger_debug_msg", level=logging.DEBUG)
        # This should not raise an exception
        logger.debug("Test debug message")

    def test_logger_warning_message(self):
        """Test that logger can log warning messages."""
        logger = setup_logger("test_logger_warning")
        # This should not raise an exception
        logger.warning("Test warning message")

    def test_logger_error_message(self):
        """Test that logger can log error messages."""
        logger = setup_logger("test_logger_error")
        # This should not raise an exception
        logger.error("Test error message")

    def test_multiple_loggers(self):
        """Test creating multiple loggers with different names."""
        logger1 = setup_logger("logger1")
        logger2 = setup_logger("logger2")

        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1 != logger2