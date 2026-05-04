import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a consistent format across all project modules.
    Can be imported and called in any file with: logger = setup_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Avoid adding duplicate handlers on reload

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
