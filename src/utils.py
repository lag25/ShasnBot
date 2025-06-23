import logging
import time
from contextlib import contextmanager
import os

# utils.py

import logging
import os

def setup_logger(name="RAGDemo", level=logging.INFO, log_to_file=False, file_path="logs/app.log",):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Prevent duplicate handlers
    if not logger.handlers:
        # Console handler (safe, no Unicode)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler (UTF-8 safe)
        if log_to_file:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


@contextmanager
def timing(label="Operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"⏱️ {label} took {round(end - start, 2)} seconds")
