import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Prevent duplicate handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(file_handler)
    return logger
