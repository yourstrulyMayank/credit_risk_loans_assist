import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_NAME = "credit_risk_assist"  # Shared name across scripts

def setup_logger(log_dir="log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.DEBUG)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{LOG_NAME}_{today}.log")

    if not logger.handlers:
        handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding='utf-8',
            utc=False
        )
        handler.suffix = "%Y-%m-%d"

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
