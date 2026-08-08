import logging
import re
from functools import wraps

def setup_logging(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger

def validate_3gpp_ref(ref: str) -> bool:
    return bool(re.match(r"^TS (2[1-9]|3[0-8])\.\d{3}$", ref))

def retry(max_attempts: int = 2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
