import logging
import re
def setup_logging(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger

def validate_3gpp_ref(ref: str) -> bool:
    ts_pattern = r'^TS\s+(2[1-9]|3[0-8])\.\d{3}(?:-\d+)?$'
    tr_pattern = r'^TR\s+(2[1-9]|3[0-8])\.\d{3}(?:-\d+)?$'
    ref = ref.strip()
    return bool(re.match(ts_pattern, ref) or re.match(tr_pattern, ref))
