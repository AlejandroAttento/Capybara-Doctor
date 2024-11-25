import os
import logging
from datetime import datetime, timezone

script_name = os.path.basename(__file__).replace(".py", "")

LOG_FILE = f"{script_name}_{datetime.now(timezone.utc).strftime('Y%m%d%YT%H%M%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")
log_file_path = os.path.join(log_path, LOG_FILE)

os.makedirs(log_path, exist_ok=True)

logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Logger initialized")