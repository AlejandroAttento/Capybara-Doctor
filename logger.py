import os
import logging
from datetime import datetime, timezone

LOG_FILE = f"{datetime.now(timezone.utc).strftime('%m_%d_%YT%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")
log_file_path = os.path.join(log_path, LOG_FILE)

os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)