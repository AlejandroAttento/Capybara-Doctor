import os
import logging
from datetime import datetime, timezone
import __main__  # Import to get the executed script's context

# Get the name of the original script being executed
script_name = os.path.basename(os.path.splitext(os.path.abspath(__main__.__file__))[0]) 

# Construct the log file name with timestamp
LOG_FILE = f"{script_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"

# Create logs directory if it doesn't exist
log_path = os.path.join(os.getcwd(), "logs")
log_file_path = os.path.join(log_path, LOG_FILE)
os.makedirs(log_path, exist_ok=True)

# Initialize the logger
logger = logging.getLogger(script_name)
logger.setLevel(logging.INFO)

# File handler for logging to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
)

# Stream handler for logging to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Logger initialized")