import logging
import os
from datetime import datetime

## create a log folder if it is not there
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

## create a log file name with date and time
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

## configure the logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

## logging.INFO is the lowest level of logging
## logging.DEBUG is the second lowest level of logging
## logging.WARNING is the second highest level of logging
## logging.ERROR is the highest level of logging

## create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)

## add console handler to logger
logger = logging.getLogger("ODP")
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
