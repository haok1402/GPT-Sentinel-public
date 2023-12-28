import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to INFO
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

# Create file handler and set level to INFO
Path("logs").mkdir(parents=True, exist_ok=True)
fileHandler = RotatingFileHandler(f'./logs/{__name__}.log', maxBytes=1024*1024, backupCount=5)
fileHandler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')

# Add formatter to handlers
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)
