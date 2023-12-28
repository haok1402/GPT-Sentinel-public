import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Create handlers
if not logger.hasHandlers():

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler and set level to INFO
    Path("logs").mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(f'./logs/{__name__}.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log messages
    logger.info(f'Initializing {__name__}')
