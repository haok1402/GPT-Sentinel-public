__version__ = "1.0.0"

##############################################################################
# Set Environment
##############################################################################
import os
import json

with open("./secret.json") as f:
    secret = json.load(f)
    for key, val in secret.items():
        os.environ[key] = val
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

##############################################################################
# Create Logger
##############################################################################
import logging
from pathlib import Path
import src.model.t5.settings as settings
from logging.handlers import RotatingFileHandler

Path("logs").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(settings.id)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = RotatingFileHandler(
    f'./logs/{settings.id}.log', 
    maxBytes=1024*1024, backupCount=5
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"Initializing Sentinel {settings.id}...")
