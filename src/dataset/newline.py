import re
import pandas as pd
from pathlib import Path
from typing import Callable
from src.dataset import logger


def reduce_newline(text: str) -> str:
    """
    Reduce text with multiple continuous newlines to one with single newline.
    """
    return re.sub(r"\n+", r"\n", text)


def process_dataset(func: Callable[[str], str] = reduce_newline) -> str:
    """
    Map document from dataset with specified reduction.
    """
    logger.info("Mapping dataset with function: %s", func.__name__)

    gpt_folder = Path("data", "open-gpt-text-final")
    gpt_strip_folder = Path("data", "open-gpt-text-final")
    gpt_strip_folder.mkdir(parents=True, exist_ok=True)
    
    web_folder = Path("data", "open-web-text-final")
    web_strip_folder = Path("data", "open-web-text-final")
    web_strip_folder.mkdir(parents=True, exist_ok=True)

    for gpt_file in gpt_folder.glob("*.jsonl"):
        logger.info("Processing file: %s", gpt_file)
        gpt_data = pd.read_json(gpt_file, lines=True)
        gpt_data["text"] = gpt_data["text"].map(func)
        gpt_data.to_json(Path(gpt_strip_folder, gpt_file.name), orient="records", lines=True)

        web_file = Path(web_folder, gpt_file.name)
        logger.info("Processing file: %s", web_file)
        web_data = pd.read_json(web_file, lines=True)
        web_data = web_data[web_data['uid'].isin(gpt_data["uid"])]
        web_data["text"] = web_data["text"].map(func)
        web_data.to_json(Path(web_strip_folder, gpt_file.name), orient="records", lines=True)


if __name__ == "__main__":
    process_dataset()
