import logging
import json
import unidecode
from pathlib import Path

logger = logging.getLogger(__name__)

WEB_DATA_ROOT = Path("data", "open-web-text-final")
WEB_ASCII_ROOT = Path("data", "open-web-text-final")
WEB_ASCII_ROOT.mkdir(parents=True, exist_ok=True)

GPT_DATA_ROOT = Path("data", "open-gpt-text-final")
GPT_ASCII_ROOT = Path("data", "open-gpt-text-final")
GPT_ASCII_ROOT.mkdir(parents=True, exist_ok=True)

logger.info("Normalizer start")

if (not WEB_DATA_ROOT.exists()) or (not GPT_DATA_ROOT.exists()):
    logger.error("Cannot find web data root or gpt data root")
    raise FileNotFoundError

AVAILABLE_GPT_SETS = [f for f in GPT_DATA_ROOT.glob("*.jsonl")]
AVAILABLE_WEB_SETS = [f for f in WEB_DATA_ROOT.glob("*.jsonl")]
for gpt_set in AVAILABLE_GPT_SETS:
    if gpt_set.name not in {f.name for f in AVAILABLE_WEB_SETS}:
        logger.error("Could not find " + str(gpt_set) + " in open-web-text")
        raise FileNotFoundError

AVAILABLE_WEB_SETS = [f for f in AVAILABLE_WEB_SETS if f.name in {g.name for g in AVAILABLE_GPT_SETS}]

# Helper Functions ##########################
# Get all tasks
def cache_check():
    for gpt_set in AVAILABLE_GPT_SETS:
        if not Path(GPT_ASCII_ROOT, gpt_set.name).exists(): return False
    for web_set in AVAILABLE_WEB_SETS:
        if not Path(WEB_ASCII_ROOT, web_set.name).exists(): return False
    return True

def get_task_uids(filePath: Path):
    with open(filePath, "r") as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        json.loads(line)
    uid = {json.loads(line)["uid"] for line in lines}
    return uid

# Process a single task
def unicode_to_ascii(string: str):
    return unidecode.unidecode(string)

def sanitize_gpt_entry(gpt_data: dict):
    new_data = gpt_data.copy()
    new_data["text"] = unicode_to_ascii(gpt_data["text"])
    return new_data

def sanitize_web_entry(web_data: dict):
    new_data = web_data.copy()
    new_data["text"] = unicode_to_ascii(web_data["text"])
    return new_data

##############################################

if cache_check():
    logger.info("Cached stripped ASCII dataset detected.")
else:
    TASK_UIDS = set()
    for gpt_set in AVAILABLE_GPT_SETS: TASK_UIDS = TASK_UIDS.union(get_task_uids(gpt_set))

    logger.info("Task UID retrieve successful.")
    logger.info(f"There are in total {len(TASK_UIDS)} tasks")

    # Process GPT Text

    for gpt_set in AVAILABLE_GPT_SETS:
        logger.info("Sanitizing " + str(gpt_set))
        new_gpt_set = Path(GPT_ASCII_ROOT, gpt_set.name)
        with open(gpt_set, "r") as old_f:
            lines = old_f.read().strip().split("\n")

        with open(new_gpt_set, "w") as new_f:
            for line in lines:
                original_text = json.loads(line)
                new_f.write(json.dumps(sanitize_gpt_entry(original_text)) + "\n")


    for web_set in AVAILABLE_WEB_SETS:
        logger.info("Sanitizing " + str(web_set))
        new_web_set = Path(WEB_ASCII_ROOT, web_set.name)
        if new_web_set.exists(): continue

        partial_web_text = []
        with open(web_set, "r") as old_f:
            lines = old_f.read().strip().split("\n")
        for line in lines:
            original_text = json.loads(line)
            if original_text["uid"] in TASK_UIDS: partial_web_text.append(original_text)
        with open(new_web_set, "w") as new_f:
            for original_text in partial_web_text:
                new_f.write(json.dumps(sanitize_web_entry(original_text)) + "\n")

logger.info("Normalizer Finished")
