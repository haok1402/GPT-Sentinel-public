import json
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

def construct_dirty_split():
    CLEAN_TEST_WEB = Path("data", "open-gpt-text-split", "test.jsonl")
    CLEAN_TEST_GPT = Path("data", "open-web-text-split", "test.jsonl")

    DIRTY_WEB_FILE = [p for p in Path("data", "open-web-text").iterdir() if p.name.endswith("jsonl")]
    DIRTY_GPT_FILE = [p for p in Path("data", "open-gpt-text").iterdir() if p.name.endswith("jsonl")]

    TARGET_WEB_TEST = Path("data", "open-web-text-split", "test-dirty.jsonl")
    TARGET_GPT_TEST = Path("data", "open-gpt-text-split", "test-dirty.jsonl")

    # Construct a UID set
    web_uid_set, gpt_uid_set = set(), set()
    with open(CLEAN_TEST_WEB, "r") as f:
        for line in f.read().strip().splitlines():
            web_uid_set.add(json.loads(line)["uid"])

    with open(CLEAN_TEST_GPT, "r") as f:
        for line in f.read().strip().splitlines():
            gpt_uid_set.add(json.loads(line)["uid"])

    # Construct Web Test
    write_f = open(TARGET_WEB_TEST, "w")
    for web_file in DIRTY_WEB_FILE:
        with open(web_file, "r") as f:
            for line in tqdm(f.read().strip().splitlines()):
                curr_uid = json.loads(line)["uid"]
                if curr_uid not in web_uid_set: continue
                write_f.write(line + "\n")

    write_f.close()

    # Construct GPT Test
    write_f = open(TARGET_GPT_TEST, "w")
    for gpt_file in DIRTY_GPT_FILE:
        with open(gpt_file, "r") as f:
            for line in tqdm(f.read().strip().splitlines()):
                curr_uid = json.loads(line)["uid"]
                if curr_uid not in gpt_uid_set: continue
                write_f.write(line + "\n")

    write_f.close()

def parse_zerogpt_result(web_input_file: Path, gpt_input_file: Path, output_file: Path):
    # In the cache file, all predictions are stored in format
    # uid<"-web" | "-gpt">: [prob_gpt, prob_human]
    with open(web_input_file, "r") as inpfd:
        web_lines = inpfd.read().strip().split("\n")
    with open(gpt_input_file, "r") as inpfd:
        gpt_lines = inpfd.read().strip().split("\n")
    
    prediction_result = dict()
    for line in web_lines:
        if line == "": continue
        entry = json.loads(line)
        uid = str(entry["uid"]) + "-web"
        pred_fake = (entry["res"]["data"]["fakePercentage"]) / 100
        prediction_result[uid] = np.array([pred_fake, 1 - pred_fake])
    
    for line in gpt_lines:
        if line == "": continue
        entry = json.loads(line)
        uid = str(entry["uid"]) + "-gpt"
        pred_fake = (entry["res"]["data"]["fakePercentage"]) / 100
        prediction_result[uid] = np.array([pred_fake, 1 - pred_fake])

    torch.save({"result": prediction_result, "source": "./data/utility_fns.py"}, output_file)

def parse_openai_result(web_input_file: Path, gpt_input_file: Path, output_file: Path):
    # In the cache file, all predictions are stored in format
    # uid<"-web" | "-gpt">: [prob_gpt, prob_human]
    with open(web_input_file, "r") as inpfd:
        web_lines = inpfd.read().strip().split("\n")
    with open(gpt_input_file, "r") as inpfd:
        gpt_lines = inpfd.read().strip().split("\n")
    
    prediction_result = dict()
    for line in web_lines:
        if line == "": continue
        entry = json.loads(line)
        uid = str(entry["uid"]) + "-web"

        top_logprob = entry["res"]["choices"][0]["logprobs"]["top_logprobs"][0]
        real_logprob = -10 if "!" not in top_logprob else top_logprob["!"]
        pred_real = np.exp(real_logprob)

        prediction_result[uid] = np.array([1 - pred_real, pred_real])
    
    for line in gpt_lines:
        if line == "": continue
        entry = json.loads(line)
        uid = str(entry["uid"]) + "-gpt"

        top_logprob = entry["res"]["choices"][0]["logprobs"]["top_logprobs"][0]
        real_logprob = -10 if "!" not in top_logprob else top_logprob["!"]
        pred_real = np.exp(real_logprob)

        prediction_result[uid] = np.array([1 - pred_real, pred_real])

    torch.save({"result": prediction_result, "source": "./data/utility_fns.py"}, output_file)
