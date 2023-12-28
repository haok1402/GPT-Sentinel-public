import os
import torch
import requests
import json

import pandas as pd
import numpy as np

from pathlib import Path
from tqdm.notebook import tqdm

def download_gpt2(dataDirectory):
    for ds in ['webtext', 'small-117M',  'small-117M-k40',
               'medium-345M', 'medium-345M-k40',
               'large-762M',  'large-762M-k40',
               'xl-1542M',    'xl-1542M-k40']:
        for split in ['test']:
            filename = ds + "." + split + '.jsonl'
            r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(os.path.join(dataDirectory, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

class OpenGPTDataset(torch.utils.data.Dataset):
    def __init__(self, webTextPath: str, gptTextPath: str, force_match=True) -> None:
        super().__init__()

        # load from filesystem
        assert Path(webTextPath).exists()
        assert Path(gptTextPath).exists()

        gptText = pd.read_json(gptTextPath, lines=True)['text']
        gptUID  = pd.read_json(gptTextPath, lines=True)['uid']
        all_uids = {uid for uid in gptUID}

        webData = pd.read_json(webTextPath, lines=True)

        if force_match: webData = webData.loc[webData['uid'].isin(all_uids)]

        webText = webData["text"]
        webUID  = webData["uid"]

        # label accordingly
        self.data = [(text, 0) for text in webText] + [(text, 1) for text in gptText]
        self.uid  = [(uid, 0) for uid in webUID] + [(uid, 1) for uid in gptUID]
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int|str):
        if isinstance(index, int):
            return self.data[index], self.uid[index]
        elif isinstance(index, str):
            web_text = self.data[self.uid.index((index, 0))]
            gpt_text = self.data[self.uid.index((index, 1))]
            return web_text, gpt_text


class GPT2_OutputDataset(torch.utils.data.Dataset):
    def __init__(self, gpt_jsonl_path: Path, human_jsonl_path: Path):
        super().__init__()
        assert gpt_jsonl_path.exists()
        assert human_jsonl_path.exists()

        with open(human_jsonl_path, "r") as f:
            human_texts = [json.loads(line) for line in tqdm(f.read().strip().splitlines(), desc="Loading Human text", ncols=100)]

        with open(gpt_jsonl_path, "r") as f:
            gpt_texts = [json.loads(line) for line in tqdm(f.read().strip().splitlines(), desc="Loading GPT text", ncols=100)]

        # We need to sanitize the dataset since some of them are invalid (e.g. #255123 in small-117M.test.jsonl is empty)
        gpt_texts = [entity for entity in gpt_texts if len(entity["text"]) > 0]
        human_texts = [entity for entity in human_texts if len(entity["text"]) > 0]

        self.human_dict = {human_text["id"] : human_text for human_text in human_texts}
        self.gpt_dict = {gpt_text["id"]: gpt_text for gpt_text in gpt_texts}

        self.data = [(human_text, 0) for human_text in human_texts] + [(gpt_text, 1) for gpt_text in gpt_texts]
        self.length = len(self.data)
        print("<All data loaded>")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        entry =  self.data[index]
        return entry[0]["text"], entry[1]

    def retrieve_id(self, text_id):
        assert text_id in self.human_dict
        assert text_id in self.gpt_dict
        return self.human_dict[text_id]["text"], self.gpt_dict[text_id]["text"]
        
