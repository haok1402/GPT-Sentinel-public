import torch
import pandas as pd

class DualDataset(torch.utils.data.Dataset):
    def __init__(self, webTextPath: str, gptTextPath: str) -> None:
        super().__init__()

        # load from filesystem

        gptText = pd.read_json(gptTextPath, lines=True)['text']
        gptUID  = pd.read_json(gptTextPath, lines=True)['uid']
        all_uids = {uid for uid in gptUID}

        webData = pd.read_json(webTextPath, lines=True)
        webData = webData.loc[webData['uid'].isin(all_uids)]
        webText = webData["text"]
        webUID  = webData["uid"]

        # label accordingly
        self.data = [(text, 0) for text in webText] + [(text, 1) for text in gptText]
        self.uid  = [(uid, 0) for uid in webUID] + [(uid, 1) for uid in gptUID]
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(index, int):
            return self.data[index], self.uid[index]
        elif isinstance(index, str):
            web_text = self.data[self.uid.index((index, 0))]
            gpt_text = self.data[self.uid.index((index, 1))]
            return web_text, gpt_text

