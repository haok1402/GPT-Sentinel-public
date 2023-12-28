import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import src.model.t5.settings as settings


class DualDataset(torch.utils.data.Dataset):
    def __init__(self, partition: str) -> None:
        super().__init__()

        web, gpt = [], []
        # load from filesystem
        web_text = pd.read_json(Path(settings.dataset["web_folder"], f"{partition}.jsonl"), lines=True)
        gpt_text = pd.read_json(Path(settings.dataset["gpt_folder"], f"{partition}.jsonl"), lines=True)
        web.extend(web_text["text"].tolist())
        gpt.extend(gpt_text["text"].tolist())
        assert len(web) == len(gpt)

        # label accordingly
        self.data = [(t, "negative") for t in web] + [(t, "positive") for t in gpt]
        self.length = len(self.data)

    def __len__(self, ) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.data[index]


def loadTrain():
    trainDataset = DualDataset("train")
    trainLoader = DataLoader(trainDataset, **settings.dataloader["train"])
    return trainDataset, trainLoader


def loadValid():
    validDataset = DualDataset("valid")
    validLoader = DataLoader(validDataset, **settings.dataloader["valid"])
    return validDataset, validLoader


def loadTest():
    testDataset = DualDataset("test")
    testLoader = DataLoader(testDataset, **settings.dataloader["test"])
    return testDataset, testLoader
