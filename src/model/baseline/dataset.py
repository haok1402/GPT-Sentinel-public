import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple


class DualDataset(torch.utils.data.Dataset):
    def __init__(self, webFolder: Path, gptFolder: Path, subsets: List[str]) -> None:
        super().__init__()
        """
        A PyTorch dataset to load webText and the rephrased gptText.
        Note that this dataset is for output from gpt-3.5-turbo.
        """

        # load from file system; ignore any webtext that hasn't been rephrased
        webText, gptText = [], []
        for subset in subsets:
            webSubset = Path(webFolder, subset)
            gptSubset = Path(gptFolder, subset)
            for gptFile in tqdm(gptSubset.iterdir(), desc=subset):
                if (webFile := Path(webSubset, gptFile.name)).is_file():
                    with open(gptFile, "r") as f:
                        if not (content:= f.read().strip()): continue
                        gptText.append(content)
                    with open(webFile, "r") as f:
                        if not (content:= f.read().strip()): continue
                        webText.append(content)        
        assert len(webText) == len(gptText)

        # assign 0 as label for webText and 1 as label for gptText.
        self.data = [(text, 0) for text in webText] + [(text, 1) for text in gptText]
        self.length = len(self.data)

    def __len__(self, ) -> int:
        """
        Return length of the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        Return (text, label) at the given index.
        """
        return self.data[index]
