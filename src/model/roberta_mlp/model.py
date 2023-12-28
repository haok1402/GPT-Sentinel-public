import torch.nn as nn
from typing import Tuple
from torch import Tensor
from transformers import RobertaModel, RobertaTokenizer


class Sentinel(nn.Module):
    def __init__(self, roberta: RobertaModel, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        """
        Roberta is passive, with classification layer attached at the end.
        """
        self.roberta = roberta
        for param in self.roberta.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer

        self.fc = nn.Sequential(
            nn.Linear(768, 768), 
            nn.GELU(), nn.Dropout(0.25),
            nn.Linear(768, 2)
        )
        
        # self.fc = nn.Sequential(
        #     nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.25),
        #     nn.Linear(768, 768), nn.GELU(), 
        #     nn.Linear(768, 768), nn.GELU(), 
        #     nn.Linear(768, 2)
        # )
        
    def forward(self, textBatch: Tuple) -> Tensor:
        
        # ensure text of same length
        encodedText = self.tokenizer(
            textBatch, max_length=512, truncation=True, 
            padding="max_length", return_tensors="pt",
        ).to("cuda")

        # forward through the roberta
        lastHiddenStates = self.roberta(**encodedText).last_hidden_state

        # take <s> token (equiv. to [CLS]) to classify
        logits = self.fc(lastHiddenStates[:, 0, :])
        return logits
