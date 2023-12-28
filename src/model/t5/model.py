import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration
import src.model.t5.settings as settings
from typing import Tuple, Optional


class Sentinel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_eos = settings.t5_eos_str
        self.t5_model = T5ForConditionalGeneration.from_pretrained(**settings.sentinel['t5_model'])
        self.t5_tokenizer = AutoTokenizer.from_pretrained(**settings.sentinel['t5_tokenizer'])

    def forward(self, text: Tuple[str], label: Optional[Tuple[int]]):
        # encode (text, label)
        t5_text = self.t5_tokenizer.batch_encode_plus(text, **settings.sentinel['t5_tokenizer_text'])
        t5_text = t5_text.input_ids.to(settings.device)
        t5_label = self.t5_tokenizer.batch_encode_plus(label, **settings.sentinel['t5_tokenizer_label'])
        t5_label = t5_label.input_ids.to(settings.device)

        if self.training:
            t5_output = self.t5_model.forward(input_ids=t5_text, labels=t5_label)
            t5_loss, t5_logits = t5_output.loss, t5_output.logits
            t5_accuracy = torch.sum(
                torch.argmax(
                    F.softmax(t5_logits[:, 0, :], dim=-1), dim=-1
                ) == t5_label[:, 0]
            ) / settings.dataloader['train']['batch_size']
            return t5_loss, t5_accuracy
        else:
            t5_output = self.t5_model.generate(input_ids=t5_text, max_length=2, output_scores=True, return_dict_in_generate=True)
            t5_scores = t5_output.scores
            t5_accuracy = torch.sum(
                torch.argmax(
                    F.softmax(t5_scores[0], dim=-1), dim=-1
                ) == t5_label[:, 0]
            ) / settings.dataloader['valid']['batch_size']
            return t5_accuracy
