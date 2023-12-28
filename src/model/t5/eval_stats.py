from pathlib import Path

import transformers
import string
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Tuple


web_file_path = "web_test.jsonl"
gpt_file_path = "gpt_test.jsonl"

t5_eos_str = "</s>"
t5sentinel_config = dict(
    device="cuda",
    t5_model=dict(
        pretrained_model_name_or_path="t5-small",
    ),
    t5_tokenizer=dict(
        pretrained_model_name_or_path="t5-small",
        model_max_length=512,
        return_tensors="pt",
    ),
    t5_tokenizer_text = dict(
        max_length=512,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    ),
    t5_tokenizer_label = dict(
        max_length=2,
        truncation=True,
        return_tensors="pt",
    ),
)


class Sentinel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_eos = t5_eos_str
        self.t5_model: transformers.T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
            **t5sentinel_config['t5_model'])
        self.t5_tokenizer = AutoTokenizer.from_pretrained(**t5sentinel_config['t5_tokenizer'])


    def forward(self, text: Tuple[str]):
        # encode (text, label)
        t5_text = self.t5_tokenizer.batch_encode_plus((text,), **t5sentinel_config['t5_tokenizer_text'])
        t5_text = t5_text.input_ids.to(t5sentinel_config["device"])
        t5_label = self.t5_tokenizer.batch_encode_plus((" ",), **t5sentinel_config['t5_tokenizer_label'])
        t5_label = t5_label.input_ids.to(t5sentinel_config["device"])
        t5_output = self.t5_model(input_ids=t5_text, output_hidden_states=True, labels=t5_label)
        t5_logits = t5_output.logits

        positive_raw, negative_raw = t5_logits[:, 0, 1465], t5_logits[:, 0, 2841]
        t5_prob = torch.nn.functional.softmax(torch.tensor([positive_raw, negative_raw]), dim=-1)

        return t5_prob


def filterout(text: string):
    result = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return result

def quick_statistics(prediction, threshold=0.5):
    TP, TN, FP, FN = 0, 0, 0, 0
    key: str
    for key in prediction:
        pred = prediction[key]
        p_gpt, p_web = pred[0], pred[1]
        pred_gpt = p_gpt > threshold
        real_gpt = key.endswith("gpt")

        if pred_gpt and real_gpt:
            TP += 1
        elif (not pred_gpt) and (not real_gpt):
            TN += 1
        elif pred_gpt and (not real_gpt):
            FP += 1
        else:
            FN += 1

    return TP, TN, FP, FN


def report_statistics(TP, TN, FP, FN):
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    with open("output.txt", "a") as file:
        file.write(f"True Positive: {TP} \t| True Negative: {TN} \n")
        file.write(f"False Positive:{FP} \t| False Negative:{FN} \n")
        file.write(f"True Positive Rate:  {round(TPR * 100, 2)}\% \n")
        file.write(f"True Negative Rate:  {round(TNR * 100, 2)}\% \n")
        file.write(f"False Positive Rate: {round(FPR * 100, 2)}\% \n")
        file.write(f"False Negative Rate: {round(FNR * 100, 2)}\% \n")
        file.write(f"Accuracy: {round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)}\% \n")
        file.write(f"F1 Score: {round((TP) / (TP + 0.5 * (FP + FN)), 2)} \n")


def calculate_t5_final():
    model = Sentinel().to('cuda')
    model.eval()
    cp = torch.load("t5.small.0422.pt")
    model.load_state_dict(cp["model"])
    model.to('cuda')

    t5_prediction_punc = dict()
    t5_prediction_nopunc = dict()

    with open(web_file_path, "r") as file:
        for line in file:
            try:
                text = json.loads(line)['text']
                id = json.loads(line)['uid']   # [urlsf_subset05] - [265829]
                prob = model(text)
                id += " web"
                t5_prediction_punc[id] = prob.cpu().numpy()

                newt = filterout(text)
                prob = model(newt)
                # print(prob[0].item(), prob[1].item())
                id += " web"
                t5_prediction_nopunc[id] = prob.cpu().numpy()

            except json.decoder.JSONDecodeError:
                continue
    print("t5_prediction_nopunc size" + str(len(t5_prediction_nopunc)))

    with open(gpt_file_path, "r") as file:
        for line in file:
            try:
                text = json.loads(line)['text']
                id = json.loads(line)['uid']  # [urlsf_subset05] - [265829]
                prob = model(text)
                id += " gpt"
                t5_prediction_punc[id] = prob.cpu().numpy()

                newt = filterout(text)
                prob = model(newt)
                # print(prob[0].item(), prob[1].item())
                id += " gpt"
                t5_prediction_nopunc[id] = prob.cpu().numpy()

            except json.decoder.JSONDecodeError:
                continue
    print("t5_prediction_nopunc size" + str(len(t5_prediction_nopunc)))
    return t5_prediction_punc, t5_prediction_nopunc


SELF_NAME = "eval.py"
PATH_CACHE = Path("result")

if __name__ == "__main__":
    t5_prediction_punc, t5_prediction_nopunc = calculate_t5_final()
    t5_statistics_punc = quick_statistics(t5_prediction_punc)
    t5_statistics_nopunc = quick_statistics(t5_prediction_nopunc)
    with open("output.txt", "a") as file:
        file.write("T5 on OpenGPTText-Final\n")
    report_statistics(*t5_statistics_punc)
    with open("output.txt", "a") as file:
        file.write("T5 on OpenGPTText-nopunc\n")
    report_statistics(*t5_statistics_nopunc)