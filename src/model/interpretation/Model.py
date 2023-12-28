import torch
import traceback
import torch.nn as nn
import transformers
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, T5ForConditionalGeneration

class Sentinel(nn.Module):
    def __init__(self, roberta: RobertaModel, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        """
        Roberta is passive, with classification layer attached at the end.
        """

        self.tokenizer = tokenizer

        self.roberta = roberta
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(768, 768), nn.Dropout(0.25), nn.Linear(768, 2)
        )

    def forward(self, textBatch) -> torch.Tensor:
        # ensure text of same length within batches
        encodedText = self.tokenizer(
            textBatch, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to("cuda")

        # forward pass with last hidden states obtained
        lastHiddenStates = self.roberta(**encodedText).last_hidden_state

        # take <s> token (equiv. to [CLS]) to classify
        return self.fc(lastHiddenStates[:, 0, :])


class SentinelNonLinear(nn.Module):
    def __init__(self, roberta: RobertaModel, tokenizer: RobertaTokenizer, ret_hidden=False) -> None:
        super().__init__()
        """
        Roberta is passive, with classification layer attached at the end.
        """
        self.ret_hidden = ret_hidden

        self.tokenizer = tokenizer
        self.roberta = roberta

        roberta.requires_grad_ = False

        # Classification Backend
        self.fc = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(), nn.Dropout(0.25), nn.Linear(768, 2)
        )
        

    def forward(self, textBatch) -> torch.Tensor:
        # ensure text of same length within batches
        encodedText = self.tokenizer(
            textBatch, max_length=512, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to("cuda")

        # forward pass with last hidden states obtained
        lastHiddenStates = self.roberta(**encodedText).last_hidden_state
        hidden = lastHiddenStates[:, 0, :]
        
        if not self.ret_hidden:
            return self.fc(hidden)
        else:
            layers = [child for child in self.fc.children()]
            pro_fc = nn.Sequential(*layers[:1])
            epi_fc = nn.Sequential(*layers[1:])
            pro_fc.requires_grad_ = False
            epi_fc.requires_grad_ = False

            hidden_state = pro_fc(hidden)
            output = epi_fc(hidden_state)
            return output, hidden_state



## T5 Model

t5_eos_str = "</s>"
t5_positive_token = 1465    # tokenizer.encode("positive")
t5_negative_token = 2841    # tokenizer.encode("negative")
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


class T5SentinelEmbedder(nn.Module):
    def __init__(self, embedder) -> None:
        super().__init__()
        self.embedder = embedder
        self.vocab_tokens = []
        self.grad_storage = []

    def forward(self, *args, **kwargs):
        self.vocab_tokens.append(args[0])
        embed_tensor = self.embedder(*args, **kwargs)
        embed_tensor.requires_grad_(True)
        embed_tensor.retain_grad()
        self.grad_storage.append(embed_tensor)
        return embed_tensor

    def get_gradient(self):
        weight_gradient = self.embedder.weight.grad # [32128 x 512]
        output_gradient = self.grad_storage[0].grad  # [1 x len x 512]
        final_grad = output_gradient @ weight_gradient.T
        return final_grad

    def get_vocab(self): return self.vocab_tokens[0]

    def clear(self):
        self.vocab_tokens = []
        self.grad_storage = []


class T5SentinelRequireGrad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_eos = t5_eos_str
        self.t5_model: transformers.T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(**t5sentinel_config['t5_model'])
        self.t5_tokenizer = AutoTokenizer.from_pretrained(**t5sentinel_config['t5_tokenizer'])
        self.t5_embedder_expose = None

        def auto_embedder_substitution(*args, **kwargs):
            embedder = self.t5_model.get_input_embeddings()
            grad_embedder = T5SentinelEmbedder(embedder)
            self.t5_embedder_expose = grad_embedder
            self.t5_model.set_input_embeddings(grad_embedder)
            print("Embedder Substitution Complete")
            print(self.t5_model.get_input_embeddings())

        self.register_load_state_dict_post_hook(auto_embedder_substitution)
        

    def forward(self, text: str, label: str = "negative"):
        # return (P[fake], P[real])
        t5_text = self.t5_tokenizer.batch_encode_plus((text,), **t5sentinel_config['t5_tokenizer_text'])
        t5_text: torch.Tensor = t5_text.input_ids.to(t5sentinel_config["device"])

        t5_label = self.t5_tokenizer.batch_encode_plus((label,), **t5sentinel_config['t5_tokenizer_label'])
        t5_label = t5_label.input_ids.to(t5sentinel_config["device"])

        t5_output = self.t5_model(input_ids=t5_text, labels=t5_label)
        t5_logits = t5_output.logits
        t5_loss = t5_output.loss
        t5_loss.backward()

        positive_raw, negative_raw = t5_logits[:, 0, 1465], t5_logits[:, 0, 2841]
        t5_prob = torch.nn.functional.softmax(torch.tensor([positive_raw, negative_raw]), dim=-1)

        t5_encoder_grad: torch.Tensor = self.t5_embedder_expose.get_gradient()
        t5_tokens = self.t5_embedder_expose.get_vocab()
        self.t5_embedder_expose.clear()


        return t5_prob, t5_encoder_grad[0], t5_tokens[0]

if __name__ == "__main__":
    from pathlib import Path

    TEXT = "Hello hello hello hello hello hello hello"

    model = T5SentinelRequireGrad()
    model.eval()
    cp = torch.load(Path("../../../result/cache/t5.small.0422.pt"))
    model.load_state_dict(cp["model"])
    prob, grad, tokens = model(TEXT)
    # print(torch.norm(grad, p="fro"))
    print(prob)
