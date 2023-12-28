import torch
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


class T5Sentinel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t5_eos = t5_eos_str
        self.t5_model: transformers.T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(**t5sentinel_config['t5_model'])
        self.t5_tokenizer = AutoTokenizer.from_pretrained(**t5sentinel_config['t5_tokenizer'])

    def forward(self, text: str):
        # return (P[fake], P[real])
        t5_text = self.t5_tokenizer.batch_encode_plus((text,), **t5sentinel_config['t5_tokenizer_text'])
        t5_text = t5_text.input_ids.to(t5sentinel_config["device"])
        t5_label = self.t5_tokenizer.batch_encode_plus((" ",), **t5sentinel_config['t5_tokenizer_label'])
        t5_label = t5_label.input_ids.to(t5sentinel_config["device"])
        t5_output = self.t5_model(input_ids=t5_text, output_hidden_states=True, labels=t5_label)
        t5_logits = t5_output.logits
        t5_hidden = t5_output.decoder_hidden_states

        positive_raw, negative_raw = t5_logits[:, 0, 1465], t5_logits[:, 0, 2841]
        t5_prob = torch.nn.functional.softmax(torch.tensor([positive_raw, negative_raw]), dim=-1)

        return t5_prob, t5_hidden

if __name__ == "__main__":
    from pathlib import Path

    TEXT = "Image caption Phobos is the larger and closer of Mars' two moons\nScientists say they have uncovered firm evidence that Mars's biggest moon, Phobos, is made from rocks blasted off the Martian surface in a catastrophic event.\nThe origin of Mars's satellites Phobos and Deimos is a long-standing puzzle.\nIt has been suggested that both moons could be asteroids that formed in the main asteroid belt and were then \"captured\" by Mars's gravity.\nThe latest evidence has been presented at a major conference in Rome.\nThe new work supports other scenarios. Material blasted off Mars's surface by a colliding space rock could have clumped together to form the Phobos moon.\nAlternatively, Phobos could have been formed from the remnants of an earlier moon destroyed by Mars's gravitational forces. However, this moon might itself have originated from material thrown into orbit from the Martian surface.\nPrevious observations of Phobos at visible and near-infrared wavelengths have been interpreted to suggest the possible presence of carbonaceous chondrites, found in meteorites that have crashed to Earth.\nThis carbon-rich, rocky material, left over from the formation of the Solar System, is thought to originate in asteroids from the so-called \"main belt\" between Mars and Jupiter.\nBut, now, data from the European Space Agency's Mars Express spacecraft appear to make the asteroid capture scenario look less likely.\n'Poor agreement'\nRecent observations as thermal infrared wavelengths using the Planetary Fourier Spectrometer (PFS) instrument on Mars Express show a poor match between the rocks on Phobos and any class of chondritic meteorite known from Earth.\nThese would seem to support the \"re-accretion\" models for the formation of Phobos, in which rocks from the surface of the Red Planet are blasted into Martian orbit to later clump and form Phobos.\n\"We detected for the first time a type of mineral called phyllosilicates on the surface of Phobos, particularly in the areas northeast of Stickney, its largest impact crater,\" said co-author Dr Marco Giuranna, from the Italian National Institute for Astrophysics in Rome.\nThese phyllosilicate rocks are thought to form in the presence of water, and have been found previously on Mars.\n\"This is very intriguing as it implies the interaction of silicate materials with liquid water on the parent body prior to incorporation into Phobos,\" said Dr Giuranna.\n\"Alternatively, phyllosilicates may have formed in situ, but this would mean that Phobos required sufficient internal heating to enable liquid water to remain stable.\"\nRocky blocks\nOther observations from Phobos appear to match the types of minerals identified on the surface of Mars. Thus, the make-up of Phobos appears more closely related to Mars than to asteroids from the main belt, say the researchers.\nIn addition, said Pascal Rosenblatt of the Royal Observatory of Belgium, \"the asteroid capture scenarios also have difficulties in explaining the current near-circular and near-equatorial orbit of both Martian moons (Phobos and Deimos)\".\nThe researchers also used Mars Express to obtain the most precise measurement yet of Phobos' density.\n\"This number is significantly lower than the density of meteoritic material associated with asteroids. It implies a sponge-like structure with voids making up 25%-45% in Phobos's interior,\" said Dr Rosenblatt.\nA highly porous asteroid would have probably not survived if captured by Mars. Alternatively, such a highly porous structure on Phobos could have resulted from the re-accretion of rocky blocks in Mars' orbit.\nRussia's robotic mission to Phobos, named Phobos-Grunt (grunt means ground , or earth, in Russian) to be launched in 2011, will investigate the moon's composition in more detail.\nThe study has been submitted for publication in the peer-reviewed journal Planetary and Space Science. It was presented at the 2010 European Planetary Science Congress in Rome."

    model = T5Sentinel()
    model.eval()
    cp = torch.load(Path("../../../result/cache/t5.small.0422.pt"))
    model.load_state_dict(cp["model"])
    prob, hidden = model(TEXT)
    print(prob)
