# roberta
This is the directory for detector model using `roberta` to do the masked-out-word prediction task, where the text we feed to the model is  appended with "The label of this text is <mask>". When the text is from `open-web-text` dataset, the training objective for the masked-out-word is "positive", and vice versa. Note that in contrary to `roberta-mlp`, the detector model contains an active `roberta` that not just extracting features, but actually being trained with an objective in mind.