import torch
import random
import transformers
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def test(
        dataset: torch.utils.data.Dataset, 
        tokenizer: transformers.tokenization_utils.PreTrainedTokenizer, 
        detector_model: torch.nn.Module,
        sampled_amount: int = 250,
    ):
    """
    Iterate through dataset, and run the testing.
    """

    # confusion matrix
    true_positive, false_positive = 0, 0
    true_negative, false_negative = 0, 0
    
    # principal component analysis
    sampled_indices = random.sample([i for i in range(len(dataset))], k=sampled_amount)
    real_hiddens, fake_hiddens = [], []

    for i, (text, label) in tqdm(enumerate(dataset), desc="Test", total=len(dataset), ncols=100):
        # encode text
        tokens = tokenizer.encode(text, max_length=512)
        tokens = tokens[:tokenizer.max_len - 2]
        tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
        tokens, mask = tokens.to("cuda"), torch.ones_like(tokens).to("cuda")

        # forward pass
        with torch.no_grad():
            logits, hidden_states = detector_model(tokens, attention_mask=mask)
            probabilities = logits.softmax(dim=-1)
        
        # obtain labels
        fake_probability, real_probability = probabilities.detach().cpu().flatten().numpy().tolist()
        pred_label = "real" if real_probability > fake_probability else "fake"
        true_label = "real" if label == 0 else "fake"
        
        # update for confusion matrix
        if true_label == "fake" and pred_label == "fake":
            true_positive += 1
        if true_label == "real" and pred_label == "fake":
            false_positive += 1
        if true_label == "real" and pred_label == "real":
            true_negative += 1
        if true_label == "fake" and pred_label == "real":
            false_negative += 1
        
        # update for principal component analysis
        if i in sampled_indices:
            last_hidden = hidden_states[-1].detach().cpu().numpy()
            last_hidden = np.pad(last_hidden, ((0,0), (0, 512 - last_hidden.shape[1]), (0,0)))
            if true_label == "real":
                real_hiddens.append(last_hidden)
            elif true_label == "fake":
                fake_hiddens.append(last_hidden)
    
    # concatenate hidden states
    real_hiddens = np.concatenate(real_hiddens, axis=0)
    fake_hiddens = np.concatenate(fake_hiddens, axis=0)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive, 
        "true_negative": true_negative, 
        "false_negative": false_negative, 
        "real_hiddens": real_hiddens,
        "fake_hiddens": fake_hiddens
    }


def save_confusion_matrix(
        true_positive: int, 
        true_negative: int,
        false_negative: int,
        false_positive: int, 
        title: str, cached_path: str, filename: str, **kwargs
    ):
    """
    Draw and save confusion matrix.
    """
    
    data = np.array([
        [true_negative, false_positive], 
        [false_negative, true_positive]
    ])
    
    plt.figure(dpi=1024)
    plt.title(title)
    plt.imshow(np.array(data), cmap=plt.cm.Blues, interpolation='nearest')
    plt.colorbar()
    plt.clim(0, np.max(data))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Real', 'Generated'])
    plt.yticks([0, 1], ['Real', 'Generated'])
    
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, data[i, j],
                size="14", horizontalalignment="center",
                color="white" if data[i, j] > (np.sum(data) / 3) else "black",
            )

    Path(cached_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(cached_path, filename))


def save_pca_analysis(
        real_hiddens: np.ndarray,
        fake_hiddens: np.ndarray,
        title: str, cached_path: str, filename: str, **kwargs
    ):
    """
    Compute principal component analysis and save the figure.
    """

    real_hiddens = np.reshape(real_hiddens, (real_hiddens.shape[0], -1))
    fake_hiddens = np.reshape(fake_hiddens, (fake_hiddens.shape[0], -1))
    
    pca_core = PCA(n_components=2)
    pca_core.fit(np.concatenate([real_hiddens, fake_hiddens], axis=0))
    real_pca_data = pca_core.transform(real_hiddens)
    fake_pca_data = pca_core.transform(fake_hiddens)

    plt.figure(dpi=1024)
    plt.title(title)
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.scatter(real_pca_data[:, 0], real_pca_data[:, 1], label="Real Text", s=3)
    plt.scatter(fake_pca_data[:, 0], fake_pca_data[:, 1], label="Fake Text", s=3)
    plt.legend()
    
    Path(cached_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(cached_path, filename))
