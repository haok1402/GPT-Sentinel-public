import torch
import torch.nn as nn
import torch.utils as utils
from tqdm import tqdm
import src.model.roberta_mlp.settings as settings


def trainModel(
    model: nn.Module, 
    loader: utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: nn.Module, 
):
    """
    Perform training.
    """

    # initialize training
    model.train()
    totalAccuracy, totalLoss, effectiveBatchSize = 0, 0, 0
    progressBar = tqdm(total=len(loader), desc="Train", ncols=120)
    
    # iterate over dataset
    for i, data in enumerate(loader):
        
        # move to cuda
        textBatch, labelBatch = data
        labelBatch = labelBatch.to("cuda")
        
        # forward pass
        logits = model(textBatch)
        losses = criterion(logits, labelBatch)
        totalLoss += losses.item()
        totalAccuracy += torch.sum(
            torch.argmax(logits, dim=1) == labelBatch
        ).item() / labelBatch.shape[0]
        
        # display progress
        progressBar.set_postfix(
            trainAccuracy="{0:.4%}".format(totalAccuracy / (i + 1)),
            trainLoss="{:.06f}".format(totalLoss / (i + 1)),
        )
        progressBar.update()
        
        # backward propagation
        losses.backward()
        effectiveBatchSize += labelBatch.shape[0]
        
        # gradient accumulation
        if effectiveBatchSize >= settings.batchSize or i + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    
    # update statistics
    progressBar.close()
    totalAccuracy /= len(loader)
    totalLoss /= len(loader)

    return totalAccuracy, totalLoss
