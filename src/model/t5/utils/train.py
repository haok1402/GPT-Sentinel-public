import torch.nn as nn
import torch.utils as utils
from tqdm import tqdm
import src.model.t5.settings as settings


def trainModel(
    model: nn.Module, 
    loader: utils.data.DataLoader, 
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
    for i, (text, label) in enumerate(loader):

        # forward pass
        loss, accuracy = model(text, label)
        totalLoss += loss.item()
        totalAccuracy += accuracy.item()
        
        # display progress
        progressBar.set_postfix(
            trainAccuracy="{:.4%}".format(totalAccuracy / (i + 1)),
            trainLoss="{:.06f}".format(totalLoss / (i + 1)),
        )
        progressBar.update()
        
        # backward propagation
        loss.backward()
        effectiveBatchSize += settings.dataloader['train']['batch_size']
        
        # gradient accumulation
        if effectiveBatchSize >= settings.batchSize or i + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()
    
    # update statistics
    progressBar.close()
    totalAccuracy /= len(loader)
    totalLoss /= len(loader)

    return totalAccuracy, totalLoss
