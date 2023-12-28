import torch.nn as nn
import torch.utils as utils
from tqdm import tqdm


def validModel(
    model: nn.Module, 
    loader: utils.data.DataLoader, 
):
    """
    Perform validation.
    """

    # initialize validation
    model.eval()
    totalAccuracy, totalLoss = 0, 0
    progressBar = tqdm(total=len(loader), desc="Valid", ncols=120)

    # iterate over dataset
    for i, (text, label) in enumerate(loader):
        
        # forward pass
        accuracy = model(text, label)
        totalAccuracy += accuracy.item()
        
        # display progress
        progressBar.set_postfix(
            validAccuracy="{:.4%}".format(totalAccuracy / (i + 1)),
        )
        progressBar.update()
    
    # update statistics
    progressBar.close()
    totalAccuracy /= len(loader)

    return totalAccuracy
