import torch
import torch.nn as nn
import torch.utils as utils
from tqdm import tqdm


def testModel(
    model: nn.Module, 
    loader: utils.data.DataLoader, 
):
    """
    Perform testing.
    """

    # initialize testing
    model.eval()
    totalAccuracy, totalLoss = 0, 0
    progressBar = tqdm(total=len(loader), desc="Testing", ncols=120)

    # iterate over dataset
    for i, data in enumerate(loader):
        
        # move to cuda
        textBatch, labelBatch = data
        labelBatch = labelBatch.to("cuda")
        
        # forward pass
        logits = model(textBatch)
        totalAccuracy += torch.sum(
            torch.argmax(logits, dim=1) == labelBatch
        ).item() / labelBatch.shape[0]
        
        # display progress
        progressBar.set_postfix(
            testAccuracy="{0:.4%}".format(totalAccuracy / (i + 1)),
        )
        progressBar.update()
    
    # update statistics
    progressBar.close()
    totalAccuracy /= len(loader)

    return totalAccuracy
