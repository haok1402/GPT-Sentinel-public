import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
from src.model.roberta_mlp.model import Sentinel
import src.model.roberta_mlp.settings as settings
from src.model.roberta_mlp.utils.dataset import loadTrain, loadValid, loadTest
from src.model.roberta_mlp.utils.train import trainModel
from src.model.roberta_mlp.utils.valid import validModel
from src.model.roberta_mlp.utils.test import testModel

##############################################################################
# Dataset
##############################################################################

settings.logger.info("Loading train dataset...")
trainDataset, trainLoader = loadTrain()

settings.logger.info("Loading valid dataset...")
validDataset, validLoader = loadValid()

settings.logger.info("Loading test dataset...")
testDataset, testLoader = loadTest()

##############################################################################
# Model
##############################################################################
model = Sentinel(
    RobertaModel.from_pretrained("roberta-base"),
    RobertaTokenizer.from_pretrained("roberta-base"),
).to("cuda")

settings.logger.info("Initializing optimizer...")
optimizer = optim.AdamW(
    model.parameters(), 
    **settings.optimizer
)

settings.logger.info("Initializing scheduler...")
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    **settings.scheduler
)

settings.logger.info("Initializing criterion...")
criterion = nn.CrossEntropyLoss()

##############################################################################
# Trial
##############################################################################

settings.logger.info("Running trial...")
text, label = next(iter(validLoader))
with torch.inference_mode(): model.forward(text)
if settings.debug: exit()

##############################################################################
# Experiment
##############################################################################

directory = Path(f"./storage/{settings.id}")
directory.mkdir(parents=True, exist_ok=True)

task = wandb.init(
    name=settings.id,
    project="GPT",
    entity="deep-learner",
    mode="offline"
)

wandb.save("./src/model/roberta_mlp/__init__.py")
wandb.save("./src/model/roberta_mlp/__main__.py")
wandb.save("./src/model/roberta_mlp/model.py")
wandb.save("./src/model/roberta_mlp/settings.py")
wandb.save("./src/model/roberta_mlp/utils/train.py")
wandb.save("./src/model/roberta_mlp/utils/valid.py")
wandb.save("./src/model/roberta_mlp/utils/dataset.py")
wandb.watch(model, log="all")

with open(Path(directory, "architecture.txt"), "w") as file: 
    file.write(str(model))

##############################################################################
# Training
##############################################################################
if Path(directory, "state.pt").is_file():
    state = torch.load(Path(directory, "state.pt"))
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    
    startIter = state["epochIter"] + 1
    bestTestAccuracy = state["testAccuracy"]
else:
    startIter = 0
    bestTestAccuracy = float("-inf")

for epoch in range(startIter, settings.epochIter):
    settings.logger.info("Epoch: {}/{}".format(epoch + 1, settings.epochIter))

    learnRate = optimizer.param_groups[0]['lr']
    trainAccuracy, trainLoss = trainModel(model, trainLoader, criterion, optimizer)
    validAccuracy, testAccuracy = validModel(model, validLoader), testModel(model, testLoader)
    scheduler.step()

    wandb.log({
        "Training Accuracy": trainAccuracy * 100,
        "Training Loss": trainLoss,
        "Validation Accuracy": validAccuracy * 100,
        "Test Accuracy": testAccuracy * 100,
        "Learning Rate": learnRate
    })
    settings.logger.info("Training Accuracy: {:.2f}%".format(trainAccuracy * 100))
    settings.logger.info("Training Loss: {:.4f}".format(trainLoss))
    settings.logger.info("Validation Accuracy: {:.2f}%".format(validAccuracy * 100))
    settings.logger.info("Test Accuracy: {:.2f}%".format(testAccuracy * 100))
    settings.logger.info("Learning Rate: {:.6f}".format(learnRate))

    checkpoint = {
        "epochIter": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validAccuracy": validAccuracy,
        "testAccuracy": testAccuracy,
    }

    if testAccuracy >= bestTestAccuracy:
        bestTestAccuracy = testAccuracy
        torch.save(checkpoint, Path(directory, "state.pt"))
