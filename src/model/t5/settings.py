import torch
import os, logging
from pathlib import Path

##############################################################################
# Experiment
##############################################################################

debug  = False
resume = False

id = "t5.small.0424.d"
logger = logging.getLogger(id)
device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################
# Hyperparameters
##############################################################################

epochIter = 10
batchSize = 64
learnRate = 5e-5
weigthDecay = 1e-3

optimizer = dict(
    lr=learnRate,
    weight_decay=weigthDecay,
)

# scheduler = dict(
#     eta_min=1e-6,
#     T_max=epochIter,
# )

# scheduler = dict(
#     step_size=1,
# <<<<<<< HEAD
#     gamma=0.7,
# =======
#     gamma=0.5,
# >>>>>>> ca35dc75fba7c7811f1232e3242ad2e69d58f386
# )

##############################################################################
# Model
##############################################################################

sentinel = dict(
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

##############################################################################
# Dataset
##############################################################################

t5_eos_str = "</s>"
t5_positive_token = 1465    # tokenizer.encode("positive")
t5_negative_token = 2841    # tokenizer.encode("negative")

dataset = dict(
    web_folder=Path(Path.home(), "GPT-Sentinel/data/open-web-text-split"),
    gpt_folder=Path(Path.home(), "GPT-Sentinel/data/open-gpt-text-split"),
)

dataloader = dict(
    train = dict(
        batch_size=32, shuffle=True, 
        num_workers=min(8, os.cpu_count()),
    ),
    valid = dict(
        batch_size=64, shuffle=True, 
        num_workers=min(8, os.cpu_count()),
    ),
    test = dict(
        batch_size=64, shuffle=True,
        num_workers=min(8, os.cpu_count()),
    )
)

##############################################################################
# Wandb
##############################################################################

wandb = dict(
    name=id,
    project="GPT",
    entity="deep-learner",
    mode="offline" if debug else "online",
    resume="must" if resume else None,
)
