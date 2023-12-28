import os, logging
from pathlib import Path

##############################################################################
# Experiment
##############################################################################

debug = False

id = "roberta+mlp.base.0424.a"
logger = logging.getLogger(id)

##############################################################################
# Hyperparameters
##############################################################################

epochIter = 15
batchSize = 64

optimizer = dict(
    lr=1e-4,
    weight_decay=1e-3,
)

scheduler = dict(
    eta_min=1e-6,
    T_max=epochIter,
)

##############################################################################
# Dataset
##############################################################################

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
