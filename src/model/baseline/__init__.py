__version__ = "1.0.0"

import os
import json

with open("./secret.json") as f:
    secret = json.load(f)
    for key, val in secret.items():
        os.environ[key] = val
