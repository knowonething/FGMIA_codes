import random

import numpy as np
import torch
from loguru import logger as log


def seed_Everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)


def initSeeds(config):
    if not hasattr(config, "init_seed") or config.init_seed is None:
        seed = random.randint(0, 2 ** 15)
        log.info("Random Seed: {}", seed)
    else:
        seed = config.init_seed
        log.info("Setting Seed: {}", seed)

    seed_Everything(seed)
