import torch, random, numpy as np


def seed_Everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)

