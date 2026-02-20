import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Ensure reproducibility across torch, numpy, and python random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)