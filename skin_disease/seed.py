import random

import numpy as np
import torch


def set_seed(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
