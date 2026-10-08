"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every relevant RNG for reproducible runs.

    Args:
        seed: the integer seed.
        deterministic: if True, force deterministic cuDNN kernels. This can
            slow training down but removes a major source of run-to-run noise,
            which matters when reporting paper-grade numbers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seeding so DataLoader augmentation stays reproducible."""
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
