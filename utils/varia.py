import torch
import numpy as np
from torch.autograd import Variable


def hasnan(x):
    if isinstance(x, Variable):
        x = x.data
    return (x != x).any() \
           or (x == torch.FloatTensor([float("-inf")])).any() \
           or (x == torch.FloatTensor([float("inf")])).any()


def debug_inits(module, logger):
    for name, param in module.named_parameters():
        assert not hasnan(param), f"{name}"
        vars_str = f"var: {torch.var(param).data.cpu().numpy()}"
        mean_str = f"mean: {torch.mean(param).data.cpu().numpy()}"
        logger.info(f"{name:14}: {vars_str:18}  {mean_str:15}")
