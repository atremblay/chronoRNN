import torch
import numpy as np
from torch.autograd import Variable
#from utils.Variable import Variable

def hasnan(x):
    if isinstance(x, Variable):
        x = x.data
        return np.isnan(x.sum())
    return (x != x).any() \
           or (x == torch.FloatTensor([float("-inf")])).any() \
           or (x == torch.FloatTensor([float("inf")])).any()


def debug_inits(module, logger):
    for name, param in module.named_parameters():
        assert not hasnan(param), f"{name}"

        var_str = None
        if np.prod(param.size()) > 1:
            var = torch.var(param).data
            assert (var < torch.FloatTensor([100.])).all(), f"{name}: {var}"
            var_str = f"var: {var.cpu().numpy()}"
            mean = torch.mean(param).data
            if name != 'b_g':
                assert (torch.abs(mean) <= torch.FloatTensor([1.])).all(), f"{name}: {mean}"
        else:
            var_str = "Vector is size 1, no variance."

        mean_str = f"mean: {mean.cpu().numpy()}"
        logger.debug(f"{name:14}: {var_str:18}  {mean_str:15}")

