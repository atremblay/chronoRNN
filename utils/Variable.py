from torch.autograd import Variable as torchVariable
import torch

# Variable Class Redirection for Cuda enabling context
FORCE_CPU = True
CUDA = (not FORCE_CPU) and torch.cuda.is_available()


def maybe_cuda(obj):
    if CUDA:
        return obj.cuda()
    else:
        return obj.cpu()


class CVariable:
    def __init__(self):
        self.isCuda = CUDA

    def __call__(self, x):
        if self.isCuda:
            return torchVariable(x).cuda()
        else:
            return torchVariable(x)


# Redirection instance
Variable = CVariable()
