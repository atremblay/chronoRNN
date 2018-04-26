from torch.autograd import Variable as torchVariable


# Variable Class Redirection for Cuda enabling context
class CVariable:

    def __init__(self):
        self.isCuda = False

    def __call__(self, x):
        if self.isCuda:
            return torchVariable(x).cuda()
        else:
            return torchVariable(x)

# Redirection instance
Variable = CVariable()
