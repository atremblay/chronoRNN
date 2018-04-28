from attr import attrs, attrib, Factory
from torch import nn
from utils.argument import get_valid_fct_args
from torch import optim
from utils.Variable import Variable
from data import warp_data, to_categorical
import torch

import models

def compute_input_size(alphabet):
    return len(alphabet) + 1


# Generator of randomized test sequences
def dataloader(batch_size,
               num_batches,
               seq_len,
               max_repeat,
               uniform_warp,
               alphabet,
               pad):
    """Generator of random sequences for the warp task.

    """
    for batch_num in range(num_batches):

        inp, outp = warp_data(
            seq_len,
            alphabet,
            max_repeat,
            uniform_warp,
            pad,
            batch_size
        )

        inp = to_categorical(inp, num_classes=len(alphabet) + 1)
        # outp = to_categorical(outp, num_classes=len(alphabet) + 1)
        inp = Variable(torch.from_numpy(inp))
        outp = Variable(torch.LongTensor(outp))

        yield batch_num + 1, inp.float(), outp


@attrs
class TaskParams(object):
    name = attrib(default="warpTask")
    # Model params
    model_type = attrib(default=None)
    batch_size = attrib(default=32, convert=int) # "a batch size of 32 is used" p8, paragraph 2
    hidden_size = attrib(default=64, convert=int) # "[For the warp task], all networks have 64 units" p9, last paragraph
    # Optimizer params
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.9, convert=float) # "RMSProp with an alpha of 0.9 [is used]" p8, paragraph 2
    # Dataloader params
    num_batches = attrib(default=50000, convert=int) # P.6, last paragraph. WE NEED TO DO EPOCHS!
    seq_len = attrib(default=500, convert=int) # P.6, last paragraph
    max_repeat = attrib(default=4, convert=int)
    uniform_warp = attrib(default=False, convert=bool)
    alphabet = attrib(default=range(1, 11), convert=list)
    pad = attrib(default=0, convert=int)
    chrono = attrib(default=False, convert=bool)
    gated = attrib(default=False, convert=bool) # it's very important that this remains False by default
    leaky = attrib(default=False, convert=bool) # it's very important that this remains False by default


@attrs
class TaskModelTraining(object):
    params = attrib(default=Factory(TaskParams))
    net, dataloader, criterion, optimizer = attrib(), attrib(), attrib(), attrib()

    @staticmethod
    def loss_fn(net, X, Y, criterion):
        loss = Variable(torch.zeros(1))
        net.create_new_state()
        for i in range(X.size(0)):
            loss += criterion(net(X[i])[0], Y[i])
        return loss

    @staticmethod
    def forward_fn(net, X, ):
        net.create_new_state()
        outputs = []  # The outputs can be of a variable size.
        for i in range(X.size(0)):
            outputs.append(net(X[i])[0])
        return outputs

    def dataloader_fn(self):
        """
        Creates a news dataloader generator, for when the old one is exhausted
        """
        return dataloader(**get_valid_fct_args(dataloader, self.params))

    @net.default
    def default_net(self):
        self.params.input_size = compute_input_size(self.params.alphabet)
        net = getattr(models, self.params.model_type)
        net = net(**get_valid_fct_args(net.__init__, self.params))
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(**get_valid_fct_args(dataloader, self.params))

    @criterion.default
    def default_criterion(self):
        return nn.CrossEntropyLoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
    @staticmethod
    def make_scheduler(optim):
        # "learning rates are divided by 2 each time the evaluation loss has not decreased after 100 batches" - p.8
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.5)

