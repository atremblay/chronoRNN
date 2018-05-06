from attr import attrs, attrib, Factory
from torch import nn
from utils.argument import get_valid_fct_args
from torch import optim
from utils.Variable import Variable
from data import warp_data, to_categorical
import torch

import models
from utils.varia import hasnan


def compute_input_size(alphabet):
    return len(alphabet) + 1


# Generator of randomized test sequences
def dataloader(batch_size,
               epochs,
               num_batches,
               seq_len,
               max_repeat,
               uniform_warp,
               alphabet,
               padding_mode,
               pad):
    """Generator of random sequences for the warp task.

    """
    data = []
    for batch_num in range(num_batches):

        inp, outp = warp_data(
            seq_len,
            alphabet,
            max_repeat,
            uniform_warp,
            padding_mode,
            pad,
            batch_size
        )
        data.append((inp, outp))

    batch_num = 0
    for epoch in range(epochs):
        for inp, outp in data:

            inp = to_categorical(inp, num_classes=len(alphabet) + 1)
            # outp = to_categorical(outp, num_classes=len(alphabet) + 1)
            inp = Variable(torch.from_numpy(inp))
            outp = Variable(torch.LongTensor(outp))

            batch_num += 1

            yield batch_num, inp.float(), outp


@attrs
class TaskParams(object):
    name = attrib(default="warpTask")
    # Model params
    model_type = attrib(default=None)
    batch_size = attrib(default=32, convert=int) # "a batch size of 32 is used" p8, paragraph 2
    hidden_size = attrib(default=64, convert=int) # "[For the warp task], all networks have 64 units" p9, last paragraph
    # Optimizer params
    rmsprop_lr = attrib(default=2e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.9, convert=float) # "RMSProp with an alpha of 0.9 [is used]" p8, paragraph 2
    # Dataloader params
    epochs = attrib(default=3, convert=int)  # P.6, last paragraph. WE NEED TO DO EPOCHS!
    num_batches = attrib(default=50000/batch_size._default, convert=int) #TODO: This is dirty
    seq_len = attrib(default=500, convert=int) # P.6, last paragraph
    max_repeat = attrib(default=4, convert=int)
    uniform_warp = attrib(default=False, convert=bool)
    alphabet = attrib(default=range(1, 11), convert=list)
    pad = attrib(default=0, convert=int)
    padding_mode = attrib(default=False, convert=bool)
    chrono = attrib(default=False, convert=bool)
    gated = attrib(default=False, convert=bool) # it's very important that this remains False by default
    leaky = attrib(default=False, convert=bool) # it's very important that this remains False by default
    orthogonal_hidden_weight_init = attrib(default=False, convert=bool)


@attrs
class TaskModelTraining(object):
    params = attrib(default=Factory(TaskParams))
    net, dataloader, criterion, optimizer = attrib(), attrib(), attrib(), attrib()

    @staticmethod
    def loss_fn(net, X, Y, criterion):
        loss = Variable(torch.zeros(1))
        state = net.create_new_state()
        for i in range(X.size(0)):
            output, state = net(X[i], state)
            output = criterion(output, Y[i])
            loss += output
        assert not hasnan(loss), f"loss has NaNs: {loss.data.cpu()}"
        return loss

    @staticmethod
    def forward_fn(net, X, ):
        state = net.create_new_state()
        outputs = []  # The outputs can be of a variable size.
        for i in range(X.size(0)):
            output, state = net(X[i], state)
            outputs.append(output)
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
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=100, factor=0.5, verbose=True, threshold=0)

