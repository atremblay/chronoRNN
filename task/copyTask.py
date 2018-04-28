# -*- coding: utf-8 -*-
from attr import attrs, attrib, Factory
from torch import nn
from utils.argument import get_valid_fct_args
from torch import optim
import models
from data import copy_data, to_categorical
import torch
from utils.Variable import Variable


def compute_input_size(alphabet):
    return len(alphabet) + 3


# Generator of randomized test sequences
def dataloader(batch_size, num_batches,
               alphabet, dummy, eos, variable, seq_len,
               ):
    """
    Generator of random sequences for the add task.
    """
    for batch_num in range(num_batches):
        inp, outp = copy_data(alphabet=alphabet, dummy=dummy, eos=eos,
                              batch_size=batch_size, variable=variable, T=seq_len)
        inp = Variable(torch.from_numpy(to_categorical(inp, num_classes=compute_input_size(alphabet))))
        outp = Variable(torch.from_numpy(outp))

        yield batch_num + 1, inp, outp.long()


@attrs
class TaskParams(object):
    # ALL OF THIS NEEDS TO BE CHECKED
    batch_size = attrib(default=1, convert=int)
    name = attrib(default="copyTask")
    # Model params
    model_type = attrib(default=None)
    hidden_size = attrib(default=128, convert=int) # p8, paragraph 7
    # Optimizer params
    rmsprop_lr = attrib(default=10**-3, convert=float)  # "The synthetic tasks use a LR of 10^-3" p8, 4th paragraph
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.9, convert=float) # "[...] a moving average parameter of 0.9" p8, paragraph 4
    # Dataloader params
    num_batches = attrib(default=1000, convert=int)
    seq_len = attrib(default=500, convert=int)
    variable = attrib(default=False, convert=bool)
    alphabet = attrib(default=range(1, 9), convert=list)
    chrono = attrib(default=False, convert=bool)
    dummy = attrib(default=9, convert=int)
    eos = attrib(default=10, convert=int)
    leaky = attrib(default=False, convert=bool) # it's very important that this remains False by default
    gated = attrib(default=False, convert=bool) # it's very important that this remains False by default

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
        net = getattr(models, self.params.model_type)
        self.params.input_size = compute_input_size(self.params.alphabet)
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
