# -*- coding: utf-8 -*-
# @Author: Jules Gagnon-Marchand
from attr import attrs, attrib, Factory
from torch import nn
from utils.argument import get_valid_fct_args
from torch import optim
import model
from data import add_data
import torch
from utils.Variable import Variable

# Generator of randomized test sequences
def dataloader(batch_size,
               num_batches,
               seq_len,
               ):
    """Generator of random sequences for the add task.

    """
    for batch_num in range(num_batches):

        inp, outp = add_data(seq_len, batch_size)

        inp = Variable(torch.from_numpy(inp))
        outp = Variable(torch.from_numpy(outp))
        yield batch_num + 1, inp.float().unsqueeze(-1).permute(1, 0, 2), outp.float()


@attrs
class TaskParams(object):
    # ALL OF THIS NEEDS TO BE CHECKED
    name = attrib(default="addTask")
    # Model params
    model_type = attrib(default="Rnn")
    batch_size = attrib(default=1, convert=int)
    input_size = attrib(default=1, convert=int)
    hidden_size = attrib(default=128, convert=int)
    # Optimizer params
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    # Dataloader params
    num_batches = attrib(default=1000, convert=int)
    seq_len = attrib(default=10, convert=int)


@attrs
class TaskModelTraining(object):
    params = attrib(default=Factory(TaskParams))
    net, dataloader, criterion, optimizer = attrib(), attrib(), attrib(), attrib()

    @net.default
    def default_net(self):
        net = getattr(model, self.params.model_type)
        net = net(**get_valid_fct_args(net.__init__, self.params))
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(**get_valid_fct_args(dataloader, self.params))

    @criterion.default
    def default_criterion(self):
        return nn.MSELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
