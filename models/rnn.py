# -*- coding: utf-8 -*-

from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.Variable import maybe_cuda, Variable
from utils.varia import hasnan, debug_inits
import logging
import numpy as np
LOGGER = logging.getLogger(__name__)
DEBUG = False


class Rnn(nn.Module):
    """A vanilla RNN implementation with a gated option"""
    def __init__(self, input_size, hidden_size, max_repeat=None, batch_size=32, gated=False, leaky=False,
                 orthogonal_hidden_weight_init=True):
        super(Rnn, self).__init__()

        assert not (gated and leaky), "should be gated or leaky or neither, but can't be both"

        self.orthogonal_hidden_init = orthogonal_hidden_weight_init
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gated = gated
        self.leaky = leaky
        self.max_repeat = max_repeat

        # Hidden state
        self.w_xh = Parameter(maybe_cuda(torch.Tensor(input_size, hidden_size)))
        self.w_hh = Parameter(maybe_cuda(torch.Tensor(hidden_size, hidden_size)))
        self.b_h = Parameter(maybe_cuda(torch.Tensor(hidden_size)))

        # Learnable leak term
        if self.leaky:
            #self.a = Parameter(maybe_cuda(torch.Tensor(1)))
            self.a = Parameter(maybe_cuda(torch.Tensor(hidden_size)))

        # Time warp gate
        if self.gated:
            self.w_gx = Parameter(maybe_cuda(torch.Tensor(input_size, hidden_size)))
            self.w_gh = Parameter(maybe_cuda(torch.Tensor(hidden_size, hidden_size)))
            self.b_g = Parameter(maybe_cuda(torch.Tensor(hidden_size)))

        self.linear = nn.Linear(hidden_size, input_size)

        self.reset_parameters()

    def create_new_state(self):
        # Dimension: (batch, hidden_size)
        h = Variable(torch.zeros(self.batch_size, self.hidden_size))

        if self.gated:
            g = Variable(torch.zeros(self.batch_size, self.hidden_size))
            return h, g
        else:
            return h,

    def reset_parameters(self):
        self.linear.reset_parameters()
        for name, weight in self.named_parameters():
            if "linear." not in name:
                if self.orthogonal_hidden_init and (name == "w_hh" or name == "w_gh"):
                    torch.nn.init.orthogonal(weight)
                if name == "b_g":
                    if self.max_repeat is None:
                        torch.nn.init.constant(weight.data, 1)
                    else:
                        # -log(U([Tmin, Tmax]) - 1).
                        #torch.nn.init.uniform(weight.data, 1, 1 / (self.max_repeat))
                        torch.nn.init.uniform(weight.data, -np.log(1) - 1, -np.log(self.max_repeat) - 1)


                elif weight.dim() == 1:
                    weight.data.zero_()
                else:
                    torch.nn.init.xavier_uniform(weight.data)

        if self.leaky:
            # This is inspired from setting the forget bias to 1
            if self.max_repeat is None:
                #torch.nn.init.constant(self.a, 1 / (1 + np.exp(-1)))
                torch.nn.init.uniform(self.a, 1, 1 / (1 + np.exp(-1)))
            else:
                torch.nn.init.uniform(self.a, 1, 1 / (self.max_repeat))
                #torch.nn.init.constant(self.a, 1 / (self.max_repeat))

        debug_inits(self, LOGGER)


    def size(self):
        return self.input_size, self.hidden_size

    def forward(self, x, state):
        """
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.input_size))
        """
        if DEBUG:
            for name, param in self.named_parameters():
                assert not hasnan(param), f"{name} has nans"

        if self.gated:
            h, g = state

            pre_activation_g = torch.mm(g, self.w_gh) + self.b_g
            if x is not None:
                pre_activation_g += torch.mm(x, self.w_gx)

            g = F.sigmoid(pre_activation_g)
            if DEBUG:
                assert not hasnan(g)

            # Hidden state
            pre_activation_h = torch.mm(h, self.w_hh) + self.b_h
            if x is not None:
                pre_activation_h += torch.mm(x, self.w_xh)

            h = g * F.tanh(pre_activation_h) + (1 - g) * h
            if DEBUG:
                assert not hasnan(h)

            # Output
            o = self.linear(h)
            if DEBUG:
                assert not hasnan(o)

            # Current state
            state = (h, g)

            return o, state
        if self.leaky:
            # Hidden state
            h, = state

            pre_activation_h = torch.mm(h, self.w_hh) + self.b_h
            if x is not None:
                pre_activation_h += torch.mm(x, self.w_xh)

            h = self.a * F.tanh(pre_activation_h) + (1 - self.a) * h
            # Output
            o = self.linear(h)

            # Current state
            state = (h,)
            return o, state
        else:
            h, = state
            pre_activation = torch.mm(h, self.w_hh) + self.b_h
            if x is not None:
                pre_activation += torch.mm(x, self.w_xh)
            h = F.tanh(pre_activation)
            o = self.linear(h)
            return o, (h,)
