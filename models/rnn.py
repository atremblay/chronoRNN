# -*- coding: utf-8 -*-

from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.Variable import maybe_cuda, Variable
from utils.varia import hasnan, debug_inits
import logging
from numpy import np
LOGGER = logging.getLogger(__name__)
DEBUG = False


class Rnn(nn.Module):
    """A vanilla Rnn implementation with a gated option"""
    def __init__(self, input_size, hidden_size, batch_size=32, gated=False, leaky=False,
                 orthogonal_hidden_weight_init=True):
        super(Rnn, self).__init__()

        assert not (gated and leaky), "should be gated or leaky or neither, but can't be both"

        self.orthogonal_hidden_init = orthogonal_hidden_weight_init
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gated = gated
        self.leaky = leaky

        # Hidden state
        self.w_xh = Parameter(maybe_cuda(torch.Tensor(input_size, hidden_size)))
        self.w_hh = Parameter(maybe_cuda(torch.Tensor(hidden_size, hidden_size)))
        self.b_h = Parameter(maybe_cuda(torch.Tensor(hidden_size)))

        # Learnable leak term
        if self.leaky:
            self.a = Parameter(maybe_cuda(torch.Tensor(1)))

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
                    weight.data.one_()
                elif weight.dim() == 1:
                    weight.data.zero_()
                else:
                    torch.nn.init.xavier_uniform(weight.data)

        if self.leaky:
            # This is inspired from setting the forget bias to 1
            torch.nn.init.constant(self.a, 1 / (1 + np.exp(-1)))

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
            g = F.sigmoid(
                torch.mm(x, self.w_gx) + torch.mm(g, self.w_gh) + self.b_g
            )
            if DEBUG:
                assert not hasnan(g)
            # Hidden state
            h = g * F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            ) + (1 - g) * h
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
            h = self.a * F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            ) + (1 - self.a) * h
            # Output
            o = self.linear(h)

            # Current state
            state = (h,)
            return o, state
        else:
            h, = state
            h = F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            )
            o = self.linear(h)
            return o, (h,)
