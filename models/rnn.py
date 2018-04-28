# -*- coding: utf-8 -*-

from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Rnn(nn.Module):
    """A vanilla Rnn implementation with a gated option"""
    def __init__(self, input_size, hidden_size, batch_size=32, gated=False, leaky=False):
        super(Rnn, self).__init__()

        assert not (gated and leaky), "should be gated or leaky or neither, but can't be both"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gated = gated
        self.leaky = leaky

        # Hidden state
        self.w_xh = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size))

        # Learnable leak term
        if self.leaky:
            self.a = Parameter(torch.Tensor(1))

        # Time warp gate
        if self.gated:
            self.w_gx = Parameter(torch.Tensor(input_size, hidden_size))
            self.w_gh = Parameter(torch.Tensor(hidden_size, hidden_size))
            self.b_g = Parameter(torch.Tensor(hidden_size))

        self.linear = nn.Linear(hidden_size, input_size)

        self.reset_parameters()
        self.create_new_state()

    def create_new_state(self):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        h = Variable(torch.zeros(1, self.hidden_size).repeat(self.batch_size, 1))
        if self.gated:
            g = Variable(torch.zeros(1, self.hidden_size).repeat(self.batch_size, 1))
            return h, g
        else:
            return h,

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if weight.dim() == 1:
                weight.data.zero_()
            else:
                weight.data.uniform_(-stdv, stdv)

    def size(self):
        return self.input_size, self.hidden_size

    def forward(self, x, state):
        """
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.input_size))
        """
        if self.gated:
            h, g = state
            g = F.tanh(
                torch.mm(x, self.w_gx) + torch.mm(h, self.w_gh) + self.b_g
            )
            # Hidden state
            h = g * F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            ) + (1 - g) * h
            # Output
            o = self.linear(h)

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
