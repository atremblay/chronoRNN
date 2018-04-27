# -*- coding: utf-8 -*-

from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Rnn(nn.Module):
    """A vanilla Rnn implementation with a gated option"""
    def __init__(self, input_size, hidden_size, batch_size=32, gated=False):
        super(Rnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gated = gated
        if gated:
            print("Gating engaged")

        # Hidden state
        self.w_xh = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # Time warp gate
        self.w_gx = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_gh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # The hidden state is a learned parameter
        self.h_ = Parameter(torch.Tensor(1, hidden_size))
        self.g_ = Parameter(torch.Tensor(1, hidden_size))

        self.linear = nn.Linear(hidden_size, input_size)

        self.reset_parameters()
        self.create_new_state()

    def create_new_state(self):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        self.h = self.h_.clone().repeat(self.batch_size, 1)
        self.g = self.g_.clone().repeat(self.batch_size, 1)
        return self.h, self.g

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def size(self):
        return self.input_size, self.hidden_size

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.input_size))

        (h, g) = self.h, self.g

        if self.gated:
            self.g = F.tanh(
                torch.mm(x, self.w_gx) + torch.mm(h, self.w_gh) + self.b_g
            )
            # Hidden state
            self.h = g * F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            ) + (1 - g) * h
            # Output
            o = self.linear(self.h)

            # Current state
            state = (self.h, self.g)
            return o, state
        else:
            # Hidden state
            self.h = F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            )
            o = self.linear(self.h)
            return o, self.h
