# -*- coding: utf-8 -*-
# @Author: Alexis Tremblay
# @Date:   2018-04-24 08:41:35
# @Last Modified by:   Alexis Tremblay
# @Last Modified time: 2018-04-24 11:50:21


from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class ChronoLSTM(nn.Module):
    """A chrono LSTM implementation"""
    def __init__(self, input_size, hidden_size, batch_size=32):
        super(ChronoLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # Input gate
        self.w_xi = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # Forget gate
        self.w_xf = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # Cell state
        self.w_xc = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hc = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # Output gate
        self.w_xo = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # The hidden state is a learned parameter
        self.h = Parameter(torch.Tensor(hidden_size)).repeat(batch_size, 1)
        self.c = Parameter(torch.Tensor(hidden_size)).repeat(batch_size, 1)

        self.reset_parameters()
        self.chrono_bias(input_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def chrono_bias(self, T):
        # Initialize the biases according to section 2 of the paper
        # b_f ∼ log(𝒰 ([1, 𝑇 − 1]))
        # b_i = -b_f
        torch.nn.init.uniform(self.b_f, 1, T - 1)
        self.b_f.data.apply_(math.log)
        self.b_i = -self.b_f.clone()

    def size(self):
        return self.num_inputs, self.hidden_size

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        (h, c) = self.h, self.c
        # Input gate
        i = F.sigmoid(
            torch.mm(x, self.w_xi) + torch.mm(h, self.w_hi) + self.b_i
        )

        # Forget gate
        f = F.sigmoid(
            torch.mm(x, self.w_xf) + torch.mm(h, self.w_hf) + self.b_f
        )

        # c gate
        self.c *= f
        self.c += i * F.tanh(
            torch.mm(x, self.w_xc) + torch.mm(h, self.w_hc) + self.b_c
        )

        # Output gate
        o = F.sigmoid(
            torch.mm(x, self.w_xo) + torch.mm(h, self.w_ho) + self.b_o
        )

        # Hidden state
        self.h = o * F.tanh(c)

        # Current state
        state = (self.h, self.c)
        return self.h, state


class RNN(nn.Module):
    """A vanilla RNN implementation with a gated option"""
    def __init__(self, input_size, hidden_size, batch_size=32, gated=False):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gated = gated

        # Hidden state
        self.w_xh = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # Time warp gate
        self.w_gx = Parameter(torch.Tensor(input_size, hidden_size))
        self.w_gh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size)).unsqueeze(0)

        # The hidden state is a learned parameter
        self.h = Parameter(torch.Tensor(hidden_size)).repeat(batch_size, 1)
        self.g = Parameter(torch.Tensor(hidden_size)).repeat(batch_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def size(self):
        return self.num_inputs, self.hidden_size

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        (h, g) = self.h, self.g

        if self.gated:
            self.g = F.tanh(
                torch.mm(x, self.w_gx) + torch.mm(h, self.w_gh) + self.b_g
            )
            # Hidden state
            self.h = g * F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            ) + (1 - g) * h

            # Current state
            state = (self.h, self.g)
            return self.h, state
        else:
            # Hidden state
            self.h = F.tanh(
                torch.mm(x, self.w_xh) + torch.mm(h, self.w_hh) + self.b_h
            )
            return self.h
