# -*- coding: utf-8 -*-
# @Author: Alexis Tremblay
# @Date:   2018-04-24 08:41:35
# @Last Modified by:   Alexis Tremblay
# @Last Modified time: 2018-04-24 08:56:55


from torch.nn import Parameter, Variable
import torch.nn as nn
import torch
import numpy as np


class ChronoLSTM(nn.Module):
    """A chrono LSTM implementation"""
    def __init__(self, num_inputs, hidden_size, num_layers=1, batch_size=32):
        super(ChronoLSTM, self).__init__()

        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = 32

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=hidden_size,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(
            torch.randn(
                self.num_layers,
                1,
                self.hidden_size
            ) * 0.05
        )
        self.lstm_c_bias = Parameter(
            torch.randn(
                self.num_layers,
                1,
                self.hidden_size
            ) * 0.05
        )

        self.reset_parameters()
        self.reset_bias()

    def create_new_state(self):
        # Dimension: (num_layers, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, self.batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, self.batch_size, 1)
        self.previous_state = (lstm_h, lstm_c)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs + self.hidden_size))
                nn.init.uniform(p, -stdev, stdev)

    def reset_bias(self):
        # input-hidden bias of the kth layer (b_ii|b_if|b_ig|b_io),
        # of shape (4*hidden_size)

        # hidden-hidden bias of the kth layer (b_hi|b_hf|b_hg|b_ho),
        # of shape (4*hidden_size)

        # TODO initialize the biases
        pass

    def size(self):
        return self.num_inputs, self.hidden_size

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        x = x.unsqueeze(0)
        outp, self.previous_state = self.lstm(x, self.previous_state)
        return outp.squeeze(0), self.previous_state
