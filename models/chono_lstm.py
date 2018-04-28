from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np


class ChronoLSTM(nn.Module):
    """docstring for ChronoLSTM"""
    def __init__(self, input_size, hidden_size, batch_size, chrono=False):
        super(ChronoLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.linear = nn.Linear(hidden_size, input_size)

        self.reset_parameters()
        if chrono:
            self.chrono_bias(self.input_size)

    def create_new_state(self):
        # Dimension: (num_layers, batch, hidden_size)
        lstm_h = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        lstm_c = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return lstm_h, lstm_c

    def chrono_bias(self, T):
        # Initialize the biases according to section 2 of the paper
        print("Chrono initialization engaged")
        # the learnable bias of the k-th layer (b_ii|b_if|b_ig|b_io),
        # of shape (4*hidden_size)

        # Set second bias vector to zero for forget and input gate
        self.lstm.bias_hh_l0.data[self.hidden_size * 2:] = 0

        # b_f ∼ log(𝒰 ([1, 𝑇 − 1]))
        # b_i = -b_f
        bias = np.log(np.random.uniform(1, T - 1, size=self.hidden_size))

        self.lstm.bias_ih_l0.data[:self.hidden_size] = -torch.Tensor(bias.copy())
        self.lstm.bias_ih_l0.data[self.hidden_size: self.hidden_size * 2] = torch.Tensor(bias)

    def reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.input_size + self.hidden_size))
                nn.init.uniform(p, -stdev, stdev)


    def size(self):
        return self.input_size, self.hidden_size

    def forward(self, x, state):
        lstm_h, lstm_c = state
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.input_size))

        x = x.unsqueeze(0)
        outp, (lstm_h, lstm_c) = self.lstm(x, (lstm_h, lstm_c))

        o = self.linear(outp.squeeze(0))

        return o.squeeze(0), (lstm_h, lstm_c)
