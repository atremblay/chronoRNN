from torch.nn import Parameter
# from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
from utils.varia import hasnan, debug_inits
import logging
from utils.Variable import Variable


LOGGER = logging.getLogger(__name__)


class LSTM(nn.Module):
    """docstring for ChronoLSTM"""
    def __init__(self, input_size, hidden_size, batch_size, chrono=0, output_size=None,
                 orthogonal_hidden_init=False):
        super(LSTM, self).__init__()
        self.orthogonal_hidden_weight_init = orthogonal_hidden_init

        if output_size is None:
            output_size = input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.chrono = chrono

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.linear = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def create_new_state(self):
        # Dimension: (num_layers, batch, hidden_size)
        lstm_h = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        lstm_c = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        return lstm_h, lstm_c

    def chrono_bias(self):
        T = self.chrono
        # Initialize the biases according to section 2 of the paper
        print("Chrono initialization engaged")
        # the learnable bias of the k-th layer (b_ii|b_if|b_ig|b_io),
        # of shape (4*hidden_size)

        # Set second bias vector to zero for forget and input gate
        self.lstm.bias_hh_l0.data[self.hidden_size * 2:] = 0

        # b_f ∼ log(𝒰 ([1, 𝑇 − 1]))
        # b_i = -b_f
        bias = np.log(np.random.uniform(1, T - 1, size=self.hidden_size))

        print(self.lstm.bias_ih_l0.size())
        self.lstm.bias_ih_l0.data[:self.hidden_size] = -torch.Tensor(bias.copy())
        self.lstm.bias_ih_l0.data[self.hidden_size: self.hidden_size * 2] = torch.Tensor(bias)


    def reset_parameters(self):
        #############
        # Set the bias of the input gates to 1.
        # Explanation:
        #  "the ordering of weights a biases is the same for all implementations and is ingate, forgetgate,
        #   cellgate, outgate. You need to initialize the values between 1/4 and 1/2 of
        #   the bias vector to the desired value."
        #    - https://github.com/pytorch/pytorch/issues/750
        # Code taken from:
        #  https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        #############
        for name, weight in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform(weight)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

            if self.orthogonal_hidden_weight_init:
                # There is only one hidden weight matrix
                hidden_weight_name = list(filter(lambda n: "weight" in n and "hh" in n, names))
                assert len(hidden_weight_name) == 1
                hidden_weight_name = hidden_weight_name[0]
                hidden_weight = getattr(self.lstm, hidden_weight_name)
                torch.nn.init.orthogonal(hidden_weight)

        if self.chrono != 0:
            self.chrono_bias()

        debug_inits(self, LOGGER)

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
