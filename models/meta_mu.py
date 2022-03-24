import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMu(nn.Module):

    def __init__(self, input_size, hidden_size, mid_size=10, sparse=None):
        super().__init__()
        concat_size = input_size + hidden_size
        self.linear_i = nn.Linear(in_features=concat_size, out_features=hidden_size)
        self.linear_c = nn.Linear(in_features=concat_size, out_features=hidden_size)
        self.linear_g = nn.Linear(in_features=concat_size, out_features=hidden_size)
        self.outl1 = nn.Linear(in_features=concat_size + 2 * hidden_size,
                               out_features=mid_size)
        self.outl2 = nn.Linear(in_features=mid_size,
                               out_features=hidden_size)
        self.gate_f = self.sparse_gate if sparse else torch.sigmoid

    def sparse_gate(self, x, bias=-1):
        return torch.sigmoid(x + bias) * nn.Softmax(1)(x)

    def forward(self, x, h_old, mean_old, precision_old):
        # TODO: How to deal with batch AND sequences?
        """
        if mean_old and precision_old is None:
            generate a prior here.
        """
        print('metaMu, x:', x.shape)

        cat_inpt = torch.cat((x, h_old), 1)

        precision_update_gate = self.gate_f(self.linear_i(cat_inpt))
        precision_update = F.softplus(self.linear_c(cat_inpt))
        new_precision = mean_old + precision_update_gate * precision_update

        mean_gate = precision_old / new_precision  # lambda
        mean_update = self.linear_g(cat_inpt)
        new_mean = mean_gate * mean_old + (1 - mean_gate) * mean_update

        h_new = self.outl2(F.softplus(self.outl1(torch.cat((cat_inpt, new_mean, 1 / new_precision), 1))))
        return h_new, new_mean, new_precision