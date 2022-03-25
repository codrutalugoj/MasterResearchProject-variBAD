import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMu(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, mid_size=10, sparse=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        concat_size = input_size + hidden_size

        self.linear_i = nn.Linear(in_features=concat_size, out_features=latent_dim)
        self.linear_c = nn.Linear(in_features=concat_size, out_features=latent_dim)
        self.linear_g = nn.Linear(in_features=concat_size, out_features=latent_dim)
        self.outl1 = nn.Linear(in_features=concat_size + 2 * latent_dim,
                               out_features=mid_size)
        self.outl2 = nn.Linear(in_features=mid_size,
                               out_features=hidden_size)
        self.gate_f = self.sparse_gate if sparse else torch.sigmoid

    def sparse_gate(self, x, bias=-1):
        return torch.sigmoid(x + bias) * nn.Softmax(1)(x)

    def forward(self, x, h_old, old_mean, old_precision):
        # TODO: How to deal with batch AND sequences?
        """
        if mean_old and precision_old is None:
            generate a prior here.
        """
        print('metaMu')
        print(' x:', x.shape) #  [1, 16, 16] / [60, 16, 16]
        print(' h_old:', h_old.shape) #  [1, 16, 64] / [1, 16, 64] # 64=gru_hidden_size

        print(' mean_old:', old_mean.shape)
        if len(old_mean.shape) == 2:
            old_mean = old_mean.unsqueeze(0)  # add sequence len dimension
            # print('  mean_old:', mean_old.shape)

        print(' prec._old:', old_precision.shape)
        if len(old_precision.shape) == 2:
            old_precision = old_precision.unsqueeze(0)  # add sequence len dimension
            # print('  prec._old:', precision_old.shape)

        seq_len = x.shape[0]
        batch = x.shape[1]

        outputs = torch.zeros((seq_len, batch, self.hidden_size))
        out_means = torch.zeros((seq_len, batch, self.latent_dim))
        out_precisions = torch.zeros((seq_len, batch, self.latent_dim))

        cat_inpt = torch.cat((x, h_old), -1)
        print(' cat_input:', cat_inpt.shape)

        for t in range(seq_len):
            precision_update_gate = self.gate_f(self.linear_i(cat_inpt[t]))
            precision_update = F.softplus(self.linear_c(cat_inpt[t]))
            # print(' old prec', old_precision)
            # print(' prec update gate', precision_update_gate)
            # print(' prec update', precision_update)
            out_precisions[t] = old_precision + precision_update_gate * precision_update
            # print(' out prec', out_precisions[t])

            mean_gate = old_precision / out_precisions[t]  # lambda
            mean_update = self.linear_g(cat_inpt[t])
            out_means[t] = mean_gate * old_mean + (1 - mean_gate) * mean_update

            h_new = self.outl2(F.softplus(self.outl1(torch.cat((cat_inpt[t], out_means[t], 1 / out_precisions[t]), -1))))
            outputs[t] = h_new

            old_precision = out_precisions[t].clone()
            old_mean = out_means[t].clone()

        # print(' output', out_means, out_precisions)
        return outputs, h_new, out_means, out_precisions