import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMu(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, mid_size, sparse=False, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.device = device
        concat_size = input_size + hidden_size * 2

        self.linear_s = nn.Linear(in_features=input_size + hidden_size, out_features=latent_dim)  # Ws
        self.linear_m = nn.Linear(in_features=input_size + hidden_size, out_features=latent_dim)  # Wm
        self.linear_z = nn.Linear(in_features=concat_size, out_features=hidden_size)
        '''self.outl1 = nn.Linear(in_features=concat_size + 2 * latent_dim,
                               out_features=mid_size)
        self.outl2 = nn.Linear(in_features=mid_size,
                               out_features=hidden_size)'''
        self.gate_f = self.sparse_gate if sparse else torch.sigmoid

    def sparse_gate(self, x):
        # return torch.sigmoid(x + bias) * nn.Softmax(1)(x)
        return F.softplus(x) * nn.Softmax(1)(x)

    def forward(self, x, old_mean, old_precision):
        """
        if mean_old and precision_old is None:
            generate a prior here.
        """

        '''print('   metaMu')
        print('     x:', x.shape) #  [1, 16, 16] / [60, 16, 16]
        print('     h_old:', h_old.shape) #  [1, 16, 64] / [1, 16, 64] # 64=gru_hidden_size
        print('     mean_old:', old_mean.shape)
        print('     prec._old:', old_precision.shape)
        stime = time.time()'''

        if len(old_mean.shape) == 2:
            old_mean = old_mean.unsqueeze(0)  # add sequence len dimension
            # print('  mean_old:', mean_old.shape)
        if len(old_precision.shape) == 2:
            old_precision = old_precision.unsqueeze(0)  # add sequence len dimension
            # print('  prec._old:', precision_old.shape)

        seq_len = x.shape[0]
        batch = x.shape[1]

        out_means = torch.zeros((seq_len, batch, self.latent_dim), device=self.device)
        out_precisions = torch.zeros((seq_len, batch, self.latent_dim), device=self.device)

        for t in range(seq_len):
            cat_input = torch.cat((x[t].unsqueeze(0) if len(old_mean.shape) == 3 else x[t], old_mean, old_precision), -1)
            z = torch.tanh(self.linear_z(cat_input))
            z = torch.cat((x[t].unsqueeze(0) if len(z.shape) == 3 else x[t], z), -1)

            precision_update = self.gate_f(self.linear_s(z))
            out_precisions[t] = old_precision + precision_update
            # print("New precision ", out_precisions[t])

            mean_gate = (old_precision / out_precisions[t]).detach()  # lambda
            mean_update = torch.tanh(self.linear_m(z))

            out_means[t] = mean_gate * old_mean + (1 - mean_gate) * mean_update

            # TODO: Could this cause a problem through call-by-reference?
            old_precision = out_precisions[t].clone()
            old_mean = out_means[t].clone()

        # print('     output', out_means, out_precisions)
        # print('     exc time ', round(time.time() - stime, 5))
        return out_means, out_precisions