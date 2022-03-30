import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMu(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, mid_size=10, sparse=False, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.device = device
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
        """
        if mean_old and precision_old is None:
            generate a prior here.
        """
        print('   metaMu')
        print('     x:', x.shape) #  [1, 16, 16] / [60, 16, 16]
        print('     h_old:', h_old.shape) #  [1, 16, 64] / [1, 16, 64] # 64=gru_hidden_size
        print('     mean_old:', old_mean.shape)
        print('     prec._old:', old_precision.shape)
        stime = time.time()

        if len(old_mean.shape) == 2:
            old_mean = old_mean.unsqueeze(0)  # add sequence len dimension
            # print('  mean_old:', mean_old.shape)
        if len(old_precision.shape) == 2:
            old_precision = old_precision.unsqueeze(0)  # add sequence len dimension
            # print('  prec._old:', precision_old.shape)

        seq_len = x.shape[0]
        batch = x.shape[1]

        outputs = torch.zeros((seq_len, batch, self.hidden_size), device=self.device)
        out_means = torch.zeros((seq_len, batch, self.latent_dim), device=self.device)
        out_precisions = torch.zeros((seq_len, batch, self.latent_dim), device=self.device)

        h_new = h_old[0]
        for t in range(seq_len):
            cat_inpt = torch.cat((x[t], h_new), -1)
            print('       cat_input', cat_inpt.shape)
            precision_update_gate = self.gate_f(self.linear_i(cat_inpt))
            precision_update = F.softplus(self.linear_c(cat_inpt))
            print('       prec update gate', precision_update_gate.shape)
            print('       prec update', precision_update.shape)
            out_precisions[t] = old_precision + precision_update_gate * precision_update
            # print(' out prec', out_precisions[t])

            mean_gate = (old_precision / out_precisions[t]).detach()  # lambda
            mean_update = F.tanh(self.linear_g(cat_inpt))
            out_means[t] = mean_gate * old_mean + (1 - mean_gate) * mean_update

            # TODO: replace softplus with tanh
            h_new = self.outl2(F.tanh(self.outl1(torch.cat((cat_inpt, out_means[t], 1 / out_precisions[t]), -1))))
            outputs[t] = h_new

            # TODO: Could this cause a problem through call-by-reference?
            old_precision = out_precisions[t].clone()
            old_mean = out_means[t].clone()

        # print('     output', out_means, out_precisions)
        print('     exc time ', round(time.time() - stime, 5))
        return outputs, h_new, out_means, out_precisions