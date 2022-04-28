import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMu2(nn.Module):

    def __init__(self, input_size, hidden_size, mid_size, sparse=False, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        concat_size = input_size + hidden_size + hidden_size
        z_size = input_size + mid_size

        self.linear_z = nn.Linear(in_features=concat_size, out_features=mid_size)
        self.linear_s = nn.Linear(in_features=z_size, out_features=hidden_size)
        self.linear_m = nn.Linear(in_features=z_size, out_features=hidden_size)

        self.f = self.sparse_gate if sparse else torch.sigmoid

    def sparse_gate(self, x):
        return F.softplus(x) * nn.Softmax(1)(x)

    def forward(self, x, old_m, old_s):

        '''print('  metaMu')
        print('    x:', x.shape) #  [1, 16, 16] / [60, 16, 16]
        print('    old_m:', old_m.shape) #  [16, 64] / [25, 64]
        print('    old_s:', old_s.shape)'''
        # stime = time.time()

        if len(old_m.shape) == 3:
            old_m = old_m.squeeze(0)  # add sequence len dimension
            # print('    squeeze old_m:', old_m.shape)
        if len(old_s.shape) == 3:
            old_s = old_s.squeeze(0)  # add sequence len dimension
            # print('    squeeze old_s:', old_s.shape)

        seq_len = x.shape[0]
        batch = x.shape[1]

        out_m = torch.zeros((seq_len, batch, self.hidden_size), device=self.device)
        out_s = torch.zeros((seq_len, batch, self.hidden_size), device=self.device)

        current_m = old_m
        current_s = old_s
        for t in range(seq_len):
            # print("t", t, "/ ", seq_len, t - 1)
            '''print(f'     seq {t}')
            print('      x:', x[t].shape)  # [1, 16, 16] / [60, 16, 16]
            print('      current_m:', current_m.shape)
            print('      current_s:', current_s.shape)'''

            z_inpt = torch.cat((x[t], current_m, 1/current_s), -1)
            z = torch.cat((x[t], torch.tanh(self.linear_z(z_inpt))), -1)

            out_s[t] = current_s + self.f(self.linear_s(z)) # * 0.75
            m_gate = (current_s / out_s[t]).detach()
            out_m[t] = m_gate * current_m + (1 - m_gate) * torch.tanh(self.linear_m(z))

            current_m = out_m[t].clone()
            current_s = out_s[t].clone()

        # print('     output', out_means, out_precisions)
        # print('     exc time ', round(time.time() - stime, 5))
        return out_m, out_s
