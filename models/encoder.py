import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


class RNNEncoder(nn.Module):
    def __init__(self,
                 args,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(RNNEncoder, self).__init__()
        self.args = args
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian

        # embed action, state, reward
        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        # curr_input_dim=16, hidden_size=64
        self.gru = nn.GRU(input_size=curr_input_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        self.learnable_prior_vars = nn.Parameter(torch.ones(1, 1, latent_dim))

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, latent_dim)

        # probabilistic reward perception
        prop_rew_eps = 0.3  # probability of randomized perceptual state
        self.prob_rew_perception_p = torch.tensor([prop_rew_eps, 1 - prop_rew_eps], device=device)
        self.prob_rew_rnd_rew_p = torch.tensor([.5, .5], device=device)

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError  # TODO: double check this code, maybe we should use .unsqueeze(0).expand((num, *logvar.shape))
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, old_means, precision, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        if (done == 1).all():
            # print(' reset belief')
            precision = torch.ones_like(precision, device=device)
            old_means = torch.zeros_like(old_means, device=device)
        return hidden_state, old_means, precision

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)

        # TODO: here we need to get the prior precision from the network
        # precision = self.fc_logvar(h)
        # precision = F.softplus(precision)

        precision = self.learnable_prior_vars.expand((-1, batch_size, -1))

        latent_logvar = torch.log(1/precision)

        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state, precision

    def forward(self, actions, states, rewards, hidden_state, old_precision, old_means, return_prior, sample=True, detach_every=None):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # we do the action-normalisation (the the env bounds) here
        actions = utl.squash_action(actions, self.args)

        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        # Probabilist VAE reward perception
        rewards = rewards.squeeze(-1)
        rew_seqlen = rewards.shape[0]
        rew_batch = rewards.shape[1]
        condition = self.prob_rew_perception_p.multinomial(rew_seqlen * rew_batch, replacement=True).view(rew_seqlen,
                                                                                                          rew_batch)
        rnd_rewards_idxs = self.prob_rew_rnd_rew_p.multinomial(rew_seqlen * rew_batch, replacement=True).view(
            rew_seqlen, rew_batch)
        rnd_rewards = torch.where(rnd_rewards_idxs.bool(), 1.0, -0.1)
        rewards = torch.where(condition.bool(), rewards, rnd_rewards).unsqueeze(-1)

        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state, prior_precision = self.prior(actions.shape[1])
            hidden_state = prior_hidden_state.clone()
            # TODO: the log variance value should be 1
            #print("prior precision", prior_precision)
            #print("prior logvar", torch.log((1 / prior_precision)))

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)
        hs = self.state_encoder(states)
        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)
        # print(' h', h.shape)
        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        if detach_every is None:
            # GRU cell (output is outputs for each time step, hidden_state is last output)
            # h.shape: [1, 16, 16] / [60, 16, 16]
            # hidden_state.shape: [1, 16, 64] / [1, 16, 64] # 64=gru_hidden_size
            output, _ = self.gru(h, hidden_state)
            # output.shape: [1, 16, 16] / [60, 16, 16]
        else:
            output = []
            for i in range(int(np.ceil(h.shape[0] / detach_every))):
                curr_input = h[i*detach_every:i*detach_every+detach_every]  # pytorch caps if we overflow, nice
                curr_output, hidden_state = self.gru(curr_input, hidden_state)
                output.append(curr_output)
                # detach hidden state; useful for BPTT when sequences are very long
                hidden_state = hidden_state.detach()
            output = torch.cat(output, dim=0)
        gru_h = output.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        new_means = self.fc_mu(gru_h)
        residual_precision = F.softplus(self.fc_logvar(gru_h))  # * 0.05
        # print(' res._precision/new_means', residual_precision.shape, new_means.shape)

        if old_precision is None:
            new_precision = torch.cumsum(residual_precision, dim=0)
            tmp_precisions = torch.cat((prior_precision, new_precision))
            tmp_means = torch.cat((prior_mean, new_means))
            lambda_gate = tmp_precisions[:-1]/tmp_precisions[1:]
            # print("lambda gate for full trajs", lambda_gate.shape, tmp_means.shape)
            # lamda mean gate
            latent_mean = lambda_gate * tmp_means[:-1] + (1 - lambda_gate) * tmp_means[1:]
            # latent_mean = new_means
        else:
            new_precision = old_precision + residual_precision
            lambda_gate = old_precision / new_precision
            # lamda mean gate
            latent_mean = lambda_gate * old_means + (1 - lambda_gate) * new_means
            # latent_mean = new_means

        # print(new_precision, residual_precision, precision)
        latent_logvar = torch.log(1 / (new_precision))

        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean


        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((torch.log(1 / prior_precision), latent_logvar))
            new_precision = torch.cat((prior_precision, new_precision))
            output = torch.cat((prior_hidden_state, output))
            #print(prior_logvar.requires_grad, prior_mean.requires_grad)
            #print(latent_logvar.requires_grad, latent_mean.requires_grad)

        if latent_mean.shape[0] == 1:  # TODO: Do this for precision as well. Done
            latent_sample, latent_mean, latent_logvar, new_precision = \
                latent_sample[0], latent_mean[0], latent_logvar[0], new_precision[0]

        # print("enc forward out", new_precision.shape, latent_mean.shape)
        # print()
        return latent_sample, latent_mean, latent_logvar, output, new_precision
