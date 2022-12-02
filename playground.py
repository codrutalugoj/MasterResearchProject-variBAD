import time

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")


def main_numpy():
    rewards = torch.ones((2, 4))

    eps = 0.5  # probability of receiving a random perceptual state
    condition = torch.tensor(np.random.choice([0, 1], p=[eps, 1 - eps], size=rewards.shape),
                             dtype=torch.bool, device=device)
    # print('encoder rew cond', condition.shape)
    rnd_rewards = torch.tensor(np.random.choice([5, 6], p=[0.5, 0.5], size=rewards.shape),
                               dtype=torch.float, device=device)
    rewards = torch.where(condition, rewards, rnd_rewards)

    print(rewards)


def main_torch():
    rewards = torch.ones((2, 4, 1))
    # rewards is seq, batch, feature -> but feature dim is 1, hence we can 'ignore' it

    eps = 0.9  # probability of randomized perceptual state
    prob_rew_perception_p = torch.tensor([eps, 1 - eps], device=device)
    prob_rew_rnd_rew_p = torch.tensor([.5, .5], device=device)

    rewards = rewards.squeeze(-1)
    rew_seqlen = rewards.shape[0]
    rew_batch = rewards.shape[1]
    condition = prob_rew_perception_p.multinomial(rew_seqlen * rew_batch, replacement=True).view(rew_seqlen, rew_batch)

    rnd_rewards_idxs = prob_rew_rnd_rew_p.multinomial(rew_seqlen * rew_batch, replacement=True).view(rew_seqlen, rew_batch)
    rnd_rewards = torch.where(rnd_rewards_idxs.bool(), 5.0, 6.0)
    rewards = torch.where(condition.bool(), rewards, rnd_rewards)  # .unsqueeze(-1)

    print('rewards\n', rewards)


if __name__ == "__main__":
    s = time.time()
    main_numpy()
    print(time.time() - s)
    print()

    s = time.time()
    main_torch()
    print(time.time() - s)
