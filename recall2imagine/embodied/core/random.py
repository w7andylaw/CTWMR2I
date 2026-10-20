import numpy as np


class RandomAgent:

  def __init__(self, act_space, seed=None):
    self.act_space = act_space
    if seed is not None:
      for index, space in enumerate(self.act_space.values()):
        space.seed(int(seed) + index)

  def policy(self, obs, state=None, mode='train'):
    batch_size = len(next(iter(obs.values())))
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return act, state
