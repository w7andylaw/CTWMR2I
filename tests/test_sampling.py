"""Reproducible prefill actions through the production space/wrapper API."""

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recall2imagine'))
import embodied
from embodied import wrappers


class ActionEnv(embodied.Env):

  def __init__(self, discrete):
    self.discrete = discrete

  @property
  def obs_space(self):
    return {'observation': embodied.Space(np.float32, (2,), -1, 1)}

  @property
  def act_space(self):
    action = (embodied.Space(np.int32, (), 0, 5) if self.discrete else
              embodied.Space(np.float32, (3,), -1, 1))
    return {'action': action, 'reset': embodied.Space(bool)}


def random_agent(discrete, seed):
  env = ActionEnv(discrete)
  if discrete:
    env = wrappers.OneHotAction(env)
  return embodied.RandomAgent(env.act_space, seed=seed)


def sample(agent):
  obs = {'observation': np.zeros((32, 2), np.float32)}
  return agent.policy(obs)[0]['action']


class SamplingTests(unittest.TestCase):

  def test_discrete_random_agents_repeat_with_same_seed(self):
    first, second = random_agent(True, 17), random_agent(True, 17)
    for _ in range(3):
      actual = sample(first)
      np.testing.assert_array_equal(actual, sample(second))
      self.assertEqual(actual.dtype, np.float32)
      self.assertEqual(actual.shape, (32, 5))
      np.testing.assert_array_equal(actual.sum(-1), np.ones(32))
      self.assertTrue(np.isin(actual, [0, 1]).all())

  def test_continuous_random_agents_repeat_with_same_seed(self):
    first, second = random_agent(False, 17), random_agent(False, 17)
    for _ in range(3):
      actual = sample(first)
      np.testing.assert_array_equal(actual, sample(second))
      self.assertEqual(actual.dtype, np.float32)
      self.assertTrue(((actual >= -1) & (actual <= 1)).all())

  def test_different_seeds_change_both_action_types(self):
    for discrete in [True, False]:
      with self.subTest(discrete=discrete):
        self.assertFalse(np.array_equal(
            sample(random_agent(discrete, 17)),
            sample(random_agent(discrete, 18))))

  def test_onehot_space_reseed_and_global_rng_independence(self):
    space = wrappers.OneHotAction(ActionEnv(True)).act_space['action']
    original = np.random.get_state()
    try:
      space.seed(23)
      expected = np.stack([space.sample() for _ in range(32)])
      np.random.seed(999)
      space.seed(23)
      np.testing.assert_array_equal(
          np.stack([space.sample() for _ in range(32)]), expected)
    finally:
      np.random.set_state(original)


if __name__ == '__main__':
  unittest.main()
