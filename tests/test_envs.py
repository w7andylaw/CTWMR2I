"""Native environment contracts: reward, observation, reset and continuation."""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import gymnasium as gym

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recall2imagine'))
from embodied.envs.bsuite import BSuite
from embodied.envs.from_dm import FromDM
from embodied.envs.from_gym import FromGym, MyObsPopgym, flatten, unmap_action


class ShortGym(gym.Env):
  observation_space = gym.spaces.Box(-1, 1, (2,), np.float32)
  action_space = gym.spaces.Discrete(2)

  def __init__(self, terminal=False, legacy=False):
    self.terminal, self.legacy = terminal, legacy
    self.seeds = []

  def reset(self, *, seed=None, options=None):
    self.seeds.append(seed)
    self.count = 0
    obs = np.zeros(2, np.float32)
    return obs if self.legacy else (obs, {'reset': True})

  def step(self, action):
    self.count += 1
    done = self.count == 2
    obs = np.full(2, self.count / 2, np.float32)
    if self.legacy:
      return obs, 0.375, done, {'TimeLimit.truncated': done and not self.terminal}
    return obs, 0.375, done and self.terminal, done and not self.terminal, {}


class EnvironmentTests(unittest.TestCase):

  def test_memory_matches_native_environment(self):
    from bsuite.experiments.memory_len.memory_len import load
    bridge, native = BSuite('memory_len/4', seed=19), load(4, seed=19)
    first = bridge.step({'action': 0, 'reset': True})
    np.testing.assert_equal(first['observation'], native.reset().observation.flatten())
    self.assertTrue(first['is_first'])
    self.assertFalse(first['is_terminal'])
    self.assertEqual(first['reward'], 0)
    for action in [0, 1, 0, 1, 1]:
      expected = native.step(action)
      actual = bridge.step({'action': action, 'reset': False})
      np.testing.assert_equal(actual['observation'], expected.observation.flatten())
      self.assertEqual(actual['reward'], expected.reward)
      self.assertEqual(actual['is_last'], expected.last())
      self.assertEqual(actual['is_terminal'], expected.discount == 0)
    again = bridge.step({'action': 1, 'reset': False})
    self.assertTrue(again['is_first'])
    self.assertEqual(again['reward'], 0)

  def test_discounting_chain_keeps_native_delays_rewards_and_termination(self):
    for delay in BSuite.DELAYS:
      env = BSuite(f'discounting_chain/{delay}')
      obs = env.step({'action': 0, 'reset': True})
      self.assertEqual(obs['observation'].shape, (2,))
      target_action = BSuite.DELAYS.index(delay)
      rewards = []
      for step in range(1, 101):
        obs = env.step({'action': target_action, 'reset': False})
        rewards.append(obs['reward'])
        self.assertEqual(obs['is_last'], step == 100)
        self.assertEqual(obs['is_terminal'], step == 100)
      self.assertAlmostEqual(float(sum(rewards)), 1.1, places=6)
      self.assertAlmostEqual(float(rewards[delay - 1]), 1.1, places=6)
      self.assertEqual(np.count_nonzero(rewards), 1)

  def test_invalid_bsuite_tasks_fail(self):
    for task in ['memory_len/0', 'discounting_chain/4', 'missing/3', 'memory_len']:
      with self.assertRaises(ValueError):
        BSuite(task)

  def test_original_dm_bsuite_sweep_id_remains_available(self):
    env = FromDM('bsuite_memory_len/5')
    obs = env.step({'action': 0, 'reset': True})
    self.assertEqual(obs['observation'].shape, (3,))
    self.assertTrue(obs['is_first'])

  def test_training_factory_exposes_onehot_native_tasks(self):
    import embodied
    from ruamel.yaml import YAML
    from train import make_env
    defaults = YAML(typ='safe').load(
        (ROOT / 'recall2imagine' / 'configs.yaml').read_text())['defaults']
    for task, action_count in [('bsuite_memory_len/4', 2),
                               ('bsuite_discounting_chain/10', 5)]:
      config = embodied.Config(defaults).update({'task': task, 'seed': 3})
      env = make_env(config)
      try:
        self.assertEqual(env.act_space['action'].shape, (action_count,))
        action = np.zeros(action_count, np.float32)
        action[0] = 1
        obs = env.step({'action': action, 'reset': True})
        self.assertTrue(obs['is_first'])
        self.assertEqual(obs['observation'].ndim, 1)
        obs = env.step({'action': action, 'reset': False})
        self.assertFalse(obs['is_first'])
      finally:
        env.close()

  def test_gymnasium_terminal_and_truncation_are_distinct(self):
    for terminal in [False, True]:
      native = ShortGym(terminal=terminal)
      env = FromGym(native, obs_key='observation', seed=13)
      reset = env.step({'action': 0, 'reset': True})
      self.assertTrue(reset['is_first'])
      self.assertEqual(env.info, {'reset': True})
      env.step({'action': 0, 'reset': False})
      last = env.step({'action': 0, 'reset': False})
      self.assertEqual(last['reward'], 0.375)
      self.assertTrue(last['is_last'])
      self.assertEqual(last['is_terminal'], terminal)
      next_first = env.step({'action': 0, 'reset': False})
      self.assertTrue(next_first['is_first'])
      self.assertEqual(native.seeds, [13, None])

  def test_legacy_gym_truncation_is_not_terminal(self):
    env = FromGym(ShortGym(legacy=True), obs_key='observation')
    env.step({'action': 0, 'reset': True})
    env.step({'action': 0, 'reset': False})
    final = env.step({'action': 0, 'reset': False})
    self.assertTrue(final['is_last'])
    self.assertFalse(final['is_terminal'])

  def test_popgym_categorical_onehot_is_float32(self):
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(4)))
    convert, target = flatten(space)
    actual = convert((1, 2))
    np.testing.assert_array_equal(actual, [0, 1, 0, 0, 1, 0])
    self.assertEqual(actual.dtype, np.float32)
    self.assertEqual(actual.shape, target.shape)

  def test_multidiscrete_action_mapping_is_bijective(self):
    nvec = np.array([2, 3, 4])
    actions = [tuple(unmap_action(index, nvec)) for index in range(24)]
    self.assertEqual(len(set(actions)), 24)
    self.assertEqual(actions[-1], (1, 2, 3))
    with self.assertRaises(ValueError):
      unmap_action(24, nvec)

  @unittest.skipUnless(importlib.util.find_spec('popgym'), 'POPGym not installed')
  def test_all_nine_native_popgym_tasks_reset_and_step(self):
    import embodied
    from ruamel.yaml import YAML
    from train import make_env
    defaults = YAML(typ='safe').load(
        (ROOT / 'recall2imagine' / 'configs.yaml').read_text())['defaults']
    for name in ['RepeatPrevious', 'Autoencode', 'Concentration']:
      for level in ['Easy', 'Medium', 'Hard']:
        task = f'popgym:popgym-{name}{level}-v0'
        with self.subTest(task=task):
          env = FromGym(task, seed=7)
          try:
            first = env.step({'action': 0, 'reset': True})
            self.assertTrue(first['is_first'])
            self.assertEqual(first['observation'].shape, env.obs_space['observation'].shape)
            self.assertEqual(first['observation'].dtype, np.float32)
            for _ in range(3):
              obs = env.step({'action': 0, 'reset': False})
              self.assertTrue(np.isfinite(obs['reward']))
          finally:
            env.close()
          # The training factory adds OneHotAction, ExpandScalars and CountSteps;
          # exercise the same interface consumed by BatchEnv and the driver.
          config = embodied.Config(defaults).update({
              'task': f'gym_{task}', 'seed': 7, 'wrapper.checks': True})
          env = make_env(config)
          try:
            action = np.zeros(env.act_space['action'].shape, np.float32)
            action[0] = 1
            for reset in [True, False, False, False]:
              obs = env.step({'action': action, 'reset': reset})
              self.assertEqual(set(obs), set(env.obs_space))
              self.assertEqual(bool(obs['is_first']), reset)
              self.assertEqual(obs['observation'].dtype, np.float32)
          finally:
            env.close()


if __name__ == '__main__':
  unittest.main()
