"""Checkpoint checks through the production JAXAgent wrapper.

These checks include the host RNG and update counter used by real training,
which a parameter-only Ninjax serialization check cannot exercise.
"""

import copy
import os
import pathlib
import pickle
import sys
import unittest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from recall2imagine import agent as agentlib
import embodied
from test_integration import config, spaces


def host_batch(cfg):
  rng = np.random.default_rng(37)
  shape = (cfg.batch_size, cfg.batch_length)
  first = np.zeros(shape, bool)
  first[:, 0] = True
  terminal = np.zeros(shape, bool)
  terminal[:, -1] = True
  return dict(
      observation=rng.normal(size=shape + (5,)).astype(np.float32),
      action=np.eye(3, dtype=np.float32)[rng.integers(3, size=shape)],
      reward=rng.normal(size=shape).astype(np.float32),
      is_first=first, is_last=terminal, is_terminal=terminal,
      reset=np.zeros(shape, bool))


class WrappedAgentCheckpointTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    previous_guard = jax.config.jax_transfer_guard
    try:
      # Negative Every interval emits metrics on every call independently of
      # its host logging cursor, which is unrelated to numerical checkpoints.
      cls.cfg = config(**{'jax.metrics_every': -1})
      obs_space, act_space = spaces()
      cls.agent = agentlib.Agent(
          obs_space, act_space, embodied.Counter(), cls.cfg)
      # Real wrapper.train consumes batches already placed by its dataset.
      cls.data = jax.device_put(host_batch(cls.cfg), cls.agent.train_devices[0])
      cls.initial_checkpoint = cls.agent.save()
    finally:
      jax.config.update('jax_transfer_guard', previous_guard)

  def setUp(self):
    self.previous_guard = jax.config.jax_transfer_guard
    jax.config.update('jax_transfer_guard', 'disallow')
    self.agent.load(copy.deepcopy(self.initial_checkpoint))

  def tearDown(self):
    # The wrapper changes this process-wide option. Other suites must retain
    # their original transfer policy even when a checkpoint assertion fails.
    jax.config.update('jax_transfer_guard', self.previous_guard)

  def assert_tree_equal(self, actual, expected):
    actual = jax.device_get(actual)
    expected = jax.device_get(expected)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    self.assertEqual(actual_tree, expected_tree)
    for left, right in zip(actual_leaves, expected_leaves):
      np.testing.assert_array_equal(left, right)

  def test_wrapped_save_load_repeats_next_update(self):
    _, state, metrics = self.agent.train(self.data)
    self.assertTrue(np.isfinite(metrics['model_loss_mean']))
    # Pickle mirrors the checkpoint transport used by embodied.Checkpoint.
    saved = pickle.loads(pickle.dumps(self.agent.save()))
    self.assertEqual(saved['format'], 'ctwm-r2i-v1')
    self.assertEqual(saved['updates'], 1)

    _, expected_state, expected_metrics = self.agent.train(self.data, state)
    expected = self.agent.save()
    self.agent.load(saved)
    self.assertEqual(int(self.agent._updates), 1)
    self.assertEqual(self.agent.rng.bit_generator.state, saved['rng_state'])
    _, repeated_state, repeated_metrics = self.agent.train(self.data, state)
    repeated = self.agent.save()
    self.assertEqual(repeated['updates'], expected['updates'])
    self.assertEqual(repeated['rng_state'], expected['rng_state'])
    self.assert_tree_equal(repeated['varibs'], expected['varibs'])
    self.assert_tree_equal(repeated_state, expected_state)
    self.assert_tree_equal(repeated_metrics, expected_metrics)

  def test_invalid_checkpoint_is_rejected_without_changing_runtime(self):
    good = self.agent.save()
    leaves, structure = jax.tree_util.tree_flatten(good['varibs'])
    for kind in ('architecture', 'legacy_schema', 'shape', 'dtype'):
      with self.subTest(kind=kind):
        invalid = copy.deepcopy(good)
        # Distinct metadata makes premature RNG/counter writes observable.
        invalid['rng_state'] = np.random.default_rng(918).bit_generator.state
        invalid['updates'] = good['updates'] + 100
        if kind == 'architecture':
          invalid['architecture']['nrsm']['ticks'] += 1
          message = 'architecture'
        elif kind == 'legacy_schema':
          invalid['architecture'].pop('nrsm_state_schema')
          message = 'architecture'
        else:
          altered = list(leaves)
          index = next(i for i, leaf in enumerate(altered)
                       if np.issubdtype(leaf.dtype, np.floating))
          if kind == 'shape':
            altered[index] = np.zeros(altered[index].shape + (1,),
                                      altered[index].dtype)
          else:
            altered[index] = altered[index].astype(np.float64)
          invalid['varibs'] = jax.tree_util.tree_unflatten(structure, altered)
          message = 'shapes or dtypes'
        before = self.agent.save()
        with self.assertRaisesRegex(ValueError, message):
          self.agent.load(invalid)
        after = self.agent.save()
        self.assertEqual(after['rng_state'], before['rng_state'])
        self.assertEqual(after['updates'], before['updates'])
        self.assert_tree_equal(after['varibs'], before['varibs'])


if __name__ == '__main__':
  unittest.main()
