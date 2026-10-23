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
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
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
      cls.cfg = config(**{
          'jax.metrics_every': -1,
          # Metadata must describe resolved inputs, not these task overrides.
          'actor.inputs': ['hidden'], 'critic.inputs': ['hidden'],
          'decoder.inputs': ['ltm'], 'reward_head.inputs': ['ltm'],
          'cont_head.inputs': ['ltm'],
      })
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
    architecture = saved['architecture']
    self.assertEqual(architecture['nrsm_state_schema'],
                     'ctm-action-conditioned-reward-v12')
    self.assertEqual(architecture['nrsm_memory_contract'], {
        'kind': 'persistent_ctm_activity_with_independent_write_gate',
        'state_keys': ['deter', 'stoch', 'logit', 'pre_trace', 'post_trace'],
        'activity_storage': 'post_trace[..., -1]',
        'trace_order': 'oldest_to_newest_thought_ticks',
        'episode_reset': 'zero_all_ctm_state_and_incoming_action',
        'cross_step_activity': 'nu[t,0]=nu[t-1,K]',
        'update': 'rho*previous_activity+write_scale*kappa*write_gate*nlm_candidate',
        'rho': 'exp(-1/(ticks*time_constant))',
        'kappa': 'expm1(-1/(ticks*time_constant))/expm1(-1/time_constant)',
        'write_scale': self.cfg.nrsm.write_scale,
        'write_gate': {
            'input': 'private_nlm_elu_hidden',
            'output': 'sigmoid', 'parameter_scope': 'private_per_neuron',
            'weights_initialization': 'zeros', 'bias_initialization': 'zeros',
            'initial_value': 0.5,
        },
        'candidate_activation': 'tanh',
        'constant_gate_and_candidate_environment_step_write': 'write_scale*write_gate*nlm_candidate',
        'time_constant_parameterization': 'bounded_log_sigmoid_per_neuron',
        'time_constant_unit': 'environment_steps',
        'time_constant_min': self.cfg.nrsm.time_constant_min,
        'time_constant_max': self.cfg.nrsm.time_constant_max,
        'time_constant_initialization': 'log_spaced_1pct_to_99pct_bounds',
        'activity_layer_norm': False,
        'extra_ssm_state': False,
        'posterior_memory_correction': False,
    })
    self.assertEqual(architecture['nrsm_synapse_contract'], {
        'kind': 'ctm_navigation_glu_layernorm', 'blocks': 2,
        'linear_width': 2 * self.cfg.nrsm.neurons,
        'post_glu_width': self.cfg.nrsm.neurons,
        'order': ['linear', 'glu', 'layer_norm'],
        'layer_norm': {'axis': -1, 'epsilon': 1e-5,
                       'elementwise_affine': True, 'variance': 'biased'},
        'linear_init': 'fan_in_uniform', 'dropout': 0.0,
    })
    contract = architecture['nrsm_readout_contract']
    self.assertEqual(contract['prediction_inputs'], {
        name: ['deter', 'stoch'] for name in ('decoder', 'reward', 'cont')})
    self.assertEqual(contract['actor_inputs'], ['deter', 'stoch'])
    self.assertEqual(contract['critic_inputs'], ['deter', 'stoch'])
    self.assertEqual(contract['deter_concatenation'], {
        'order': ['current_activity', 'projected_activity_and_synchrony'],
        'raw_activity_width': self.cfg.nrsm.neurons,
        'raw_activity_transform': 'identity', 'total_width': self.cfg.nrsm.deter,
    })
    selected = np.rint(np.linspace(
        0, self.cfg.nrsm.neurons - 1, self.cfg.nrsm.sync_neurons)).astype(int)
    self.assertEqual(contract['synchronization'], {
        'index_selection': 'rint(linspace(0,neurons-1,sync_neurons))',
        'indices': selected.tolist(),
        'pairs': [[int(i), int(j)] for offset, i in enumerate(selected)
                  for j in selected[offset:]],
        'pair_order': 'upper_triangle_with_diagonal_in_selected_index_order',
        'indices_remain_fixed_during_learning': True,
    })
    self.assertEqual(contract['deter_projection'], {
        'input_order': ['current_activity', 'window_synchrony'],
        'input_width': self.cfg.nrsm.neurons + (
            self.cfg.nrsm.sync_neurons * (self.cfg.nrsm.sync_neurons + 1) // 2),
        'output_width': self.cfg.nrsm.deter - self.cfg.nrsm.neurons,
        'activation': 'tanh',
    })
    self.assertFalse(contract['direct_extra_memory_bypass'])
    self.assertFalse(contract['direct_action_concatenation'])
    self.assertNotIn('direct_memory_or_action_bypass', contract)
    self.assertNotIn('ssm_hidden', contract)
    self.assertNotIn('reward_action', contract)
    self.assertEqual(contract['reward_first_layer'], {
        'kind': 'action_conditioned_activity_readout',
        'activity': 'deter[..., :neurons]',
        'shared': 'concat(deter[..., neurons:], flatten(stoch))',
        'discrete_weights': 'incoming_action_onehot',
        'continuous_weights': 'concat(ones,incoming_action_flat)',
        'action_width': 3, 'action_discrete': True,
        'action_processing': 'rssm._action_flatten_and_clip',
        'affine': 'shared@U+einsum(activity,W,action_weights)+bias_if_no_norm',
        'norm_activation_and_remaining_layers': 'unchanged_reward_mlp',
        'initialization_fan': 'original_activity_plus_shared_width',
        'distribution_and_loss': 'unchanged',
        'training_action': 'concat(carry_action,replay_action[:,:-1]); zero_on_is_first',
        'imagination_action': 'concat(zero,trajectory.action[:-1])',
        'ephemeral_key': 'reward_action',
        'ephemeral_key_in_recurrent_state': False,
    })
    self.assertNotIn('nrsm_memory_backbone_contract', architecture)
    for removed in ('ssm_modes', 'ssm_dt_min', 'ssm_dt_max', 'ltm_slots', 'memory_heads'):
      self.assertNotIn(removed, architecture['nrsm'])
    self.assertNotEqual(contract['actor_inputs'], list(self.cfg.actor.inputs))
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
    changes = [
        (('nrsm', 'ticks'), self.cfg.nrsm.ticks + 1),
        (('nrsm', 'time_constant_min'), self.cfg.nrsm.time_constant_min * 2),
        (('nrsm', 'time_constant_max'), self.cfg.nrsm.time_constant_max * 2),
        (('nrsm', 'write_scale'), self.cfg.nrsm.write_scale * 2),
        (('nrsm', 'ssm_modes'), 64),
        (('nrsm_memory_contract', 'activity_storage'), 'separate_ssm_hidden'),
        (('nrsm_memory_contract', 'time_constant_unit'), 'thought_ticks'),
        (('nrsm_memory_contract', 'rho'), 'exp(-1/time_constant)'),
        (('nrsm_memory_contract', 'kappa'), '1-rho'),
        (('nrsm_memory_contract', 'write_scale'), self.cfg.nrsm.write_scale * 2),
        (('nrsm_memory_contract', 'write_gate', 'output'), 'tanh'),
        (('nrsm_memory_contract', 'write_gate', 'input'), 'time_constant_raw'),
        (('nrsm_memory_contract', 'write_gate', 'weights_initialization'), 'uniform'),
        (('nrsm_memory_contract', 'write_gate', 'initial_value'), 0.1),
        (('nrsm_memory_contract', 'constant_gate_and_candidate_environment_step_write'),
         '(1-rho**ticks)*nlm_candidate'),
        (('nrsm_memory_contract', 'activity_layer_norm'), True),
        (('nrsm_memory_contract', 'extra_ssm_state'), True),
        (('nrsm_memory_contract', 'posterior_memory_correction'), True),
        (('nrsm_memory_contract', 'state_keys'), ['deter', 'stoch', 'ltm']),
        (('nrsm_readout_contract', 'actor_inputs'), ['deter', 'stoch', 'ltm']),
        (('nrsm_readout_contract', 'critic_inputs'), ['deter', 'stoch', 'ltm']),
        (('nrsm_readout_contract', 'prediction_inputs', 'reward'),
         ['deter', 'stoch', 'reward_action']),
        (('nrsm_readout_contract', 'direct_extra_memory_bypass'), True),
        (('nrsm_readout_contract', 'direct_action_concatenation'), True),
        (('nrsm_readout_contract', 'reward_first_layer', 'training_action'), 'outgoing_action'),
        (('nrsm_readout_contract', 'reward_first_layer', 'imagination_action'), 'trajectory.action'),
        (('nrsm_readout_contract', 'reward_first_layer', 'action_discrete'), False),
        (('nrsm_readout_contract', 'reward_first_layer', 'ephemeral_key_in_recurrent_state'), True),
        (('nrsm_readout_contract', 'deter_projection', 'input_order'), ['window_synchrony']),
        (('nrsm_readout_contract', 'deter_projection', 'input_width'), 1),
        (('nrsm_readout_contract', 'deter_projection', 'output_width'), self.cfg.nrsm.deter),
        (('nrsm_readout_contract', 'deter_concatenation', 'order'),
         ['projected_activity_and_synchrony', 'current_activity']),
        (('nrsm_readout_contract', 'deter_concatenation', 'raw_activity_transform'), 'tanh'),
        (('nrsm_readout_contract', 'synchronization', 'indices'),
         list(range(self.cfg.nrsm.sync_neurons))),
        (('nrsm_readout_contract', 'synchronization', 'pairs'), [[0, 0]]),
        (('nrsm_readout_contract', 'synchronization', 'indices_remain_fixed_during_learning'), False),
        (('nrsm_synapse_contract', 'order'), ['linear', 'layer_norm', 'glu']),
        (('nrsm_synapse_contract', 'post_glu_width'), 1),
        (('nrsm_synapse_contract', 'layer_norm', 'epsilon'), 1e-3),
        (('nrsm_synapse_contract', 'linear_init'), 'glorot_uniform_zero_bias'),
    ]
    for schema in (
        'ctm-gated-write-activity-sync-v11',
        'ctm-timescale-activity-sync-v10',
        'slow-ssm-full-hidden-glu-ln-ac-hidden-reward-action-v9',
        'slow-ssm-full-hidden-glu-ln-ac-hidden-v8',
        'slow-ssm-ctm-glu-ln-ac-hidden-v7',
        'slow-ssm-direct-ctm-ac-hidden-v6', 'slow-ssm-direct-ctm-v5',
        'slow-ssm-thought-read-v4', 'persistent-traces-ltm-v3',
        'persistent-traces-v2', 'future-unknown-schema'):
      changes.append((('nrsm_state_schema',), schema))
    for missing in ('nrsm_state_schema', 'nrsm_memory_contract',
                    'nrsm_synapse_contract', 'nrsm_readout_contract'):
      changes.append(((missing,), None))
    cases = [(path, value, 'architecture') for path, value in changes]
    cases.extend([(('shape',), None, 'shapes or dtypes'),
                  (('dtype',), None, 'shapes or dtypes'),
                  (('updates',), -1, 'nonnegative')])
    for path, value, message in cases:
      with self.subTest(path=path, value=value):
        invalid = copy.deepcopy(good)
        # Metadata mismatch must be detected before changing weights, RNG or
        # update count. Incompatible v10 metadata is rejected before any
        # shape validation or restoration; no silent partial loading occurs.
        invalid['rng_state'] = np.random.default_rng(918).bit_generator.state
        invalid['updates'] = good['updates'] + 100
        if path[0] in ('shape', 'dtype'):
          altered = list(leaves)
          index = next(i for i, leaf in enumerate(altered)
                       if np.issubdtype(leaf.dtype, np.floating))
          altered[index] = (
              np.zeros(altered[index].shape + (1,), altered[index].dtype)
              if path[0] == 'shape' else altered[index].astype(np.float64))
          invalid['varibs'] = jax.tree_util.tree_unflatten(structure, altered)
        elif path[0] == 'updates':
          invalid['updates'] = value
        else:
          target = invalid['architecture']
          for field in path[:-1]:
            target = target[field]
          if value is None:
            target.pop(path[-1])
          else:
            target[path[-1]] = value
        before = self.agent.save()
        with self.assertRaisesRegex(ValueError, message):
          self.agent.load(invalid)
        after = self.agent.save()
        self.assertEqual(after['rng_state'], before['rng_state'])
        self.assertEqual(after['updates'], before['updates'])
        self.assert_tree_equal(after['varibs'], before['varibs'])

  def test_unversioned_nrsm_parameters_are_rejected_even_with_equal_shapes(self):
    before = self.agent.save()
    with self.assertRaisesRegex(ValueError, 'require versioned architecture metadata'):
      self.agent.load(copy.deepcopy(before['varibs']))
    after = self.agent.save()
    self.assertEqual(after['rng_state'], before['rng_state'])
    self.assertEqual(after['updates'], before['updates'])
    self.assert_tree_equal(after['varibs'], before['varibs'])

  def test_saved_architecture_is_detached_and_mutated_snapshot_is_rejected(self):
    before = self.agent.save()
    architecture = copy.deepcopy(self.agent._architecture)
    snapshot = self.agent.save()
    # Use the raw save result: consumers may edit checkpoint metadata without
    # first copying it, including dictionaries nested below the architecture.
    snapshot['architecture']['nrsm']['time_constant_max'] *= 2
    self.assertEqual(self.agent._architecture, architecture)
    snapshot['rng_state'] = np.random.default_rng(719).bit_generator.state
    snapshot['updates'] += 100
    with self.assertRaisesRegex(ValueError, 'architecture'):
      self.agent.load(snapshot)
    after = self.agent.save()
    self.assertEqual(self.agent._architecture, architecture)
    self.assertEqual(after['rng_state'], before['rng_state'])
    self.assertEqual(after['updates'], before['updates'])
    self.assert_tree_equal(after['varibs'], before['varibs'])


if __name__ == '__main__':
  unittest.main()
