"""CTM-only readouts and action-through-transition alignment.

All predictions and behavior share deter/stoch; reward additionally conditions
its first activity map on the incoming action. Neuronal histories remain carry.
These are connectivity/interface checks, not benchmark convergence claims.
"""

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import pathlib
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from recall2imagine import agent as agentlib
from recall2imagine import behaviors
from recall2imagine import jaxutils
from recall2imagine import ninjax as nj
from test_integration import batch_data, config, spaces


READOUT_KEYS = ('deter', 'stoch')


def latent(cfg, leading):
  result = {}
  for index, (key, trailing) in enumerate((
      ('deter', (cfg.nrsm.deter,)),
      ('stoch', (cfg.nrsm.stoch, cfg.nrsm.classes)))):
    shape = leading + trailing
    result[key] = jnp.linspace(-.7 + .1 * index, .9, np.prod(shape)).reshape(shape)
  return result


class CTMReadoutTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    jaxutils.COMPUTE_DTYPE = jnp.float32

  def test_shared_input_preserves_online_replay_and_imagined_axes(self):
    cfg = config()
    inputs = agentlib.nets.Input(READOUT_KEYS)
    for leading in ((2,), (2, 3), (3, 2)):
      with self.subTest(leading=leading):
        state = latent(cfg, leading)
        actual = inputs(state)
        expected = np.concatenate([
            np.asarray(state[key]).reshape(leading + (-1,))
            for key in READOUT_KEYS], axis=-1)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, leading + (
            cfg.nrsm.deter + cfg.nrsm.stoch * cfg.nrsm.classes,))

  def test_actual_reward_head_reads_deter_and_stoch_but_no_extra_state(self):
    # Production outscale=0 starts with no input gradient. Nonzero test-only
    # output initialization checks the actual readout's connectivity.
    cfg = config(**{'reward_head.outscale': 1.})
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='reward_shared')
    self.assertEqual(wm.config.reward_head.inputs, READOUT_KEYS)
    self.assertFalse(hasattr(wm, 'reward_features'))
    for leading in ((2,), (2, 3), (3, 2)):
      with self.subTest(leading=leading):
        state = latent(cfg, leading)
        action = jnp.broadcast_to(jnp.array([1., 0., 0.]), leading + (3,))
        predict = nj.pure(lambda features: wm.reward(features, action).logits)
        expected, params = predict({}, jax.random.PRNGKey(201), state)
        altered = dict(state, ltm=jnp.full(leading + (64, 2), 99.),
                       reward_action=jnp.full(leading + (3,), -99.),
                       action=jnp.full(leading + (3,), 999.),
                       pre_trace=jnp.full(leading + (8, 3), 999.),
                       post_trace=jnp.full(leading + (8, 3), -999.))
        actual, _ = predict(params, jax.random.PRNGKey(202), altered)
        np.testing.assert_array_equal(actual, expected)
        width = cfg.nrsm.deter - cfg.nrsm.neurons + cfg.nrsm.stoch * cfg.nrsm.classes
        kernel = next(value for key, value in params.items()
                      if key.endswith('/rew/h0/shared_kernel'))
        self.assertEqual(kernel.shape[0], width)
        banks = next(value for key, value in params.items()
                     if key.endswith('/rew/h0/action_kernel'))
        self.assertEqual(banks.shape, (3, cfg.nrsm.neurons, cfg.reward_head.units))
    state = latent(cfg, (2,))

    def objective(deter, stoch):
      return -wm.reward(dict(deter=deter, stoch=stoch),
                        jnp.array([[1., 0., 0.], [0., 1., 0.]])).log_prob(
          jnp.array([1., -1.], jnp.float32)).sum()

    pure = nj.pure(objective)
    _, params = pure({}, jax.random.PRNGKey(203), state['deter'], state['stoch'])
    gradients = jax.grad(lambda deter, stoch: pure(
        params, jax.random.PRNGKey(204), deter, stoch,
        create=False, modify=False)[0], argnums=(0, 1))(state['deter'], state['stoch'])
    for name, value in zip(READOUT_KEYS, gradients):
      self.assertTrue(np.all(np.isfinite(np.asarray(value))), name)
      self.assertGreater(float(jnp.linalg.norm(value)), 1e-8, name)
    # The conditioned reward MLP can use both parts of the deter layout.
    # This checks connectivity, not that a trained head learns to use either.
    raw_gradient = gradients[0][..., :cfg.nrsm.neurons]
    projected_gradient = gradients[0][..., cfg.nrsm.neurons:]
    self.assertGreater(float(jnp.linalg.norm(raw_gradient)), 1e-8)
    self.assertGreater(float(jnp.linalg.norm(projected_gradient)), 1e-8)

  def test_training_transitions_use_carry_and_shifted_replay_actions(self):
    cfg = config(**{'batch_length': 4, 'nrsm.ticks': 1})
    obs, act = spaces(False)
    wm = agentlib.WorldModel(obs, act, cfg, name='transition_training_alignment')
    data = batch_data(cfg, False)
    data['cont'] = 1. - data['is_terminal'].astype(jnp.float32)
    data['action'] = jnp.arange(16, dtype=jnp.float32).reshape((2, 4, 2)) / 20.
    data['is_first'] = jnp.array([[True, False, True, False],
                                 [False, False, False, True]])
    state, params = nj.pure(wm.initial)({}, jax.random.PRNGKey(205), 2)
    incoming = jnp.array([[.91, -.82], [-.73, .64]])
    state = state[0], incoming
    captured = {}
    observe = wm.rssm.observe_with_ticks
    reward = wm.heads['reward']

    def capture_observe(embed, actions, first, previous):
      captured['actions'] = actions
      captured['first'] = first
      return observe(embed, actions, first, previous)

    def capture_reward(features, action):
      captured['features'] = features
      captured['reward_actions'] = action
      return reward(features, action)

    wm.rssm.observe_with_ticks = capture_observe
    wm.heads['reward'] = capture_reward
    (_, (next_state, outs, _)), _ = nj.pure(wm.loss)(
        params, jax.random.PRNGKey(206), data, state)
    expected = np.concatenate((np.asarray(incoming)[:, None],
                               np.asarray(data['action'])[:, :-1]), axis=1)
    np.testing.assert_array_equal(captured['actions'], expected)
    masked = expected * (1. - np.asarray(data['is_first']))[..., None]
    np.testing.assert_array_equal(captured['reward_actions'], masked)
    np.testing.assert_array_equal(captured['first'], data['is_first'])
    np.testing.assert_array_equal(next_state[1], data['action'][:, -1])
    for key in READOUT_KEYS:
      np.testing.assert_array_equal(captured['features'][key], outs['post'][key])
    np.testing.assert_array_equal(
        captured['features']['deter'][..., :cfg.nrsm.neurons],
        outs['post']['post_trace'][..., -1])
    for removed in ('ltm', 'reward_action'):
      self.assertNotIn(removed, captured['features'])
      self.assertNotIn(removed, outs['post'])
      self.assertNotIn(removed, next_state[0])

  def test_imagination_and_both_behaviors_read_the_action_result_state(self):
    cfg = config(**{'expl_rewards.disag': 0.})
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='reward_imagination_alignment')

    class ActionCore:
      state_keys = READOUT_KEYS
      _action = staticmethod(lambda action, stoch: action)

      def img_step(self, previous, action):
        if set(previous) != set(self.state_keys):
          raise AssertionError('Unexpected metadata entered recurrent state.')
        influence = (action * jnp.array([1., 3., 7.])).sum(-1, keepdims=True)
        return dict(previous, deter=previous['deter'] + influence)

    class Continuation:
      def __init__(self, features):
        self.features = features

      def mode(self):
        return jnp.ones(self.features['deter'].shape[:-1])

    class StateReward:
      def __init__(self, features, incoming_action):
        self.features = features
        np.testing.assert_array_equal(incoming_action, features['reward_action'])

      def mean(self):
        return self.features['deter'][..., 0]

    wm.rssm = ActionCore()
    wm.heads['cont'] = Continuation
    wm.heads['reward'] = StateReward
    start = latent(cfg, (2,))
    start['deter'] = jnp.zeros_like(start['deter'])
    start['is_terminal'] = jnp.zeros(2)
    policy = lambda state: jnp.broadcast_to(
        jnp.array([.1, .2, .3]), (2, 3)) + .1 * state['deter'][..., :1]
    trajectory, _ = nj.pure(lambda: wm.imagine(policy, start, 3))(
        {}, jax.random.PRNGKey(207))
    np.testing.assert_array_equal(trajectory['reward_action'][1:], trajectory['action'][:-1])
    np.testing.assert_array_equal(trajectory['reward_action'][0], jnp.zeros((2, 3)))
    influence = (np.asarray(trajectory['action'][:-1]) * [1., 3., 7.]).sum(-1)
    expected = np.cumsum(influence, axis=0)
    np.testing.assert_allclose(trajectory['deter'][1:, :, 0], expected, rtol=1e-6)
    greedy = behaviors.Greedy(wm, act['action'], cfg, name='reward_greedy')
    explore = behaviors.Explore(wm, act['action'], cfg, name='reward_explore')
    for behavior in (greedy, explore):
      reward = behavior.ac.critics['extr'].rewfn(trajectory)
      self.assertEqual(reward.shape, (3, 2))
      np.testing.assert_allclose(reward, expected, rtol=1e-6)
      # No transition consumes the last outgoing action. Reward callbacks
      # must ignore it, and exclude the unused initial-state reward.
      altered = dict(trajectory)
      altered['action'] = trajectory['action'].at[-1].set(-99.)
      altered['deter'] = trajectory['deter'].at[0].set(999.)
      np.testing.assert_array_equal(
          behavior.ac.critics['extr'].rewfn(altered), reward)

  def test_upstream_gru_and_s3m_input_overrides_are_preserved(self):
    for variant in ('gru', 'siso', 'mimo'):
      with self.subTest(variant=variant):
        cfg = config(**{'ssm_type': variant,
                        'reward_head.inputs': ['deter', 'stoch'],
                        'actor.inputs': ['deter', 'stoch', 'hidden']})
        self.assertIs(agentlib.latent_input_config(cfg), cfg)
        self.assertEqual(cfg.actor.inputs, ('deter', 'stoch', 'hidden'))


if __name__ == '__main__':
  unittest.main()
