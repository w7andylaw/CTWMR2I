"""Small numerical tests of the native R2I Agent with the NRSM core.

These tests exercise vector and image observations, both action types, the
upstream model/actor/critic updates, and replay reset/discount conventions.
They are smoke checks, not evidence of benchmark learning performance.
"""

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import pathlib
import pickle
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from recall2imagine import agent as agentlib
from recall2imagine import jaxutils
from recall2imagine import ninjax as nj
import embodied


STATE_KEYS = {'deter', 'pre_trace', 'post_trace', 'stoch', 'logit'}
MEMORY_KEYS = ('deter', 'pre_trace', 'post_trace')


def config(*presets, **updates):
  result = embodied.Config(agentlib.Agent.configs['defaults'])
  for preset in ('ctwm', 'ctwm_smoke', *presets):
    result = result.update(agentlib.Agent.configs[preset])
  return result.update(updates)


def spaces(discrete=True, image=False):
  obs = {
      'reward': embodied.Space(np.float32),
      'is_first': embodied.Space(bool),
      'is_last': embodied.Space(bool),
      'is_terminal': embodied.Space(bool),
  }
  obs['image' if image else 'observation'] = (
      embodied.Space(np.uint8, (16, 16, 3), 0, 255) if image else
      embodied.Space(np.float32, (5,)))
  act = {
      'action': embodied.Space(np.float32, (3,), 0, 1) if discrete else
      embodied.Space(np.float32, (2,), -1, 1),
      'reset': embodied.Space(bool),
  }
  # R2I's action wrapper represents discrete actions as float one-hot vectors.
  if discrete:
    act['action']._discrete = True
  return obs, act


def batch_data(cfg, discrete=True, image=False):
  batch, length = cfg.batch_size, cfg.batch_length
  rng = np.random.default_rng(12)
  first = np.zeros((batch, length), bool)
  first[:, 0] = True
  first[0, length // 2] = True
  terminal = np.zeros((batch, length), bool)
  terminal[:, -1] = True
  actions = (
      np.eye(3, dtype=np.float32)[rng.integers(3, size=(batch, length))]
      if discrete else rng.uniform(-1, 1, (batch, length, 2)).astype(np.float32))
  result = dict(
      reward=rng.normal(size=(batch, length)).astype(np.float32),
      is_first=first, is_last=terminal, is_terminal=terminal,
      action=actions, reset=np.zeros((batch, length), bool))
  if image:
    result['image'] = rng.integers(
        256, size=(batch, length, 16, 16, 3), dtype=np.uint8)
  else:
    result['observation'] = rng.normal(
        size=(batch, length, 5)).astype(np.float32)
  return jax.tree_util.tree_map(jnp.asarray, result)


class NativeAgentIntegrationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    jaxutils.COMPUTE_DTYPE = jnp.float32

  def test_task_hidden_overrides_are_resolved_only_for_nrsm(self):
    cfg = config('popgym')
    self.assertIn('hidden', cfg.actor.inputs)
    resolved = agentlib.latent_input_config(cfg)
    self.assertEqual(resolved.actor.inputs, ('deter', 'stoch'))
    self.assertEqual(resolved.critic.inputs, ('deter', 'stoch'))
    source = embodied.Config(agentlib.Agent.configs['defaults'])
    self.assertEqual(agentlib.latent_input_config(source).actor.inputs,
                     source.actor.inputs)

  def test_nrsm_rejects_mixed_precision(self):
    with self.assertRaisesRegex(ValueError, 'float32'):
      agentlib.latent_input_config(config(**{'jax.precision': 'float16'}))

  def test_base_kl_keeps_r2i_gradient_routing_and_free_nats(self):
    core = agentlib.nrsm_nets.NRSM(**config().nrsm, name='kl_core')
    posterior = jnp.array([[[3., -1., 0.], [-1., 2., 0.]]])
    prior = jnp.array([[[-2., 2., 0.], [2., -1., 0.]]])

    def losses(q, p, free):
      post, predicted = {'logit': q}, {'logit': p}
      return (core.dyn_loss(post, predicted, free=free).sum(),
              core.rep_loss(post, predicted, free=free).sum())

    pure = nj.pure(losses)
    evaluate = lambda q, p, free: pure(
        {}, jax.random.PRNGKey(0), q, p, free)[0]
    raw = agentlib.nrsm_nets.categorical_kl(posterior, prior).sum()
    np.testing.assert_allclose(evaluate(posterior, prior, 0.), (raw, raw), rtol=1e-6)
    # Dynamics learns the prior only; representation learns the posterior only.
    for index, frozen_arg in ((0, 0), (1, 1)):
      gradients = jax.grad(
          lambda q, p: evaluate(q, p, 0.)[index], argnums=(0, 1))(
              posterior, prior)
      np.testing.assert_array_equal(gradients[frozen_arg],
                                    np.zeros_like(gradients[frozen_arg]))
      self.assertGreater(float(jnp.linalg.norm(gradients[1 - frozen_arg])), 0.)
    np.testing.assert_allclose(evaluate(posterior, posterior, 1.), (1., 1.))

  def test_world_loss_adds_only_configured_tick_regularizer(self):
    cfg = config(**{'nrsm_tick_scale': 2.5})
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm')
    data = batch_data(cfg)
    data['cont'] = 1.0 - data['is_terminal'].astype(jnp.float32)
    fn = nj.pure(lambda d: wm.loss(d, wm.initial(cfg.batch_size)))
    (loss, (state, outs, metrics)), _ = fn({}, jax.random.PRNGKey(1), data)
    expected = sum(outs[key + '_loss'] * scale
                   for key, scale in wm.scales.items()
                   if key + '_loss' in outs)
    expected += 2.5 * outs['tick_loss']
    np.testing.assert_allclose(loss, expected.mean(), rtol=2e-6)
    self.assertEqual(set(state[0]), STATE_KEYS)
    self.assertEqual(set(outs['post']), set(state[0]))
    self.assertEqual(outs['tick_diagnostics']['tick_kl'].shape,
                     (cfg.batch_size, cfg.batch_length, cfg.nrsm.ticks))
    self.assertNotIn('aux_loss', outs)
    self.assertTrue(np.isfinite(float(metrics['model_loss_mean'])))

  def test_chunk_first_resets_previous_state_and_action(self):
    cfg = config()
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm_reset')
    data = batch_data(cfg)
    data['cont'] = 1.0 - data['is_terminal'].astype(jnp.float32)
    initial = nj.pure(wm.initial)
    state, params = initial({}, jax.random.PRNGKey(0), cfg.batch_size)
    altered = jax.tree_util.tree_map(lambda value: jnp.ones_like(value) * 7, state)
    fn = nj.pure(wm.loss)
    out, params = fn(params, jax.random.PRNGKey(4), data, state)
    other, _ = fn(params, jax.random.PRNGKey(4), data, altered)
    for key in STATE_KEYS:
      np.testing.assert_allclose(out[1][1]['post'][key],
                                 other[1][1]['post'][key], rtol=1e-6, atol=1e-6)

  def test_imagination_applies_gamma_once_and_masks_terminal_starts(self):
    cfg = config()
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm_discount')

    class ConstantContinuation:
      def __init__(self, trajectory):
        self.trajectory = trajectory
      def mode(self):
        return jnp.ones(self.trajectory['deter'].shape[:-1], jnp.float32)

    wm.heads['cont'] = ConstantContinuation

    def imagine():
      start = wm.rssm.initial(2)
      start['is_terminal'] = jnp.array([0., 1.])
      return wm.imagine(
          lambda state: jnp.zeros((state['deter'].shape[0], 3)), start, 3)

    trajectory, _ = nj.pure(imagine)({}, jax.random.PRNGKey(7))
    gamma = 1 - 1 / cfg.horizon
    np.testing.assert_allclose(trajectory['weight'][:, 0],
                               gamma ** np.arange(4), rtol=1e-6)
    np.testing.assert_array_equal(trajectory['weight'][:, 1], np.zeros(4))
    self.assertEqual(set(trajectory),
                     STATE_KEYS | {'action', 'cont', 'weight'})

  def test_native_discrete_and_continuous_joint_updates_and_restore(self):
    for discrete in (True, False):
      with self.subTest(discrete=discrete):
        cfg = config()
        obs, act = spaces(discrete)
        inner = agentlib.Agent.inner(
            obs, act, embodied.Counter(), cfg, name='agent')
        state, params = nj.pure(inner.train_initial)(
            {}, jax.random.PRNGKey(0), cfg.batch_size)
        train = jax.jit(nj.pure(inner.train))
        data = batch_data(cfg, discrete)
        (_, state, metrics), params = train(
            params, jax.random.PRNGKey(10), data, state)
        jax.block_until_ready(metrics['model_loss_mean'])
        before = jax.tree_util.tree_map(np.asarray, params)
        (_, next_state, next_metrics), updated = train(
            params, jax.random.PRNGKey(11), data, state)
        for value in jax.tree_util.tree_leaves(next_metrics):
          self.assertTrue(np.all(np.isfinite(np.asarray(value))))
        self.assertEqual(set(next_state[0]), STATE_KEYS)
        for value in jax.tree_util.tree_leaves(updated):
          self.assertTrue(np.all(np.isfinite(np.asarray(value))))
        for marker in ('/rssm/', '/actor/', '/critic/net/'):
          changed = [key for key in before if marker in key and
                     not np.array_equal(before[key], np.asarray(updated[key]))]
          self.assertTrue(changed, 'No updated parameters for ' + marker)
        policy_state, _ = nj.pure(inner.policy_initial)(
            updated, jax.random.PRNGKey(17), cfg.batch_size)
        observation = {key: data[key][:, 0] for key in obs}
        (policy_out, policy_state), _ = nj.pure(inner.policy)(
            updated, jax.random.PRNGKey(18), observation, policy_state,
            mode='eval')
        self.assertEqual(policy_out['action'].shape,
                         (cfg.batch_size,) + act['action'].shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(policy_out['action']))))
        self.assertEqual(set(policy_state[0][0]), STATE_KEYS)
        # The native actor/critic optimizer never changes world-model params.
        prepared = inner.preprocess(data)
        (_, (_, wm_outs, _)), _ = nj.pure(inner.wm.loss)(
            updated, jax.random.PRNGKey(20), prepared, next_state)
        context = {**prepared, **wm_outs['post']}
        start = jax.tree_util.tree_map(
            lambda value: value.reshape((-1,) + value.shape[2:]), context)
        (_, actor_params) = nj.pure(
            lambda s, c: inner.task_behavior.train(inner.wm.imagine, s, c))(
                updated, jax.random.PRNGKey(21), start, context)
        world_keys = [key for key in updated if '/wm/' in key]
        self.assertTrue(world_keys)
        for key in world_keys:
          for value, expected in zip(jax.tree_util.tree_leaves(actor_params[key]),
                                      jax.tree_util.tree_leaves(updated[key])):
            np.testing.assert_array_equal(value, expected)
        # A serialized host save/device restore round trip preserves updates.
        snapshot = pickle.dumps(jax.tree_util.tree_map(np.asarray, params))
        restored = jax.tree_util.tree_map(jnp.asarray, pickle.loads(snapshot))
        (replay, replay_params) = train(
            restored, jax.random.PRNGKey(11), data, state)
        for actual, expected in zip(jax.tree_util.tree_leaves(updated),
                                    jax.tree_util.tree_leaves(replay_params)):
          np.testing.assert_array_equal(actual, expected)

  def test_world_loss_backpropagates_into_both_incoming_traces(self):
    # This chunk continues an episode. Reset chunks intentionally have zero
    # derivative with respect to the incoming carry, covered above.
    cfg = config(**{'batch_length': 3, 'nrsm.ticks': 1})
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm_trace_gradient')
    data = batch_data(cfg)
    data['is_first'] = jnp.zeros_like(data['is_first'])
    data['cont'] = 1.0 - data['is_terminal'].astype(jnp.float32)
    latent, action = wm.initial(cfg.batch_size)
    for index, key in enumerate(MEMORY_KEYS):
      latent[key] = jnp.linspace(
          -.4, .6 + index / 10, latent[key].size).reshape(latent[key].shape)
    state = latent, action
    loss = nj.pure(lambda incoming: wm.loss(data, incoming)[0])
    value, params = loss({}, jax.random.PRNGKey(30), state)
    self.assertTrue(np.isfinite(float(value)))
    derivatives = jax.jit(jax.grad(lambda incoming: loss(
        params, jax.random.PRNGKey(31), incoming,
        create=False, modify=False)[0]))(state)
    for key in ('pre_trace', 'post_trace'):
      derivative = np.asarray(derivatives[0][key])
      self.assertTrue(np.all(np.isfinite(derivative)), key)
      self.assertGreater(np.linalg.norm(derivative), 0., key)

  def test_policy_carries_history_across_steps_and_checkpoint_round_trip(self):
    cfg = config()
    obs, act = spaces()
    inner = agentlib.Agent.inner(
        obs, act, embodied.Counter(), cfg, name='policy_memory')
    state, params = nj.pure(inner.policy_initial)(
        {}, jax.random.PRNGKey(40), cfg.batch_size)
    data = batch_data(cfg)
    observation = {key: data[key][:, 0] for key in obs}
    policy = jax.jit(nj.pure(inner.policy), static_argnames=('mode',))
    (_, state), params = policy(
        params, jax.random.PRNGKey(41), observation, state, mode='eval')
    observation['is_first'] = jnp.zeros(cfg.batch_size, jnp.bool_)
    (_, state), params = policy(
        params, jax.random.PRNGKey(42), observation, state, mode='eval')
    incoming = state[0][0]
    self.assertEqual(set(incoming), STATE_KEYS)
    for key in ('pre_trace', 'post_trace'):
      self.assertEqual(incoming[key].shape,
                       (cfg.batch_size, cfg.nrsm.neurons, cfg.nrsm.memory))
      self.assertGreater(float(jnp.linalg.norm(incoming[key])), 0., key)

    # A checkpoint must carry the complete neuronal state, as well as the
    # weights. Restoring it reproduces the next sampled policy transition.
    snapshot = pickle.dumps(jax.tree_util.tree_map(np.asarray, (params, state)))
    restored_params, restored_state = jax.tree_util.tree_map(
        jnp.asarray, pickle.loads(snapshot))
    (expected, continued), _ = policy(
        params, jax.random.PRNGKey(43), observation, state, mode='eval')
    (actual, restored_next), _ = policy(
        restored_params, jax.random.PRNGKey(43), observation,
        restored_state, mode='eval')
    for left, right in zip(jax.tree_util.tree_leaves((expected, continued)),
                           jax.tree_util.tree_leaves((actual, restored_next))):
      np.testing.assert_array_equal(left, right)
    for key in ('pre_trace', 'post_trace'):
      # With K < M the surviving tail must remain exactly the oldest part
      # of the next state, rather than being regenerated from deter/stoch.
      np.testing.assert_array_equal(
          continued[0][0][key][..., :cfg.nrsm.memory - cfg.nrsm.ticks],
          incoming[key][..., cfg.nrsm.ticks:])

    blank = jax.tree_util.tree_map(lambda value: value, state)
    blank[0][0]['pre_trace'] = jnp.zeros_like(incoming['pre_trace'])
    blank[0][0]['post_trace'] = jnp.zeros_like(incoming['post_trace'])
    (_, forgotten), _ = policy(
        params, jax.random.PRNGKey(43), observation, blank, mode='eval')
    self.assertGreater(float(jnp.max(jnp.abs(
        continued[0][0]['deter'] - forgotten[0][0]['deter']))), 1e-7)

  def test_imagination_preserves_start_traces_and_uses_the_same_transition(self):
    cfg = config()
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm_imagination_memory')
    data = batch_data(cfg)
    data['cont'] = 1.0 - data['is_terminal'].astype(jnp.float32)
    (_, (state, _, _)), params = nj.pure(
        lambda d: wm.loss(d, wm.initial(cfg.batch_size)))(
            {}, jax.random.PRNGKey(50), data)
    start = dict(state[0], is_terminal=jnp.zeros(cfg.batch_size))
    action = jnp.broadcast_to(jnp.array([1., 0., 0.]), (cfg.batch_size, 3))
    imagine = jax.jit(nj.pure(
        lambda initial: wm.imagine(lambda latent: action, initial, 3)))
    trajectory, _ = imagine(params, jax.random.PRNGKey(51), start)
    self.assertEqual(set(trajectory), STATE_KEYS | {'action', 'cont', 'weight'})
    for key in STATE_KEYS:
      np.testing.assert_array_equal(trajectory[key][0], start[key])
      self.assertTrue(np.all(np.isfinite(np.asarray(trajectory[key]))))
    # Random prior samples can differ when called outside scan; the
    # deterministic neuronal transition must agree exactly for the same input
    # state/action at every imagined step.
    step = jax.jit(nj.pure(wm.rssm.img_step))
    for index in range(3):
      previous = {key: trajectory[key][index] for key in STATE_KEYS}
      reference, _ = step(params, jax.random.PRNGKey(52 + index), previous,
                          trajectory['action'][index])
      for key in MEMORY_KEYS:
        np.testing.assert_allclose(trajectory[key][index + 1], reference[key],
                                   rtol=2e-6, atol=2e-6)
    blank = {key: value for key, value in start.items()}
    blank['pre_trace'] = jnp.zeros_like(start['pre_trace'])
    blank['post_trace'] = jnp.zeros_like(start['post_trace'])
    forgotten, _ = imagine(params, jax.random.PRNGKey(51), blank)
    self.assertGreater(float(jnp.max(jnp.abs(
        trajectory['deter'][1] - forgotten['deter'][1]))), 1e-7)

  def test_imagination_detaches_all_start_memory_but_preserves_action_gradients(self):
    cfg = config()
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='wm_detached_start')
    data = batch_data(cfg)
    data['cont'] = 1.0 - data['is_terminal'].astype(jnp.float32)
    (_, (state, _, _)), params = nj.pure(
        lambda d: wm.loss(d, wm.initial(cfg.batch_size)))(
            {}, jax.random.PRNGKey(60), data)
    start = dict(state[0], is_terminal=jnp.zeros(cfg.batch_size))

    def imagined_score(initial, action_parameters):
      policy = lambda latent: jnp.broadcast_to(
          action_parameters, (latent['deter'].shape[0], 3))
      trajectory = wm.imagine(policy, initial, 3)
      rewards = wm.heads['reward'](trajectory).mean()
      return rewards[1:].sum() + trajectory['deter'][1:].sum()

    score = nj.pure(imagined_score)
    derivatives = jax.jit(jax.value_and_grad(
        lambda initial, action_parameters: score(
            params, jax.random.PRNGKey(61), initial, action_parameters,
            create=False, modify=False)[0], argnums=(0, 1)))
    value, (start_gradient, action_gradient) = derivatives(
        start, jnp.array([.2, -.15, .4], jnp.float32))
    self.assertTrue(np.isfinite(float(value)))
    for key in STATE_KEYS:
      np.testing.assert_array_equal(start_gradient[key],
                                     jnp.zeros_like(start[key]), err_msg=key)
    self.assertTrue(np.all(np.isfinite(np.asarray(action_gradient))))
    self.assertGreater(float(jnp.linalg.norm(action_gradient)), 1e-7)

  def test_image_encoder_decoder_loss(self):
    cfg = config(**{
        'encoder.mlp_keys': '$^', 'encoder.cnn_keys': 'image',
        'decoder.mlp_keys': '$^', 'decoder.cnn_keys': 'image'})
    obs, act = spaces(image=True)
    inner = agentlib.Agent.inner(obs, act, embodied.Counter(), cfg, name='image_agent')
    data = inner.preprocess(batch_data(cfg, image=True))
    (loss, (_, outs, _)), _ = nj.pure(
        lambda d: inner.wm.loss(d, inner.wm.initial(cfg.batch_size)))(
            {}, jax.random.PRNGKey(14), data)
    self.assertTrue(np.isfinite(float(loss)))
    self.assertEqual(outs['image_loss'].shape,
                     (cfg.batch_size, cfg.batch_length))


if __name__ == '__main__':
  unittest.main()
