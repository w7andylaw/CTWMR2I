"""Small numerical tests of the native R2I Agent with the NRSM core.

These tests exercise vector and image observations, both action types, the
upstream model/actor/critic updates, and replay reset/discount conventions.
They are smoke checks, not evidence of benchmark learning performance.
"""

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

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
    self.assertEqual(set(state[0]), {'deter', 'stoch', 'logit'})
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
    for key in ('deter', 'stoch', 'logit'):
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
                     {'deter', 'stoch', 'logit', 'action', 'cont', 'weight'})

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
        self.assertEqual(set(next_state[0]), {'deter', 'stoch', 'logit'})
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
        self.assertEqual(set(policy_state[0][0]), {'deter', 'stoch', 'logit'})
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
