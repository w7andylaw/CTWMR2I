"""Numerical and protocol checks for the single changed reward layer."""

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
from recall2imagine import jaxutils
from recall2imagine import nets
from recall2imagine import ninjax as nj
from recall2imagine import reward_nets
from test_integration import batch_data, config, spaces


def leaf(params, suffix):
  matches = [key for key in params if key.endswith(suffix)]
  if len(matches) != 1:
    raise AssertionError((suffix, matches))
  return matches[0]


class ActionRewardTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    jaxutils.COMPUTE_DTYPE = jnp.float32

  def test_discrete_affine_matches_independent_numpy_and_action_gradient(self):
    layer = reward_nets.ActionConditionedLinear(
        3, 2, 2, True, name='conditioned')
    activity = jnp.array([[.3, -.7], [.1, .9]])
    shared = jnp.array([[.2], [-.4]])
    action = jnp.array([[1., 0.], [0., 1.]])
    pure = nj.pure(layer.__call__)
    actual, params = pure({}, jax.random.PRNGKey(11), activity, shared, action)
    banks = np.asarray(params[leaf(params, '/action_kernel')])
    kernel = np.asarray(params[leaf(params, '/shared_kernel')])
    bias = np.asarray(params[leaf(params, '/bias')])
    expected = np.asarray(shared) @ kernel + bias
    for row in range(2):
      for index in range(2):
        expected[row] += np.asarray(action)[row, index] * np.asarray(activity)[row] @ banks[index]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    gradient = jax.grad(lambda a: pure(
        params, jax.random.PRNGKey(12), activity, shared, a,
        create=False, modify=False)[0].sum())(action)
    np.testing.assert_allclose(
        gradient, np.einsum('bn,knu->bk', activity, banks), atol=1e-6)
    self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-7)

  def test_continuous_affine_has_zero_action_memory_and_exact_gradient(self):
    layer = reward_nets.ActionConditionedLinear(
        2, 2, 2, False, norm='none', bias=False, name='continuous')
    activity = jnp.array([[.3, -.8]])
    shared = jnp.array([[.7]])
    action = jnp.array([[.1, -.2]])
    pure = nj.pure(layer.__call__)
    actual, params = pure({}, jax.random.PRNGKey(13), activity, shared, action)
    banks = np.asarray(params[leaf(params, '/action_kernel')])
    kernel = np.asarray(params[leaf(params, '/shared_kernel')])
    expected = np.asarray(shared) @ kernel + np.einsum(
        'bn,knu,bk->bu', activity, banks, [[1., .1, -.2]])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    zero, _ = pure(params, jax.random.PRNGKey(14), activity, shared,
                   jnp.zeros_like(action), create=False, modify=False)
    np.testing.assert_allclose(zero, np.asarray(shared) @ kernel + np.asarray(activity) @ banks[0], atol=1e-6)
    gradient = jax.grad(lambda a: pure(
        params, jax.random.PRNGKey(14), activity, shared, a,
        create=False, modify=False)[0].sum())(action)
    np.testing.assert_allclose(gradient, np.einsum('bn,knu->bk', activity, banks[1:]), atol=1e-6)

  def test_expanding_old_first_kernel_exactly_preserves_full_mlp(self):
    kw = dict(layers=2, units=7, inputs=['deter', 'stoch'],
              act='elu', norm='layer', dist='symlog_disc', outscale=1., bins=255)
    for symlog in (False, True):
      with self.subTest(symlog=symlog):
        old = nets.MLP((), symlog_inputs=symlog, **kw, name='old')
        new = reward_nets.ActionConditionedMLP(
            (), 2, 3, True, symlog_inputs=symlog, **kw, name='new')
        feat = {'deter': jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 17.,
                'stoch': jnp.ones((2, 3, 2, 2)) / 2.}
        action = jax.nn.one_hot(jnp.array([[0, 1, 2], [2, 0, 1]]), 3)
        oldpure = nj.pure(lambda f: old(f).logits)
        newpure = nj.pure(lambda f, a: new(f, a).logits)
        expected, oldparams = oldpure({}, jax.random.PRNGKey(15), feat)
        _, newparams = newpure({}, jax.random.PRNGKey(16), feat, action)
        oldkernel = oldparams[leaf(oldparams, '/h0/kernel')]
        for key in newparams:
          suffix = key.split('/', 1)[1]
          if suffix == 'h0/shared_kernel':
            newparams[key] = oldkernel[2:]
          elif suffix == 'h0/action_kernel':
            newparams[key] = jnp.repeat(oldkernel[None, :2], 3, 0)
          else:
            newparams[key] = oldparams['old/' + suffix]
        actual, _ = newpure(newparams, jax.random.PRNGKey(17), feat, action,
                            create=False, modify=False)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertFalse(any(key.endswith('/h0/bias') for key in newparams))

  def test_action_banks_express_cue_action_matching_without_new_hidden_layer(self):
    layer = reward_nets.ActionConditionedLinear(
        1, 1, 2, True, bias=False, name='matching')
    activity = jnp.array([[-1.], [-1.], [1.], [1.]])
    shared = jnp.zeros((4, 1))
    action = jnp.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    pure = nj.pure(layer.__call__)
    _, params = pure({}, jax.random.PRNGKey(18), activity, shared, action)
    params[leaf(params, '/action_kernel')] = jnp.array([[[-1.]], [[1.]]])
    params[leaf(params, '/shared_kernel')] = jnp.zeros((1, 1))
    output, _ = pure(params, jax.random.PRNGKey(19), activity, shared, action,
                     create=False, modify=False)
    np.testing.assert_array_equal(output[:, 0], [1., -1., -1., 1.])

  def test_invalid_shapes_and_missing_incoming_action_fail_explicitly(self):
    cfg = config()
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='required_action')
    state = {'deter': jnp.zeros((2, cfg.nrsm.deter)),
             'stoch': jnp.ones((2, cfg.nrsm.stoch, cfg.nrsm.classes))}
    with self.assertRaisesRegex(ValueError, 'requires the incoming action'):
      wm.reward(state)
    with self.assertRaisesRegex(ValueError, 'at least one'):
      reward_nets.ActionConditionedMLP(
          (), 2, 2, True, layers=0, units=4, inputs=['deter', 'stoch'], name='empty')
    with self.assertRaisesRegex(ValueError, 'in that order'):
      reward_nets.ActionConditionedMLP(
          (), 2, 2, True, layers=1, units=4, inputs=['stoch', 'deter'], name='reversed')
    pure = nj.pure(lambda a: wm.reward(state, a).mean())
    with self.assertRaisesRegex(ValueError, 'leading axes and width'):
      pure({}, jax.random.PRNGKey(20), jnp.zeros((2, 2)))

  def test_reward_normalizes_factored_actions_and_clip_like_dynamics(self):
    cfg = config(**{'nrsm.action_clip': .5, 'reward_head.outscale': 1.})
    obs, act = spaces(False)
    wm = agentlib.WorldModel(obs, act, cfg, name='processed_action')
    state = {'deter': jnp.arange(2 * cfg.nrsm.deter, dtype=jnp.float32).reshape(2, cfg.nrsm.deter) / 19.,
             'stoch': jnp.ones((2, cfg.nrsm.stoch, cfg.nrsm.classes))}
    action = jnp.array([[[1.], [-2.]], [[.1], [-.3]]])
    processed = wm.rssm._action(action, state['stoch'])
    np.testing.assert_allclose(processed, [[.5, -.5], [.1, -.3]])
    call = nj.pure(lambda a: wm.reward(state, a).logits)
    expected, params = call({}, jax.random.PRNGKey(21), action)
    actual, _ = call(params, jax.random.PRNGKey(22), processed,
                     create=False, modify=False)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_continuous_imagination_preserves_reward_action_gradient(self):
    cfg = config(**{'reward_head.outscale': 1.})
    obs, act = spaces(False)
    wm = agentlib.WorldModel(obs, act, cfg, name='continuous_imagined_reward')
    data = batch_data(cfg, False)
    data['cont'] = 1. - data['is_terminal'].astype(jnp.float32)
    (_, (state, _, _)), params = nj.pure(
        lambda d: wm.loss(d, wm.initial(cfg.batch_size)))(
            {}, jax.random.PRNGKey(23), data)
    start = dict(state[0], is_terminal=jnp.zeros(cfg.batch_size))
    def score(action):
      trajectory = wm.imagine(
          lambda latent: jnp.broadcast_to(action, (cfg.batch_size, 2)), start, 2)
      return wm.reward(trajectory).mean()[1:].sum()
    pure = nj.pure(score)
    gradient = jax.grad(lambda action: pure(
        params, jax.random.PRNGKey(24), action,
        create=False, modify=False)[0])(jnp.array([.2, -.3]))
    self.assertTrue(np.isfinite(np.asarray(gradient)).all())
    self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-7)

  def test_non_nrsm_keeps_original_reward_class_and_signature(self):
    cfg = config(**{'ssm_type': 'gru'})
    obs, act = spaces()
    wm = agentlib.WorldModel(obs, act, cfg, name='original_gru_reward')
    self.assertIs(type(wm.heads['reward']), nets.MLP)
    feat = {'deter': jnp.ones((2, cfg.rssm.deter)),
            'stoch': jnp.ones((2, cfg.rssm.stoch, cfg.rssm.classes))}
    out, _ = nj.pure(lambda f: wm.reward(f).mean())({}, jax.random.PRNGKey(25), feat)
    self.assertEqual(out.shape, (2,))


if __name__ == '__main__':
  unittest.main()
