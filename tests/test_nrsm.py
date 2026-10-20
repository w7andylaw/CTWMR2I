"""Numerical and recurrent-state contracts for the CT-WM/R2I core."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recall2imagine import ninjax as nj
from recall2imagine.nrsm_nets import NRSM, PrivateNLM, sync_step, tick_statistics


class TickStatisticsTest(unittest.TestCase):

  def setUp(self):
    self.prior = jnp.asarray([
        [[1.1, -.2, .5], [.1, 1.3, -.4]],
        [[-.7, .2, 1.0], [.8, -.1, -.2]],
        [[.3, -.5, .2], [-.3, .1, .8]],
    ], jnp.float32)
    self.post = jnp.asarray([
        [[-.2, .8, .1], [1.0, -.1, .2]],
        [[.4, -.3, .7], [-.2, .6, .4]],
        [[-.1, .9, -.3], [.2, .5, -.4]],
    ], jnp.float32)

  def test_modes_and_unclipped_kl(self):
    statistics = tick_statistics(self.prior, self.post)
    logq = np.asarray(jax.nn.log_softmax(self.post, -1))
    logp = np.asarray(jax.nn.log_softmax(self.prior, -1))
    expected_kl = (np.exp(logq) * (logq - logp)).sum((-2, -1))
    np.testing.assert_allclose(statistics['tick_kl'], expected_kl, atol=2e-7)
    self.assertTrue(np.all(expected_kl < 1.0))  # No free-nats clipping here.
    np.testing.assert_allclose(statistics['tick_weights'].sum(), 1., atol=1e-7)
    np.testing.assert_allclose(
        tick_statistics(self.prior, self.post, 'uniform')['tick_loss'], expected_kl.mean())
    np.testing.assert_array_equal(
        tick_statistics(self.prior, self.post, 'off')['tick_weights'], np.zeros(3))
    self.assertEqual(float(tick_statistics(self.prior, self.post, 'off')['tick_loss']), 0.)
    with self.assertRaises(ValueError):
      tick_statistics(self.prior, self.post, 'unknown')

  def test_adaptive_weight_gradients_are_not_stopped(self):
    objective = lambda prior: tick_statistics(prior, self.post)['tick_loss']
    full = jax.grad(objective)(self.prior)
    def stopped_objective(prior):
      stats = tick_statistics(prior, self.post)
      return (jax.lax.stop_gradient(stats['tick_weights']) * stats['tick_kl']).sum()
    stopped = jax.grad(stopped_objective)(self.prior)
    self.assertGreater(float(jnp.max(jnp.abs(full - stopped))), 1e-4)
    index = (1, 0, 1)
    eps = .002
    finite_difference = (objective(self.prior.at[index].add(eps)) -
                         objective(self.prior.at[index].add(-eps))) / (2 * eps)
    self.assertAlmostEqual(float(full[index]), float(finite_difference), places=4)
    posterior_grad = jax.grad(lambda post: tick_statistics(self.prior, post)['tick_loss'])(self.post)
    self.assertGreater(float(jnp.linalg.norm(posterior_grad)), 0.)


class NeuronalMathTest(unittest.TestCase):

  def test_synchronization_matches_explicit_weighted_sum_and_decay_gradient(self):
    activities = jnp.asarray([[.2, -.5, .7], [.9, .1, -.3], [-.4, .8, .6]])
    pairs = jnp.asarray([[0, 0], [0, 1], [1, 2]], jnp.int32)
    beta = jnp.asarray([-.2, .3, .8])
    def recurrence(decay_parameter):
      numerator = jnp.zeros(3)
      normalizer = jnp.zeros(3)
      values = []
      for activity in activities:
        numerator, normalizer, sync = sync_step(
            numerator, normalizer, activity, decay_parameter, pairs)
        values.append(sync)
      return jnp.stack(values)
    def explicit(decay_parameter):
      decay = jnp.exp(-jax.nn.softplus(decay_parameter))
      products = activities[:, pairs[:, 0]] * activities[:, pairs[:, 1]]
      values = []
      for index in range(len(activities)):
        weights = decay[None] ** jnp.arange(index, -1, -1)[:, None]
        values.append((weights * products[:index + 1]).sum(0) / jnp.sqrt(weights.sum(0)))
      return jnp.stack(values)
    np.testing.assert_allclose(recurrence(beta), explicit(beta), atol=1e-7)
    np.testing.assert_allclose(jax.jacrev(recurrence)(beta), jax.jacrev(explicit)(beta), atol=2e-7)

  def test_private_nlm_has_no_cross_neuron_connections(self):
    nlm = PrivateNLM(3, 4, 2, name='private_nlm')
    history = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    apply = nj.pure(nlm.__call__)
    _, parameters = apply({}, jax.random.PRNGKey(0), history)
    value = lambda inputs: apply(parameters, jax.random.PRNGKey(1), inputs,
                                 create=False, modify=False)[0]
    derivative = np.asarray(jax.jacrev(value)(history))
    for output_neuron in range(3):
      for input_neuron in range(3):
        if output_neuron != input_neuron:
          np.testing.assert_array_equal(derivative[output_neuron, input_neuron], np.zeros(4))
      self.assertGreater(np.linalg.norm(derivative[output_neuron, output_neuron]), 0.)
    w1 = next(v for k, v in parameters.items() if k.endswith('/w1'))
    w2 = next(v for k, v in parameters.items() if k.endswith('/w2'))
    self.assertLessEqual(float(jnp.abs(w1).max()), math.sqrt(6 / (4 + 2)))
    self.assertLessEqual(float(jnp.abs(w2).max()), math.sqrt(6 / (2 + 1)))


class NRSMTest(unittest.TestCase):

  def setUp(self):
    self.core = NRSM(neurons=4, memory=3, ticks=3, deter=5, stoch=2,
                     classes=3, units=7, nlm_hidden=3, sync_neurons=3,
                     name='nrsm')
    self.embed = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 20
    self.action = jnp.asarray([[[.2, -.1], [.4, .6], [-.3, .2], [.1, -.4]]])
    self.first = jnp.asarray([[True, False, True, False]])
    _, self.parameters = nj.pure(self.core.observe_with_ticks)(
        {}, jax.random.PRNGKey(0), self.embed, self.action, self.first)

  def apply(self, method, *args, seed=1):
    return nj.pure(method)(self.parameters, jax.random.PRNGKey(seed), *args,
                           create=False, modify=False)[0]

  def test_only_expected_state_and_no_posterior_context_correction(self):
    post, prior, diagnostics = self.apply(
        self.core.observe_with_ticks, self.embed, self.action, self.first)
    self.assertEqual(set(post), {'deter', 'stoch', 'logit'})
    self.assertEqual(set(prior), set(post))
    self.assertEqual(set(self.core.initial(1)), set(post))
    np.testing.assert_array_equal(post['deter'], prior['deter'])
    np.testing.assert_array_equal(post['deter'], diagnostics['tick_contexts'][..., -1, :])
    self.assertEqual(diagnostics['tick_kl'].shape, (1, 4, 3))
    self.assertEqual(diagnostics['tick_loss'].shape, (1, 4))
    np.testing.assert_array_equal(post['stoch'].sum(-1), np.ones((1, 4, 2)))
    self.assertTrue(np.all(np.isin(np.asarray(post['stoch']), [0., 1.])))
    self.assertFalse(any('posterior_gate' in key or 'candidate' in key for key in self.parameters))

  def test_reset_discards_incoming_memory_and_action(self):
    initial = self.core.initial(1)
    nonzero = {key: jnp.ones_like(value) * 3 for key, value in initial.items()}
    args = self.embed[:, 0], jnp.ones(1, jnp.bool_)
    reset = self.apply(self.core.obs_step, nonzero, self.action[:, 0], *args)
    fresh = self.apply(self.core.obs_step, initial, jnp.zeros_like(self.action[:, 0]), *args)
    for first, second in zip(jax.tree_util.tree_leaves(reset), jax.tree_util.tree_leaves(fresh)):
      np.testing.assert_array_equal(first, second)

  def test_prior_does_not_use_current_observation_and_local_buffers_do_not_persist(self):
    initial = self.core.initial(1)
    first = jnp.zeros(1, jnp.bool_)
    post1, prior1 = self.apply(self.core.obs_step, initial, self.action[:, 0], self.embed[:, 0], first)
    post2, prior2 = self.apply(self.core.obs_step, initial, self.action[:, 0], self.embed[:, 0] + 10, first)
    for key in prior1:
      np.testing.assert_array_equal(prior1[key], prior2[key])
    self.assertGreater(float(jnp.abs(post1['logit'] - post2['logit']).max()), .001)
    # The same persistent state/action reproduces all local trajectories,
    # regardless of other calls between them.
    before = self.apply(self.core.prior_context, initial, self.action[:, 0])
    self.apply(self.core.prior_context, initial, self.action[:, 0] * 2)
    after = self.apply(self.core.prior_context, initial, self.action[:, 0])
    for left, right in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)):
      np.testing.assert_array_equal(left, right)

  def test_transition_and_logits_match_independent_numpy_formulas(self):
    parameters = {key: np.asarray(value) for key, value in self.parameters.items()}
    beta_key = next(key for key in parameters if key.endswith('/sync_beta'))
    parameters[beta_key] = np.linspace(-.3, .5, 6, dtype=np.float32)
    def parameter(suffix):
      matches = [value for key, value in parameters.items() if key.endswith('/' + suffix)]
      self.assertEqual(len(matches), 1, suffix)
      return matches[0]
    def dense(name, value, activation='none'):
      value = value @ parameter(name + '/kernel') + parameter(name + '/bias')
      if activation == 'elu':
        return np.where(value > 0, value, np.expm1(value))
      if activation == 'tanh':
        return np.tanh(value)
      return value
    initial = dict(deter=jnp.asarray([[.1, -.2, .3, .4, -.5]]),
                   stoch=jax.nn.one_hot(jnp.asarray([[0, 2]]), 3),
                   logit=jnp.zeros((1, 2, 3)))
    u = np.concatenate((np.asarray(initial['deter']),
                        np.asarray(initial['stoch']).reshape(1, -1),
                        np.asarray(self.action[:, 0])), -1)
    activity = dense('activity_initializer', u, 'tanh')
    history = dense('history_initializer', u, 'tanh').reshape(1, 4, 3)
    pairs = np.asarray([(i, j) for i in range(3) for j in range(i, 3)])
    decay = np.exp(-np.logaddexp(0, parameters[beta_key]))
    contexts, activities, synchronies, products = [], [], [], []
    for tick in range(3):
      alpha = dense('synapse_hidden', np.concatenate((activity, u), -1), 'elu')
      alpha = dense('synapse_output', alpha)
      history = np.concatenate((history[..., 1:], alpha[..., None]), -1)
      hidden = np.einsum('bdm,dmh->bdh', history, parameter('nlm/w1')) + parameter('nlm/b1')
      hidden = np.where(hidden > 0, hidden, np.expm1(hidden))
      activity = np.tanh((hidden * parameter('nlm/w2')).sum(-1) + parameter('nlm/b2'))
      products.append(activity[:, pairs[:, 0]] * activity[:, pairs[:, 1]])
      weights = decay[None, :] ** np.arange(tick, -1, -1)[:, None]
      sync = (weights[:, None, :] * np.stack(products)).sum(0) / np.sqrt(weights.sum(0))
      contexts.append(dense('sync_projection', sync, 'tanh'))
      activities.append(activity)
      synchronies.append(sync)
    def logits(name, inputs):
      hidden = dense(name + '_hidden', inputs, 'elu')
      raw = dense(name + '_stats', hidden).reshape(1, 2, 3)
      probabilities = np.exp(raw - raw.max(-1, keepdims=True))
      probabilities /= probabilities.sum(-1, keepdims=True)
      return np.log(.99 * probabilities + .01 / 3)
    expected_prior = logits('prior', contexts[-1])
    expected_post = logits('posterior', np.concatenate((contexts[-1], np.asarray(self.embed[:, 0])), -1))
    outputs, _ = nj.pure(self.core.observe_with_ticks)(
        {key: jnp.asarray(value) for key, value in parameters.items()},
        jax.random.PRNGKey(1), self.embed[:, :1], self.action[:, :1],
        jnp.zeros((1, 1), jnp.bool_), initial, create=False, modify=False)
    post, prior, diagnostics = outputs
    np.testing.assert_allclose(diagnostics['tick_contexts'][:, 0], np.stack(contexts, 1), atol=2e-7)
    np.testing.assert_allclose(diagnostics['tick_activities'][:, 0], np.stack(activities, 1), atol=2e-7)
    np.testing.assert_allclose(diagnostics['tick_synchrony'][:, 0], np.stack(synchronies, 1), atol=2e-7)
    np.testing.assert_allclose(prior['logit'][:, 0], expected_prior, atol=3e-7)
    np.testing.assert_allclose(post['logit'][:, 0], expected_post, atol=3e-7)

  def test_observe_interfaces_jit_and_scan(self):
    observed = self.apply(self.core.observe_with_ticks, self.embed, self.action, self.first)
    ordinary = self.apply(self.core.observe, self.embed, self.action, self.first)
    for left, right in zip(jax.tree_util.tree_leaves(observed[:2]), jax.tree_util.tree_leaves(ordinary)):
      np.testing.assert_array_equal(left, right)
    function = nj.pure(self.core.observe_with_ticks)
    compiled = jax.jit(lambda p, x, a, f: function(
        p, jax.random.PRNGKey(1), x, a, f, create=False, modify=False)[0])
    result = compiled(self.parameters, self.embed, self.action, self.first)
    for left, right in zip(jax.tree_util.tree_leaves(observed), jax.tree_util.tree_leaves(result)):
      np.testing.assert_allclose(left, right, atol=3e-6, rtol=3e-5)
    program = str(jax.make_jaxpr(lambda x: compiled(self.parameters, x, self.action, self.first))(self.embed))
    self.assertIn('scan[', program)

  def test_tick_loss_reaches_all_core_parameter_groups(self):
    def objective(parameters):
      (_, _, diagnostics), _ = nj.pure(self.core.observe_with_ticks)(
          parameters, jax.random.PRNGKey(2), self.embed, self.action,
          self.first, create=False, modify=False)
      return diagnostics['tick_loss'].mean()
    derivatives = jax.grad(objective)(self.parameters)
    groups = ('activity_initializer', 'history_initializer', 'synapse_hidden',
              'synapse_output', 'nlm/w1', 'nlm/w2', 'sync_beta',
              'sync_projection', 'prior_hidden', 'prior_stats',
              'posterior_hidden', 'posterior_stats')
    for group in groups:
      with self.subTest(group=group):
        selected = [value for name, value in derivatives.items() if group in name]
        self.assertTrue(selected, group)
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in selected))
        self.assertGreater(sum(float(jnp.linalg.norm(value)) for value in selected), 0., group)

  def test_inherited_r2i_losses_and_imagination(self):
    post, prior, _ = self.apply(self.core.observe_with_ticks, self.embed, self.action, self.first)
    for method in (self.core.dyn_loss, self.core.rep_loss):
      value = self.apply(method, post, prior)
      self.assertEqual(value.shape, (1, 4))
      self.assertTrue(np.all(np.asarray(value) >= 1.0))
    state = {key: value[:, -1] for key, value in post.items()}
    imagined = self.apply(self.core.img_step, state, self.action[:, 0])
    self.assertEqual(set(imagined), {'deter', 'stoch', 'logit'})
    np.testing.assert_array_equal(imagined['stoch'].sum(-1), np.ones((1, 2)))

  def test_invalid_configuration_rejected(self):
    for kwargs in ({'classes': 1}, {'sync_neurons': 65}, {'ticks': 0},
                   {'sequence_unroll': True}, {'tick_mode': 'minimum'},
                   {'unimix': -1.0}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        NRSM(**kwargs, name='invalid')

  def test_constructor_allows_jaxagent_transfer_guard(self):
    # The production JAXAgent sets this guard while constructing WorldModel;
    # implicit jnp.asarray(host_pairs) fails even though ordinary tests pass.
    with jax.transfer_guard('disallow'):
      guarded = NRSM(neurons=4, sync_neurons=3, name='guarded_nrsm')
    np.testing.assert_array_equal(
        np.asarray(guarded._pairs), [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])


if __name__ == '__main__':
  unittest.main()
