"""Numerical and recurrent-state contracts for the CT-WM/R2I core."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recall2imagine import ninjax as nj
from recall2imagine.nrsm_nets import (
    NRSM, PrivateNLM, tick_statistics, window_synchronization)


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

  def test_window_synchronization_matches_numpy_and_decay_gradient(self):
    trace = jnp.asarray([[.2, .9, -.4], [-.5, .1, .8], [.7, -.3, .6]])
    pairs = jnp.asarray([[0, 0], [0, 1], [1, 2]], jnp.int32)
    beta = jnp.asarray([.2, .3, .8])
    def explicit(decay_parameter):
      ages = np.arange(trace.shape[-1] - 1, -1, -1)
      weights = np.exp(-np.clip(np.asarray(decay_parameter), 0, 4)[:, None] * ages)
      products = np.asarray(trace)[np.asarray(pairs)[:, 0]] * np.asarray(trace)[np.asarray(pairs)[:, 1]]
      return (weights * products).sum(-1) / np.sqrt(weights.sum(-1))
    value = lambda decay_parameter: window_synchronization(trace, decay_parameter, pairs)
    np.testing.assert_allclose(value(beta), explicit(beta), atol=1e-7)
    derivative = np.asarray(jax.jacrev(value)(beta))
    eps = 1e-3
    for index in range(len(beta)):
      offset = np.eye(len(beta))[index] * eps
      expected = (explicit(np.asarray(beta) + offset) - explicit(np.asarray(beta) - offset)) / (2 * eps)
      np.testing.assert_allclose(derivative[:, index], expected, atol=2e-5)
    # Clipping and zero initialization follow the official RL window variant.
    np.testing.assert_allclose(value(jnp.asarray([-2., 9., 0.])), explicit([-2., 9., 0.]), atol=1e-7)
    no_decay = window_synchronization(trace, jnp.zeros(3), pairs)
    products = trace[pairs[:, 0]] * trace[pairs[:, 1]]
    np.testing.assert_allclose(no_decay, products.sum(-1) / np.sqrt(3), atol=1e-7)
    np.testing.assert_allclose(window_synchronization(trace[None], beta, pairs)[0], value(beta), atol=1e-7)
    np.testing.assert_array_equal(window_synchronization(jnp.zeros_like(trace), beta, pairs), np.zeros(3))

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
    self.core = NRSM(neurons=4, memory=5, ticks=3, deter=5, stoch=2,
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

  def test_structured_memory_and_no_posterior_memory_correction(self):
    post, prior, diagnostics = self.apply(
        self.core.observe_with_ticks, self.embed, self.action, self.first)
    self.assertEqual(set(post), {'deter', 'stoch', 'logit', 'pre_trace', 'post_trace'})
    self.assertEqual(set(prior), set(post))
    self.assertEqual(set(self.core.initial(1)), set(post))
    np.testing.assert_array_equal(post['deter'], prior['deter'])
    for key in ('pre_trace', 'post_trace'):
      self.assertEqual(post[key].shape, (1, 4, 4, 5))
      np.testing.assert_array_equal(post[key], prior[key])
    for value in self.core.initial(1).values():
      np.testing.assert_array_equal(value, np.zeros(value.shape))
    np.testing.assert_array_equal(post['deter'], diagnostics['tick_contexts'][..., -1, :])
    self.assertEqual(diagnostics['tick_kl'].shape, (1, 4, 3))
    self.assertEqual(diagnostics['tick_loss'].shape, (1, 4))
    np.testing.assert_array_equal(post['stoch'].sum(-1), np.ones((1, 4, 2)))
    self.assertTrue(np.all(np.isin(np.asarray(post['stoch']), [0., 1.])))
    self.assertFalse(any('posterior_gate' in key or 'candidate' in key for key in self.parameters))
    self.assertFalse(any('activity_initializer' in key or 'history_initializer' in key for key in self.parameters))
    beta = next(value for key, value in self.parameters.items() if key.endswith('/sync_beta'))
    np.testing.assert_array_equal(beta, np.zeros(6))

  def test_reset_discards_incoming_memory_and_action(self):
    initial = self.core.initial(1)
    nonzero = {key: jnp.ones_like(value) * 3 for key, value in initial.items()}
    args = self.embed[:, 0], jnp.ones(1, jnp.bool_)
    reset = self.apply(self.core.obs_step, nonzero, self.action[:, 0], *args)
    fresh = self.apply(self.core.obs_step, initial, jnp.zeros_like(self.action[:, 0]), *args)
    for first, second in zip(jax.tree_util.tree_leaves(reset), jax.tree_util.tree_leaves(fresh)):
      np.testing.assert_array_equal(first, second)

  def test_reset_is_per_environment_and_preserves_other_rows(self):
    initial = self.core.initial(2)
    previous = {key: jnp.ones_like(value) * .3 for key, value in initial.items()}
    actions = jnp.asarray([[.2, -.1], [.4, .3]])
    embed = jnp.repeat(self.embed[:, 0], 2, 0)
    first = jnp.asarray([True, False])
    result = self.apply(self.core.obs_step, previous, actions, embed, first)
    manual_previous = {key: value.at[0].set(0) for key, value in previous.items()}
    manual = self.apply(self.core.obs_step, manual_previous, actions.at[0].set(0),
                        embed, jnp.zeros(2, jnp.bool_))
    for left, right in zip(jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(manual)):
      np.testing.assert_array_equal(left, right)
    post, _ = result
    for key in ('pre_trace', 'post_trace'):
      np.testing.assert_array_equal(post[key][0, :, :2], np.zeros((4, 2)))
      np.testing.assert_array_equal(post[key][1, :, :2], np.full((4, 2), .3, np.float32))

  def test_prior_does_not_use_current_observation_or_hidden_python_buffers(self):
    initial = self.core.initial(1)
    first = jnp.zeros(1, jnp.bool_)
    post1, prior1 = self.apply(self.core.obs_step, initial, self.action[:, 0], self.embed[:, 0], first)
    post2, prior2 = self.apply(self.core.obs_step, initial, self.action[:, 0], self.embed[:, 0] + 10, first)
    for key in prior1:
      np.testing.assert_array_equal(prior1[key], prior2[key])
    self.assertGreater(float(jnp.abs(post1['logit'] - post2['logit']).max()), .001)
    # The same explicit persistent state/action reproduces all trajectories,
    # regardless of other calls between them.
    before = self.apply(self.core.prior_context, initial, self.action[:, 0])
    self.apply(self.core.prior_context, initial, self.action[:, 0] * 2)
    after = self.apply(self.core.prior_context, initial, self.action[:, 0])
    for left, right in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)):
      np.testing.assert_array_equal(left, right)

  def test_both_histories_continue_across_environment_steps(self):
    initial = self.core.initial(1)
    values = jnp.arange(20, dtype=jnp.float32).reshape(1, 4, 5) / 20
    previous = dict(initial, pre_trace=values, post_trace=-values)
    memory, _ = self.apply(self.core._transition, previous, self.action[:, 0])
    for key in ('pre_trace', 'post_trace'):
      np.testing.assert_array_equal(memory[key][..., :2], previous[key][..., 3:])
    next_state = dict(initial, **memory)
    following, _ = self.apply(self.core._transition, next_state, self.action[:, 1])
    for key in ('pre_trace', 'post_trace'):
      np.testing.assert_array_equal(following[key][..., :2], memory[key][..., 3:])
    erased = dict(previous, pre_trace=jnp.zeros_like(values), post_trace=jnp.zeros_like(values))
    without_memory, _ = self.apply(self.core._transition, erased, self.action[:, 0])
    self.assertGreater(float(jnp.linalg.norm(memory['deter'] - without_memory['deter'])), 1e-5)

    post, prior, _ = self.apply(self.core.observe_with_ticks, self.embed, self.action, self.first)
    first_state = {key: value[:, 0] for key, value in post.items()}
    _, expected = self.apply(self.core.obs_step, first_state, self.action[:, 1],
                             self.embed[:, 1], self.first[:, 1])
    for key in expected:
      np.testing.assert_allclose(prior[key][:, 1], expected[key], atol=3e-7)

  def test_transition_and_logits_match_independent_numpy_formulas(self):
    parameters = {key: np.asarray(value) for key, value in self.parameters.items()}
    beta_key = next(key for key in parameters if key.endswith('/sync_beta'))
    parameters[beta_key] = np.asarray([-.3, 0., .2, .5, 1., 5.], np.float32)
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
                   logit=jnp.zeros((1, 2, 3)),
                   pre_trace=jnp.arange(20, dtype=jnp.float32).reshape(1, 4, 5) / 50,
                   post_trace=-jnp.arange(20, dtype=jnp.float32).reshape(1, 4, 5) / 50)
    u = np.concatenate((np.asarray(initial['deter']),
                        np.asarray(initial['stoch']).reshape(1, -1),
                        np.asarray(self.action[:, 0])), -1)
    history = np.asarray(initial['pre_trace'])
    post_trace = np.asarray(initial['post_trace'])
    activity = post_trace[..., -1]
    pairs = np.asarray([(i, j) for i in range(3) for j in range(i, 3)])
    ages = np.arange(4, -1, -1, dtype=np.float32)
    weights = np.exp(-np.clip(parameters[beta_key], 0, 4)[:, None] * ages)
    contexts, activities, synchronies = [], [], []
    for tick in range(3):
      alpha = dense('synapse_hidden', np.concatenate((activity, u), -1), 'elu')
      alpha = dense('synapse_output', alpha)
      history = np.concatenate((history[..., 1:], alpha[..., None]), -1)
      hidden = np.einsum('bdm,dmh->bdh', history, parameter('nlm/w1')) + parameter('nlm/b1')
      hidden = np.where(hidden > 0, hidden, np.expm1(hidden))
      activity = np.tanh((hidden * parameter('nlm/w2')).sum(-1) + parameter('nlm/b2'))
      post_trace = np.concatenate((post_trace[..., 1:], activity[..., None]), -1)
      products = post_trace[:, pairs[:, 0], :] * post_trace[:, pairs[:, 1], :]
      sync = (weights[None] * products).sum(-1) / np.sqrt(weights.sum(-1))
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
    np.testing.assert_allclose(post['pre_trace'][:, 0], history, atol=2e-7)
    np.testing.assert_allclose(post['post_trace'][:, 0], post_trace, atol=2e-7)
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
    groups = ('synapse_hidden', 'synapse_output', 'nlm/w1', 'nlm/w2', 'sync_beta',
              'sync_projection', 'prior_hidden', 'prior_stats',
              'posterior_hidden', 'posterior_stats')
    for group in groups:
      with self.subTest(group=group):
        selected = [value for name, value in derivatives.items() if group in name]
        self.assertTrue(selected, group)
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in selected))
        self.assertGreater(sum(float(jnp.linalg.norm(value)) for value in selected), 0., group)

  def test_gradients_reach_previous_histories_across_environment_steps(self):
    initial = self.core.initial(1)
    values = jnp.arange(20, dtype=jnp.float32).reshape(1, 4, 5) / 20
    def objective(pre_trace, post_trace):
      def advance_twice():
        previous = dict(initial, pre_trace=pre_trace, post_trace=post_trace)
        memory, _ = self.core._transition(previous, self.action[:, 0])
        memory, _ = self.core._transition(dict(initial, **memory), self.action[:, 1])
        return memory['deter'].sum()
      return nj.pure(advance_twice)(self.parameters, jax.random.PRNGKey(7),
                                    create=False, modify=False)[0]
    gradients = jax.jit(jax.grad(objective, (0, 1)))(values, -values)
    for derivative in gradients:
      self.assertTrue(np.isfinite(np.asarray(derivative)).all())
      self.assertGreater(float(jnp.linalg.norm(derivative)), 1e-7)

  def test_inherited_r2i_losses_and_imagination(self):
    post, prior, _ = self.apply(self.core.observe_with_ticks, self.embed, self.action, self.first)
    for method in (self.core.dyn_loss, self.core.rep_loss):
      value = self.apply(method, post, prior)
      self.assertEqual(value.shape, (1, 4))
      self.assertTrue(np.all(np.asarray(value) >= 1.0))
    state = {key: value[:, -1] for key, value in post.items()}
    imagined = self.apply(self.core.img_step, state, self.action[:, 0])
    self.assertEqual(set(imagined), set(state))
    np.testing.assert_array_equal(imagined['stoch'].sum(-1), np.ones((1, 2)))
    # Filtering and imagination share the entire deterministic transition;
    # only the distribution used to sample the stochastic state differs.
    _, real_prior = self.apply(self.core.obs_step, state, self.action[:, 0],
                               self.embed[:, 0], jnp.zeros(1, jnp.bool_))
    for key in ('deter', 'pre_trace', 'post_trace', 'logit'):
      np.testing.assert_allclose(imagined[key], real_prior[key], atol=3e-7)

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
