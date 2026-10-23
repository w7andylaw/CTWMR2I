"""Numerical and recurrent-state contracts for the CT-WM/R2I core."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from recall2imagine import ninjax as nj
from recall2imagine.nrsm_nets import (
    NRSM, PrivateNLM, _NavigationSynapseBlock, tick_statistics,
    window_synchronization)






def numpy_navigation_synapse(parameter, inputs):
  """Float64 oracle for PyTorch's default GLU and affine LayerNorm equations."""
  linear = np.asarray(inputs, np.float64) @ np.asarray(parameter('kernel'), np.float64)
  linear += np.asarray(parameter('bias'), np.float64)
  value, gate = np.split(linear, 2, axis=-1)
  glu = value / (1. + np.exp(-gate))
  mean = np.mean(glu, axis=-1, keepdims=True)
  variance = np.mean(np.square(glu - mean), axis=-1, keepdims=True)
  normalized = (glu - mean) / np.sqrt(variance + 1e-5)
  return normalized * parameter('norm_scale') + parameter('norm_bias')


class NavigationSynapseTest(unittest.TestCase):

  def test_forward_matches_independent_numpy_for_batch_and_sequence_axes(self):
    block = _NavigationSynapseBlock(4, name='navigation')
    apply = nj.pure(block.__call__)
    rng = np.random.default_rng(391)
    inputs = rng.normal(size=(2, 3, 6)).astype(np.float32)
    _, parameters = apply({}, jax.random.PRNGKey(90), jnp.asarray(inputs))
    self.assertEqual(parameters['navigation/kernel'].shape, (6, 8))
    self.assertEqual(parameters['navigation/bias'].shape, (8,))
    for name in ('kernel', 'bias'):
      self.assertLessEqual(float(jnp.abs(parameters['navigation/' + name]).max()),
                           1 / math.sqrt(6))
    # Original Linear biases are sampled uniformly rather than zero-filled.
    self.assertGreater(float(jnp.linalg.norm(parameters['navigation/bias'])), 0.)
    np.testing.assert_array_equal(parameters['navigation/norm_scale'], np.ones(4))
    np.testing.assert_array_equal(parameters['navigation/norm_bias'], np.zeros(4))
    parameters['navigation/norm_scale'] = jnp.asarray([.5, -1.2, 1.8, .7])
    parameters['navigation/norm_bias'] = jnp.asarray([.2, -.3, .1, .6])
    for shape in ((2, 6), (2, 3, 6)):
      for small_variance in (False, True):
        with self.subTest(shape=shape, small_variance=small_variance):
          weights = dict(parameters)
          if small_variance:
            weights['navigation/kernel'] *= 1e-3
            weights['navigation/bias'] *= 1e-3
          data = inputs[:, 0] if len(shape) == 2 else inputs
          expected = numpy_navigation_synapse(
              lambda name: np.asarray(weights['navigation/' + name]), data)
          actual, _ = jax.jit(lambda weights, data: apply(
              weights, jax.random.PRNGKey(91), data, create=False, modify=False))(
                  weights, jnp.asarray(data))
          np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=3e-6)
          self.assertEqual(actual.shape, shape[:-1] + (4,))
          self.assertEqual(actual.dtype, jnp.float32)

  def test_value_gate_and_affine_gradients_match_independent_finite_differences(self):
    block = _NavigationSynapseBlock(4, name='navigation_grad')
    apply = nj.pure(block.__call__)
    inputs = np.random.default_rng(481).normal(size=(2, 5)).astype(np.float32)
    _, parameters = apply({}, jax.random.PRNGKey(92), jnp.asarray(inputs))
    parameters['navigation_grad/norm_scale'] = jnp.asarray([.5, -1.2, 1.8, .7])
    objective_weights = np.asarray([[.2, -.7, 1.1, .4], [-.3, .5, .9, -.8]])
    def objective(weights, data):
      result, _ = apply(weights, jax.random.PRNGKey(93), data,
                         create=False, modify=False)
      return (result * jnp.asarray(objective_weights, jnp.float32)).sum()
    derivatives, input_derivative = jax.grad(objective, (0, 1))(
        parameters, jnp.asarray(inputs))
    kernel_grad = np.asarray(derivatives['navigation_grad/kernel'])
    for half in (kernel_grad[:, :4], kernel_grad[:, 4:]):
      self.assertTrue(np.isfinite(half).all())
      self.assertGreater(np.linalg.norm(half), 1e-5)
    def oracle(weights, data):
      value = numpy_navigation_synapse(
          lambda name: np.asarray(weights['navigation_grad/' + name]), data)
      return (value * objective_weights).sum()
    eps = .001
    for name, index in (('kernel', (0, 0)), ('kernel', (0, 4)),
                        ('bias', (6,)), ('norm_scale', (1,)), ('norm_bias', (2,))):
      key = 'navigation_grad/' + name
      left = {key: np.asarray(value, np.float64).copy()
              for key, value in parameters.items()}
      right = {key: value.copy() for key, value in left.items()}
      left[key][index] += eps
      right[key][index] -= eps
      expected = (oracle(left, inputs) - oracle(right, inputs)) / (2 * eps)
      self.assertAlmostEqual(float(derivatives[key][index]), expected, delta=2e-4)
    left, right = inputs.astype(np.float64), inputs.astype(np.float64)
    left[0, 1] += eps
    right[0, 1] -= eps
    expected = (oracle(parameters, left) - oracle(parameters, right)) / (2 * eps)
    self.assertAlmostEqual(float(input_derivative[0, 1]), expected, delta=2e-4)

  def test_default_two_blocks_use_neuron_width_independently_of_units(self):
    core = NRSM(name='navigation_default')
    def run():
      return core._transition(core.initial(1), jnp.zeros((1, 2), jnp.float32))
    (_, diagnostics), parameters = nj.pure(run)({}, jax.random.PRNGKey(94))
    self.assertEqual(diagnostics['tick_activities'].shape, (1, 4, 64))
    for block, fan_in in (('synapse_hidden', 258), ('synapse_output', 64)):
      prefix = 'navigation_default/' + block + '/'
      self.assertEqual(parameters[prefix + 'kernel'].shape, (fan_in, 128))
      self.assertEqual(parameters[prefix + 'bias'].shape, (128,))
      self.assertEqual(parameters[prefix + 'norm_scale'].shape, (64,))
      self.assertEqual(parameters[prefix + 'norm_bias'].shape, (64,))
    self.assertEqual(parameters['navigation_default/sync_projection/kernel'].shape,
                     (64 + 136, 64))
    np.testing.assert_array_equal(diagnostics['tick_write_gates'], .5)
    self.assertFalse(any('slow_memory' in key or 'memory_backbone' in key
                         or 'ltm' in key for key in parameters))
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
    raw = jnp.asarray([-6., -.3, 2.])
    def explicit(decay_parameter):
      ages = np.arange(trace.shape[-1] - 1, -1, -1)
      decay = 4. / (1. + np.exp(-np.asarray(decay_parameter)))
      weights = np.exp(-decay[:, None] * ages)
      products = np.asarray(trace)[np.asarray(pairs)[:, 0]] * np.asarray(trace)[np.asarray(pairs)[:, 1]]
      return (weights * products).sum(-1) / np.sqrt(weights.sum(-1))
    value = lambda decay_parameter: window_synchronization(trace, decay_parameter, pairs)
    np.testing.assert_allclose(value(raw), explicit(raw), atol=1e-7)
    derivative = np.asarray(jax.jacrev(value)(raw))
    eps = 1e-3
    for index in range(len(raw)):
      offset = np.eye(len(raw))[index] * eps
      expected = (explicit(np.asarray(raw) + offset) - explicit(np.asarray(raw) - offset)) / (2 * eps)
      np.testing.assert_allclose(derivative[:, index], expected, atol=2e-5)
    # Both sides of the old hard-clamp interval remain differentiable.
    outside = jnp.asarray([-6., -2., 5.])
    np.testing.assert_allclose(value(outside), explicit(outside), atol=1e-7)
    outside_gradient = np.diag(np.asarray(jax.jacrev(value)(outside)))
    self.assertTrue(np.all(np.abs(outside_gradient) > 1e-7))
    no_decay = window_synchronization(trace, jnp.full(3, -40.), pairs)
    products = trace[pairs[:, 0]] * trace[pairs[:, 1]]
    np.testing.assert_allclose(no_decay, products.sum(-1) / np.sqrt(3), atol=1e-7)
    np.testing.assert_allclose(window_synchronization(trace[None], raw, pairs)[0], value(raw), atol=1e-7)
    np.testing.assert_array_equal(window_synchronization(jnp.zeros_like(trace), raw, pairs), np.zeros(3))

  def test_private_nlm_has_no_cross_neuron_connections(self):
    nlm = PrivateNLM(3, 4, 2, name='private_nlm')
    history = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
    apply = nj.pure(nlm.__call__)
    _, parameters = apply({}, jax.random.PRNGKey(0), history)
    # Nonzero gate weights test neuron locality for both independent outputs;
    # the production initializer below is separately checked to produce .5.
    np.testing.assert_array_equal(parameters['private_nlm/write_w'], np.zeros((3, 2)))
    np.testing.assert_array_equal(parameters['private_nlm/write_b'], np.zeros(3))
    (candidate, gate), _ = apply(parameters, jax.random.PRNGKey(1), history,
                                 create=False, modify=False)
    np.testing.assert_array_equal(gate, np.full(3, .5))
    parameters['private_nlm/write_w'] = jnp.asarray([[.2, -.4], [.1, .3], [-.2, .5]])
    value = lambda inputs: jnp.stack(apply(
        parameters, jax.random.PRNGKey(1), inputs, create=False, modify=False)[0], 0)
    derivative = np.asarray(jax.jacrev(value)(history))
    for output in range(2):
      for output_neuron in range(3):
        for input_neuron in range(3):
          if output_neuron != input_neuron:
            np.testing.assert_array_equal(
                derivative[output, output_neuron, input_neuron], np.zeros(4))
        self.assertGreater(np.linalg.norm(derivative[output, output_neuron, output_neuron]), 0.)
    w1 = next(v for k, v in parameters.items() if k.endswith('/w1'))
    w2 = next(v for k, v in parameters.items() if k.endswith('/w2'))
    self.assertLessEqual(float(jnp.abs(w1).max()), math.sqrt(6 / (4 + 2)))
    self.assertLessEqual(float(jnp.abs(w2).max()), math.sqrt(6 / (2 + 1)))

  def test_private_candidate_and_gate_match_numpy_and_gate_gradients(self):
    nlm = PrivateNLM(3, 4, 2, name='gated_nlm')
    apply = nj.pure(nlm.__call__)
    history = np.random.default_rng(418).normal(size=(2, 3, 3, 4)).astype(np.float32)
    _, parameters = apply({}, jax.random.PRNGKey(8), jnp.asarray(history))
    parameters['gated_nlm/write_w'] = jnp.asarray([[.2, -.4], [.1, .3], [-.2, .5]])
    parameters['gated_nlm/write_b'] = jnp.asarray([-.3, .6, -.8])
    def oracle(weights):
      p = lambda name: np.asarray(weights['gated_nlm/' + name], np.float64)
      hidden = np.einsum('...dm,dmh->...dh', history, p('w1')) + p('b1')
      hidden = np.where(hidden > 0, hidden, np.expm1(hidden))
      candidate = np.tanh((hidden * p('w2')).sum(-1) + p('b2'))
      gate = 1. / (1. + np.exp(-((hidden * p('write_w')).sum(-1) + p('write_b'))))
      return candidate, gate
    actual, _ = jax.jit(lambda p: apply(
        p, jax.random.PRNGKey(9), jnp.asarray(history), create=False, modify=False))(parameters)
    for value, expected in zip(actual, oracle(parameters)):
      self.assertEqual(value.shape, (2, 3, 3))
      np.testing.assert_allclose(value, expected, atol=2e-7, rtol=1e-6)
    def objective(weights):
      (candidate, gate), _ = apply(weights, jax.random.PRNGKey(9), jnp.asarray(history),
                                   create=False, modify=False)
      return (gate * candidate).sum()
    derivative = jax.grad(objective)(parameters)
    def numpy_objective(weights):
      candidate, gate = oracle(weights)
      return (candidate * gate).sum()
    for name, index in (('write_w', (1, 0)), ('write_b', (2,)), ('w1', (0, 1, 1))):
      key = 'gated_nlm/' + name
      left = {key: np.asarray(value, np.float64).copy() for key, value in parameters.items()}
      right = {key: value.copy() for key, value in left.items()}
      left[key][index] += .001
      right[key][index] -= .001
      expected = (numpy_objective(left) - numpy_objective(right)) / .002
      self.assertAlmostEqual(float(derivative[key][index]), expected, delta=3e-5)




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
    initial = self.apply(self.core.initial, 1)
    self.assertEqual(set(initial), set(post))
    np.testing.assert_array_equal(post['deter'], prior['deter'])
    for key in ('pre_trace', 'post_trace'):
      self.assertEqual(post[key].shape, (1, 4, 4, 5))
      np.testing.assert_array_equal(post[key], prior[key])
    for key, value in initial.items():
      np.testing.assert_array_equal(value, np.zeros(value.shape))
    np.testing.assert_array_equal(post['deter'], diagnostics['tick_contexts'][..., -1, :])
    self.assertEqual(diagnostics['tick_kl'].shape, (1, 4, 3))
    self.assertEqual(diagnostics['tick_loss'].shape, (1, 4))
    self.assertNotIn('tick_queries', diagnostics)
    self.assertNotIn('tick_memory_reads', diagnostics)
    self.assertNotIn('ssm_read', diagnostics)
    np.testing.assert_array_equal(post['stoch'].sum(-1), np.ones((1, 4, 2)))
    self.assertTrue(np.all(np.isin(np.asarray(post['stoch']), [0., 1.])))
    self.assertFalse(any('posterior_gate' in key or 'candidate' in key for key in self.parameters))
    self.assertFalse(any('activity_initializer' in key or 'history_initializer' in key for key in self.parameters))
    self.assertFalse(any('slow_memory' in key or 'ltm' in key or 'memory_backbone' in key
                         for key in self.parameters))
    self.assertFalse(any('memory_read' in key for key in self.parameters))
    decay_raw = next(value for key, value in self.parameters.items()
                     if key.endswith('/sync_decay_raw'))
    np.testing.assert_allclose(4. * jax.nn.sigmoid(decay_raw), .01, atol=2e-9)
    self.assertEqual(self.parameters['nrsm/time_constant_raw'].shape, (4,))
    self.assertEqual(self.parameters['nrsm/sync_projection/kernel'].shape, (10, 1))
    np.testing.assert_array_equal(post['deter'][..., :4], post['post_trace'][..., -1])
    np.testing.assert_array_equal(diagnostics['tick_contexts'][..., :4],
                                  diagnostics['tick_activities'])
    np.testing.assert_array_equal(diagnostics['tick_write_gates'], .5)
    self.assertEqual(diagnostics['tick_candidates'].shape, (1, 4, 3, 4))

  def test_reset_discards_incoming_memory_and_action(self):
    initial = self.apply(self.core.initial, 1)
    nonzero = {key: jnp.ones_like(value) * 3 for key, value in initial.items()}
    args = self.embed[:, 0], jnp.ones(1, jnp.bool_)
    reset = self.apply(self.core.obs_step, nonzero, self.action[:, 0], *args)
    fresh = self.apply(self.core.obs_step, initial, jnp.zeros_like(self.action[:, 0]), *args)
    for first, second in zip(jax.tree_util.tree_leaves(reset), jax.tree_util.tree_leaves(fresh)):
      np.testing.assert_array_equal(first, second)

  def test_reset_is_per_environment_and_preserves_other_rows(self):
    initial = self.apply(self.core.initial, 2)
    previous = {key: jnp.ones_like(value) * .3 for key, value in initial.items()}
    actions = jnp.asarray([[.2, -.1], [.4, .3]])
    embed = jnp.repeat(self.embed[:, 0], 2, 0)
    first = jnp.asarray([True, False])
    result = self.apply(self.core.obs_step, previous, actions, embed, first)
    manual_previous = {key: value.at[0].set(initial[key][0])
                       for key, value in previous.items()}
    manual = self.apply(self.core.obs_step, manual_previous, actions.at[0].set(0),
                        embed, jnp.zeros(2, jnp.bool_))
    for left, right in zip(jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(manual)):
      np.testing.assert_array_equal(left, right)
    post, _ = result
    for key in ('pre_trace', 'post_trace'):
      np.testing.assert_array_equal(post[key][0, :, :2], np.zeros((4, 2)))
      np.testing.assert_array_equal(post[key][1, :, :2], np.full((4, 2), .3, np.float32))

  def test_reset_cuts_previous_episode_memory_and_action_gradients_per_row(self):
    initial = self.apply(self.core.initial, 2)
    previous = {key: jnp.ones_like(value) * .3 for key, value in initial.items()}
    action = jnp.asarray([[.2, -.1], [.4, .3]])
    embed = jnp.repeat(self.embed[:, 0], 2, 0)
    def objective(pre_trace, post_trace, actions):
      state = dict(previous, pre_trace=pre_trace, post_trace=post_trace)
      (post, _), _ = nj.pure(self.core.obs_step)(
          self.parameters, jax.random.PRNGKey(19), state, actions, embed,
          jnp.asarray([True, False]), create=False, modify=False)
      return (post['deter'] * jnp.arange(1., 6.)).sum()
    derivatives = jax.grad(objective, (0, 1, 2))(
        previous['pre_trace'], previous['post_trace'], action)
    for derivative in derivatives:
      np.testing.assert_array_equal(derivative[0], np.zeros_like(derivative[0]))
      self.assertTrue(np.isfinite(np.asarray(derivative[1])).all())
      self.assertGreater(float(jnp.linalg.norm(derivative[1])), 1e-7)

  def test_prior_does_not_use_current_observation_or_hidden_python_buffers(self):
    initial = self.apply(self.core.initial, 1)
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
    initial = self.apply(self.core.initial, 1)
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
      # Compiled scan and eager GLU/LayerNorm reductions can round differently
      # in float32. FIFO retention above remains an exact identity check.
      np.testing.assert_allclose(prior[key][:, 1], expected[key], atol=1e-6)

  def test_time_constants_are_positive_bounded_and_log_spaced(self):
    time_constant, retention = self.apply(self.core.memory_coefficients)
    expected = np.exp(np.log(.1) + np.linspace(.01, .99, 4) * np.log(1000.))
    np.testing.assert_allclose(time_constant, expected, atol=2e-5, rtol=8e-7)
    np.testing.assert_allclose(retention ** 3, np.exp(-1 / expected),
                               atol=2e-7, rtol=2e-6)
    self.assertTrue(np.all(np.diff(np.asarray(time_constant)) > 0.))
    self.assertEqual(time_constant.dtype, jnp.float32)
    parameters = dict(self.parameters)
    parameters['nrsm/time_constant_raw'] = jnp.asarray([-60., -1., 1., 60.])
    (bounded, _), _ = nj.pure(self.core.memory_coefficients)(
        parameters, jax.random.PRNGKey(5), create=False, modify=False)
    self.assertTrue(np.all(np.asarray(bounded) >= .1 * (1 - 1e-6)))
    self.assertTrue(np.all(np.asarray(bounded) <= 100. * (1 + 1e-6)))
    single = NRSM(neurons=1, sync_neurons=1, name='single_timescale')
    (one, _), _ = nj.pure(single.memory_coefficients)({}, jax.random.PRNGKey(5))
    np.testing.assert_allclose(one, np.sqrt(.1 * 100.), rtol=3e-7)

  def test_direct_activity_retention_is_calibrated_in_environment_time(self):
    # Remove only the learned candidate. This independently isolates the
    # direct memory path from all nonlinear synapse/NLM feedback.
    parameters = dict(self.parameters)
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.zeros_like(parameters['nrsm/nlm/b2'])
    previous = self.apply(self.core.initial, 1)
    activity = jnp.asarray([[.3, -.5, .7, -.9]])
    previous = dict(previous, post_trace=jnp.broadcast_to(activity[..., None], (1, 4, 5)))
    raw = np.asarray(parameters['nrsm/time_constant_raw'])
    time_constant = np.exp(np.log(.1) + np.log(1000.) / (1. + np.exp(-raw)))
    expected = np.asarray(activity) * np.exp(-1. / time_constant)
    for ticks in (1, 3, 7):
      core = NRSM(neurons=4, memory=5, ticks=ticks, deter=5, stoch=2,
                  classes=3, units=7, nlm_hidden=3, sync_neurons=3, name='nrsm')
      (memory, diagnostics), _ = nj.pure(core._transition)(
          parameters, jax.random.PRNGKey(5), previous, self.action[:, 0],
          create=False, modify=False)
      self.assertEqual(diagnostics['tick_activities'].shape, (1, ticks, 4))
      np.testing.assert_allclose(memory['post_trace'][..., -1], expected,
                                 atol=2e-7, rtol=3e-6)
      # A hidden LayerNorm on persisted activity would break this scaling.
      scaled = dict(previous, post_trace=previous['post_trace'] * 3.)
      (scaled_memory, _), _ = nj.pure(core._transition)(
          parameters, jax.random.PRNGKey(5), scaled, self.action[:, 0],
          create=False, modify=False)
      np.testing.assert_allclose(scaled_memory['post_trace'][..., -1], expected * 3.,
                                 atol=4e-7, rtol=3e-6)
    def retained_sum(raw_parameter):
      selected = dict(parameters, **{'nrsm/time_constant_raw': raw_parameter})
      (memory, _), _ = nj.pure(self.core._transition)(
          selected, jax.random.PRNGKey(5), previous, self.action[:, 0],
          create=False, modify=False)
      return memory['post_trace'][..., -1].sum()
    derivative = jax.grad(retained_sum)(parameters['nrsm/time_constant_raw'])
    fraction = 1. / (1. + np.exp(-raw))
    expected_derivative = expected[0] / time_constant * np.log(1000.) * fraction * (1. - fraction)
    np.testing.assert_allclose(derivative, expected_derivative, atol=2e-8, rtol=1e-5)
    # Six environment steps exceed the five-entry FIFO window: old activity
    # survives through recurrence, without storing a separate long-term tensor.
    current = previous
    for _ in range(6):
      (memory, _), _ = nj.pure(self.core._transition)(
          parameters, jax.random.PRNGKey(5), current, self.action[:, 0],
          create=False, modify=False)
      current = dict(previous, **memory)
    np.testing.assert_allclose(current['post_trace'][..., -1],
                               np.asarray(activity) * np.exp(-6. / time_constant),
                               atol=3e-7, rtol=5e-6)

  def test_activity_readout_preserves_first_order_sign_information(self):
    parameters = dict(self.parameters)
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.zeros_like(parameters['nrsm/nlm/b2'])
    # Pairwise synchrony is unchanged by a global sign flip. Raw activity is
    # nevertheless available exactly, independently of learned projection.
    projection = jnp.zeros_like(parameters['nrsm/sync_projection/kernel'])
    projection = projection.at[0, 0].set(1.)
    parameters['nrsm/sync_projection/kernel'] = projection
    parameters['nrsm/sync_projection/bias'] = jnp.zeros(1, jnp.float32)
    previous = self.apply(self.core.initial, 1)
    activity = jnp.asarray([[.2, -.4, .6, -.8]])
    previous = dict(previous, post_trace=jnp.broadcast_to(activity[..., None], (1, 4, 5)))
    transition = nj.pure(self.core._transition)
    (left, dleft), _ = transition(parameters, jax.random.PRNGKey(5), previous,
                                  self.action[:, 0], create=False, modify=False)
    (right, dright), _ = transition(parameters, jax.random.PRNGKey(5),
                                    dict(previous, post_trace=-previous['post_trace']),
                                    self.action[:, 0], create=False, modify=False)
    np.testing.assert_allclose(dleft['tick_synchrony'], dright['tick_synchrony'], atol=1e-8)
    np.testing.assert_allclose(left['deter'], -right['deter'], atol=1e-8)
    self.assertGreater(float(jnp.linalg.norm(left['deter'] - right['deter'])), .1)
    np.testing.assert_array_equal(left['deter'][..., :4], left['post_trace'][..., -1])
    np.testing.assert_allclose(left['deter'][..., 4],
                               jnp.tanh(left['post_trace'][..., 0, -1]), atol=1e-8)
    def objective(weights):
      (memory, _), _ = transition(weights, jax.random.PRNGKey(5), previous,
                                   self.action[:, 0], create=False, modify=False)
      return (memory['deter'] * jnp.arange(1., 6.)).sum()
    gradient = jax.grad(objective)(self.parameters)['nrsm/sync_projection/kernel']
    for part in (gradient[:4], gradient[4:]):
      self.assertTrue(np.isfinite(np.asarray(part)).all())
      self.assertGreater(float(jnp.linalg.norm(part)), 1e-7)

  def test_independent_write_budget_is_calibrated_across_ticks_and_time_scales(self):
    parameters = dict(self.parameters)
    candidate, gate, write_scale = .6, .25, .1
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.full(4, np.arctanh(candidate))
    parameters['nrsm/nlm/write_w'] = jnp.zeros_like(parameters['nrsm/nlm/write_w'])
    parameters['nrsm/nlm/write_b'] = jnp.full(4, math.log(gate / (1. - gate)))
    previous = self.apply(self.core.initial, 1)
    initial_activity = np.asarray([[.3, -.5, .7, -.9]], np.float64)
    previous = dict(previous, post_trace=jnp.broadcast_to(
        jnp.asarray(initial_activity[..., None], jnp.float32), (1, 4, 5)))
    raw = np.asarray(parameters['nrsm/time_constant_raw'], np.float64)
    time_constant = np.exp(np.log(.1) + np.log(1000.) / (1. + np.exp(-raw)))
    expected = np.exp(-1. / time_constant) * initial_activity + write_scale * gate * candidate
    for ticks in (1, 3, 7, 16):
      core = NRSM(neurons=4, memory=5, ticks=ticks, deter=5, stoch=2,
                  classes=3, units=7, nlm_hidden=3, sync_neurons=3, name='nrsm')
      (memory, diagnostics), _ = nj.pure(core._transition)(
          parameters, jax.random.PRNGKey(5), previous, self.action[:, 0],
          create=False, modify=False)
      # Exact real-arithmetic one-step formula gives every neuron the same
      # write budget, although the direct retained activity depends on T.
      np.testing.assert_allclose(memory['post_trace'][..., -1], expected,
                                 atol=2e-6, rtol=7e-6)
      np.testing.assert_allclose(diagnostics['tick_candidates'], candidate, atol=6e-8)
      np.testing.assert_allclose(diagnostics['tick_write_gates'], gate, atol=3e-8)
      rho = np.exp(-1. / (ticks * time_constant))
      stable = np.expm1(-1. / (ticks * time_constant)) / np.expm1(-1. / time_constant)
      step_activity = initial_activity.copy()
      for tick in range(ticks):
        step_activity = rho * step_activity + write_scale * stable * gate * candidate
        np.testing.assert_allclose(diagnostics['tick_activities'][:, tick], step_activity,
                                   atol=2e-6, rtol=7e-6)
    # A closed write gate must retain old activity regardless of the nonzero
    # candidate, rather than forcing an update toward that candidate.
    parameters['nrsm/nlm/write_b'] = jnp.full(4, -1000.)
    (memory, diagnostics), _ = nj.pure(self.core._transition)(
        parameters, jax.random.PRNGKey(5), previous, self.action[:, 0],
        create=False, modify=False)
    np.testing.assert_array_equal(diagnostics['tick_write_gates'], 0.)
    np.testing.assert_allclose(memory['post_trace'][..., -1],
                               initial_activity * np.exp(-1. / time_constant),
                               atol=2e-7, rtol=3e-6)

  def test_stable_write_calibration_survives_float32_retention_rounding_to_one(self):
    # At these configurable time scales rho rounds to one in float32. A
    # ratio made from 1-rho would be 0/0, whereas the expm1 ratio stays finite.
    core = NRSM(neurons=4, memory=5, ticks=3, deter=5, stoch=2,
                classes=3, units=7, nlm_hidden=3, sync_neurons=3,
                time_constant_min=1e9, time_constant_max=1e12, name='nrsm')
    parameters = dict(self.parameters)
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.full(4, np.arctanh(.6))
    previous = self.apply(self.core.initial, 1)
    (time_constant, retention), _ = nj.pure(core.memory_coefficients)(
        parameters, jax.random.PRNGKey(5), create=False, modify=False)
    np.testing.assert_array_equal(retention, np.ones(4))
    (memory, _), _ = nj.pure(core._transition)(
        parameters, jax.random.PRNGKey(5), previous, self.action[:, 0],
        create=False, modify=False)
    self.assertTrue(np.isfinite(np.asarray(memory['post_trace'])).all())
    np.testing.assert_allclose(memory['post_trace'][..., -1], .1 * .5 * .6,
                               atol=8e-9, rtol=1e-6)

  def test_write_scale_changes_write_without_changing_retention(self):
    parameters = dict(self.parameters)
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.full(4, np.arctanh(.6))
    previous = self.apply(self.core.initial, 1)
    outputs = []
    for scale in (.1, .3):
      core = NRSM(neurons=4, memory=5, ticks=3, deter=5, stoch=2,
                  classes=3, units=7, nlm_hidden=3, sync_neurons=3,
                  write_scale=scale, name='nrsm')
      (memory, _), _ = nj.pure(core._transition)(
          parameters, jax.random.PRNGKey(5), previous, self.action[:, 0],
          create=False, modify=False)
      outputs.append(memory['post_trace'][..., -1])
      np.testing.assert_allclose(outputs[-1], scale * .5 * .6, atol=2e-8)
    np.testing.assert_allclose(outputs[1], 3. * outputs[0], atol=3e-8)

  def test_slow_activity_can_accumulate_above_one_without_readout_saturation(self):
    parameters = dict(self.parameters)
    parameters['nrsm/nlm/w2'] = jnp.zeros_like(parameters['nrsm/nlm/w2'])
    parameters['nrsm/nlm/b2'] = jnp.full(4, np.arctanh(.9))
    parameters['nrsm/nlm/write_b'] = jnp.full(4, math.log(.9 / .1))
    initial = self.apply(self.core.initial, 1)
    transition = nj.pure(self.core._transition)
    def run():
      def advance(_, previous):
        (memory, _), _ = transition(parameters, jax.random.PRNGKey(5), previous,
                                     self.action[:, 0], create=False, modify=False)
        return dict(previous, **memory)
      return jax.lax.fori_loop(0, 30, advance, initial)
    memory = jax.jit(run)()
    time_constant, _ = self.apply(self.core.memory_coefficients)
    retention = np.exp(-1. / np.asarray(time_constant, np.float64))
    expected = .1 * .9 * .9 * (1. - retention ** 30) / (1. - retention)
    activity = memory['post_trace'][..., -1]
    np.testing.assert_allclose(activity[0], expected, atol=3e-6, rtol=1e-5)
    self.assertGreater(float(activity[0, -1]), 1.)
    np.testing.assert_array_equal(memory['deter'][..., :4], activity)
    self.assertTrue(np.all(np.abs(np.asarray(memory['deter'][..., 4:])) <= 1.))
    bound = .1 / (-np.expm1(-1. / 100.))
    self.assertLessEqual(float(jnp.abs(activity).max()), bound)

  def test_sync_subset_spans_fast_and_slow_neurons_with_fixed_unique_pairs(self):
    for neurons, selected in ((64, 16), (7, 4), (4, 3), (5, 5), (1, 1)):
      core = NRSM(neurons=neurons, sync_neurons=selected,
                  deter=neurons + 1, name=f'pairs_{neurons}_{selected}')
      expected = np.rint(np.linspace(0, neurons - 1, selected)).astype(np.int32)
      np.testing.assert_array_equal(core._sync_indices, expected)
      self.assertEqual(len(np.unique(core._sync_indices)), selected)
      self.assertEqual(core._sync_indices[0], 0)
      if selected > 1:
        self.assertEqual(core._sync_indices[-1], neurons - 1)
        self.assertTrue(any(i == 0 and j == neurons - 1 for i, j in core._pairs))
      expected_pairs = np.asarray([(i, j) for offset, i in enumerate(expected)
                                    for j in expected[offset:]], np.int32)
      np.testing.assert_array_equal(core._pairs, expected_pairs)
      self.assertEqual(len(core._pairs), selected * (selected + 1) // 2)
    np.testing.assert_array_equal(self.core._pairs,
                                  [[0, 0], [0, 2], [0, 3], [2, 2], [2, 3], [3, 3]])

  def test_transition_raw_activity_identity_with_multiple_leading_axes(self):
    initial = self.apply(self.core.initial, 2)
    previous = jax.tree_util.tree_map(lambda value: jnp.repeat(value[:, None], 3, 1), initial)
    previous = dict(previous, post_trace=jnp.arange(120, dtype=jnp.float32).reshape(2, 3, 4, 5) / 100)
    actions = jnp.ones((2, 3, 2), jnp.float32) * .2
    transition = nj.pure(self.core._transition)
    (memory, diagnostics), _ = jax.jit(lambda p, state, action: transition(
        p, jax.random.PRNGKey(8), state, action, create=False, modify=False))(
            self.parameters, previous, actions)
    self.assertEqual(memory['deter'].shape, (2, 3, 5))
    self.assertEqual(diagnostics['tick_write_gates'].shape, (2, 3, 3, 4))
    np.testing.assert_array_equal(memory['deter'][..., :4], memory['post_trace'][..., -1])
    np.testing.assert_array_equal(diagnostics['tick_contexts'][..., :4],
                                  diagnostics['tick_activities'])

  def test_observation_changes_memory_only_on_a_later_transition(self):
    initial = self.apply(self.core.initial, 1)
    first = jnp.zeros(1, jnp.bool_)
    post1, _ = self.apply(self.core.obs_step, initial, self.action[:, 0],
                         self.embed[:, 0], first)
    post2, _ = self.apply(self.core.obs_step, initial, self.action[:, 0],
                         self.embed[:, 0] + 10., first)
    for key in ('deter', 'pre_trace', 'post_trace'):
      np.testing.assert_array_equal(post1[key], post2[key])
    # Explicit different posterior z avoids depending on a lucky random draw.
    post1 = dict(post1, stoch=jax.nn.one_hot(jnp.asarray([[0, 1]]), 3))
    post2 = dict(post2, stoch=jax.nn.one_hot(jnp.asarray([[2, 0]]), 3))
    next1, _ = self.apply(self.core._transition, post1, self.action[:, 1])
    next2, _ = self.apply(self.core._transition, post2, self.action[:, 1])
    self.assertGreater(float(jnp.linalg.norm(next1['post_trace'] - next2['post_trace'])), 1e-7)

  def test_sampled_scan_matches_explicit_steps_with_the_same_rngs(self):
    # nj.scan splits the outer RNG once into one key per environment record.
    # Match those keys exactly to compare actual posterior samples and carries,
    # not merely deterministic modes or independent samples.
    seed = 37
    initial = self.apply(self.core.initial, 1)
    scan = self.apply(self.core.observe_with_ticks, self.embed, self.action,
                       self.first, initial, seed=seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), self.embed.shape[1] + 1)[1:]
    current, records = initial, []
    for index, key in enumerate(keys):
      record, _ = nj.pure(self.core._obs_step)(
          self.parameters, key, current, self.action[:, index], self.embed[:, index],
          self.first[:, index], True, create=False, modify=False)
      current = record[0]
      records.append(record)
    manual = jax.tree_util.tree_map(lambda *values: jnp.stack(values, 1), *records)
    for left, right in zip(jax.tree_util.tree_leaves(scan), jax.tree_util.tree_leaves(manual)):
      np.testing.assert_allclose(left, right, atol=3e-6, rtol=3e-5)







  def test_transition_and_logits_match_independent_numpy_formulas(self):
    parameters = {key: np.asarray(value) for key, value in self.parameters.items()}
    decay_key = next(key for key in parameters if key.endswith('/sync_decay_raw'))
    parameters[decay_key] = np.asarray([-6., -2., -.3, .5, 1., 5.], np.float32)
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
    fraction = 1. / (1. + np.exp(-parameter('time_constant_raw')))
    time_constant = np.exp(np.log(.1) + fraction * np.log(100. / .1))
    retention = np.exp(-1. / (3 * time_constant))
    # Independent fixed spread: N=4, D_s=3 selects [0,2,3], not the first 3.
    pairs = np.asarray([[0, 0], [0, 2], [0, 3], [2, 2], [2, 3], [3, 3]])
    ages = np.arange(4, -1, -1, dtype=np.float32)
    decay = 4. / (1. + np.exp(-parameters[decay_key]))
    weights = np.exp(-decay[:, None] * ages)
    calibration = np.expm1(-1. / (3 * time_constant)) / np.expm1(-1. / time_constant)
    # Nonconstant gate parameters exercise dependence on the shared NLM hidden.
    parameters['nrsm/nlm/write_w'] = np.asarray([
        [.2, -.4, .1], [.1, .3, -.2], [-.2, .5, .4], [.7, -.1, -.3]], np.float32)
    parameters['nrsm/nlm/write_b'] = np.asarray([-.3, .6, -.8, .2], np.float32)
    contexts, activities, synchronies, candidates, gates = [], [], [], [], []
    for tick in range(3):
      alpha = numpy_navigation_synapse(
          lambda name: parameter('synapse_hidden/' + name),
          np.concatenate((activity, u), -1))
      alpha = numpy_navigation_synapse(
          lambda name: parameter('synapse_output/' + name), alpha)
      history = np.concatenate((history[..., 1:], alpha[..., None]), -1)
      hidden = np.einsum('bdm,dmh->bdh', history, parameter('nlm/w1')) + parameter('nlm/b1')
      hidden = np.where(hidden > 0, hidden, np.expm1(hidden))
      candidate = np.tanh((hidden * parameter('nlm/w2')).sum(-1) + parameter('nlm/b2'))
      gate = 1. / (1. + np.exp(-((hidden * parameter('nlm/write_w')).sum(-1)
                                  + parameter('nlm/write_b'))))
      activity = retention * activity + .1 * calibration * gate * candidate
      post_trace = np.concatenate((post_trace[..., 1:], activity[..., None]), -1)
      products = post_trace[:, pairs[:, 0], :] * post_trace[:, pairs[:, 1], :]
      sync = (weights[None] * products).sum(-1) / np.sqrt(weights.sum(-1))
      projected = dense('sync_projection', np.concatenate((activity, sync), -1), 'tanh')
      contexts.append(np.concatenate((activity, projected), -1))
      activities.append(activity)
      synchronies.append(sync)
      candidates.append(candidate)
      gates.append(gate)
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
    np.testing.assert_allclose(diagnostics['tick_contexts'][:, 0], np.stack(contexts, 1), atol=3e-6)
    np.testing.assert_allclose(diagnostics['tick_activities'][:, 0], np.stack(activities, 1), atol=3e-6)
    np.testing.assert_allclose(diagnostics['tick_synchrony'][:, 0], np.stack(synchronies, 1), atol=3e-6)
    np.testing.assert_allclose(diagnostics['tick_candidates'][:, 0], np.stack(candidates, 1), atol=3e-6)
    np.testing.assert_allclose(diagnostics['tick_write_gates'][:, 0], np.stack(gates, 1), atol=3e-6)
    np.testing.assert_allclose(post['pre_trace'][:, 0], history, atol=3e-6)
    np.testing.assert_allclose(post['post_trace'][:, 0], post_trace, atol=3e-6)
    np.testing.assert_allclose(prior['logit'][:, 0], expected_prior, atol=3e-6)
    np.testing.assert_allclose(post['logit'][:, 0], expected_post, atol=3e-6)

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
    groups = ('synapse_hidden', 'synapse_output', 'nlm/w1', 'nlm/w2',
              'nlm/write_w', 'nlm/write_b', 'sync_decay_raw',
              'time_constant_raw',
              'sync_projection', 'prior_hidden', 'prior_stats',
              'posterior_hidden', 'posterior_stats')
    for group in groups:
      with self.subTest(group=group):
        selected = [value for name, value in derivatives.items() if group in name]
        self.assertTrue(selected, group)
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in selected))
        self.assertGreater(sum(float(jnp.linalg.norm(value)) for value in selected), 0., group)

  def test_gradients_reach_previous_histories_across_environment_steps(self):
    initial = self.apply(self.core.initial, 1)
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
    # This inherited Ninjax scope context does not restore itself when an
    # expected constructor exception is raised. Keep validation tests from
    # changing the namespace seen by unrelated numerical tests afterward.
    self.addCleanup(setattr, nj, 'SCOPE', nj.SCOPE)
    for kwargs in ({'classes': 1}, {'sync_neurons': 65}, {'ticks': 0},
                   {'sequence_unroll': True}, {'tick_mode': 'minimum'},
                   {'unimix': -1.0}, {'time_constant_min': 0.},
                   {'time_constant_min': float('nan')},
                   {'time_constant_max': float('inf')},
                   {'time_constant_min': 100.},
                   {'deter': 64}, {'deter': 63}, {'write_scale': 0.},
                   {'write_scale': -1.}, {'write_scale': float('nan')},
                   {'write_scale': float('inf')},
                   {'neurons': 0}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        NRSM(**kwargs, name='invalid')

  def test_constructor_allows_jaxagent_transfer_guard(self):
    # The production JAXAgent sets this guard while constructing WorldModel;
    # implicit jnp.asarray(host_pairs) fails even though ordinary tests pass.
    with jax.transfer_guard('disallow'):
      guarded = NRSM(neurons=4, sync_neurons=3, name='guarded_nrsm')
    np.testing.assert_array_equal(
        np.asarray(guarded._pairs), [[0, 0], [0, 2], [0, 3], [2, 2], [2, 3], [3, 3]])


if __name__ == '__main__':
  unittest.main()
