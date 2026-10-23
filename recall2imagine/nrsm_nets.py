"""CT-WM neuronal transitions using the R2I RSSM interface.

The deterministic memory comprises ``pre_trace, post_trace``. Neuronal activity
persists in the newest post-trace entry and follows a learned time scale per
neuron with an independent input-dependent write gate; there is no additional
SSM state. In v11, ``deter`` concatenates unmodified activity with a learned
synchronization readout for prediction and policy heads. ``stoch`` is
the stochastic environment state and ``logit`` its distribution cache. The
same neuronal transition serves filtering and imagination, without posterior
memory correction or an auxiliary reconstruction branch.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import ninjax as nj
from . import nets


def _uniform(shape, bound):
  return jax.random.uniform(nj.rng(), shape, jnp.float32, -bound, bound)


def _initial_time_constant_raw(shape):
  """Finite logits giving logarithmically spaced fast and slow neurons.

  Fractions stay strictly inside the configured log-time interval, avoiding
  infinite endpoint logits. A single-neuron core starts at its log midpoint.
  """
  fraction = (jnp.linspace(.01, .99, shape[0], dtype=jnp.float32)
              if shape[0] > 1 else jnp.full(shape, .5, jnp.float32))
  return jnp.log(fraction) - jnp.log1p(-fraction)


def categorical_kl(posterior_logits, prior_logits):
  """Unclipped KL(q || p), with gradients through both distributions."""
  logq = jax.nn.log_softmax(posterior_logits.astype(jnp.float32), -1)
  logp = jax.nn.log_softmax(prior_logits.astype(jnp.float32), -1)
  return (jnp.exp(logq) * (logq - logp)).sum((-2, -1))


def prior_certainty(prior_logits):
  logp = jax.nn.log_softmax(prior_logits.astype(jnp.float32), -1)
  variables, classes = prior_logits.shape[-2:]
  return 1.0 + (jnp.exp(logp) * logp).sum((-2, -1)) / (
      variables * math.log(classes))


def tick_statistics(prior_logits, posterior_logits, mode='adaptive',
                    alpha_error=1.0, alpha_cert=1.0):
  """Return per-tick diagnostics and a per-record consistency objective.

  This is an additional CT-WM regularizer. R2I's existing balanced/free-nats
  losses remain separate. In adaptive mode the two softmax weight derivatives
  remain active. The final tick is included in this additional regularizer.
  """
  if mode not in ('adaptive', 'uniform', 'off'):
    raise ValueError(f'Unknown tick mode: {mode}')
  kl = categorical_kl(posterior_logits, prior_logits)
  certainty = prior_certainty(prior_logits)
  error = jax.nn.softmax(-alpha_error * kl, -1)
  confidence = jax.nn.softmax(alpha_cert * certainty, -1)
  if mode == 'adaptive':
    weights = 0.5 * (error + confidence)
  elif mode == 'uniform':
    weights = jnp.ones_like(kl) / kl.shape[-1]
  else:
    weights = jnp.zeros_like(kl)
  return dict(tick_kl=kl, tick_certainty=certainty, tick_weights=weights,
              error_weights=error, certainty_weights=confidence,
              tick_loss=(weights * kl).sum(-1))


def window_synchronization(post_trace, decay_raw, pairs):
  """Read a CTM-style finite post-activation window with smooth learned decay.

  Traces have shape [..., neurons, memory], ordered oldest to newest. Each
  raw parameter is mapped to decay by 4 * sigmoid(raw), removing the hard
  clamp's finite interval with identically zero derivative. As with any
  sigmoid, extreme floating-point inputs can still saturate. The newest
  entry always has weight one, including zero-padded episode-start windows; the
  normalization therefore never needs an epsilon or a growing time counter.
  """
  post_trace = post_trace.astype(jnp.float32)
  products = post_trace[..., pairs[:, 0], :] * post_trace[..., pairs[:, 1], :]
  ages = jnp.arange(post_trace.shape[-1] - 1, -1, -1, dtype=jnp.float32)
  decay = 4.0 * jax.nn.sigmoid(decay_raw.astype(jnp.float32))
  weights = jnp.exp(-decay[..., None] * ages)
  return (products * weights).sum(-1) / jnp.sqrt(weights.sum(-1))


class PrivateNLM(nj.Module):
  """Private ELU hidden layers with separate candidate and write-gate outputs.

  Both outputs use only that neuron's history. The zero-initialized gate
  output starts at one half and remains trainable independently of retention.
  """

  def __init__(self, neurons, memory, hidden):
    self.neurons = neurons
    self.memory = memory
    self.hidden = hidden

  def __call__(self, history):
    # Explicit per-neuron Glorot bounds avoid including the neuron axis in
    # fan-in/fan-out, which an ordinary rank-3 initializer would do.
    first = math.sqrt(6.0 / (self.memory + self.hidden))
    second = math.sqrt(6.0 / (self.hidden + 1))
    w1 = self.get('w1', _uniform,
                  (self.neurons, self.memory, self.hidden), first)
    b1 = self.get('b1', jnp.zeros, (self.neurons, self.hidden), jnp.float32)
    w2 = self.get('w2', _uniform, (self.neurons, self.hidden), second)
    b2 = self.get('b2', jnp.zeros, (self.neurons,), jnp.float32)
    write_w = self.get('write_w', jnp.zeros,
                       (self.neurons, self.hidden), jnp.float32)
    write_b = self.get('write_b', jnp.zeros, (self.neurons,), jnp.float32)
    hidden = jax.nn.elu(jnp.einsum('...dm,dmh->...dh', history, w1) + b1)
    candidate = jnp.tanh((hidden * w2).sum(-1) + b2)
    write_gate = jax.nn.sigmoid((hidden * write_w).sum(-1) + write_b)
    return candidate, write_gate


class _NavigationSynapseBlock(nj.Module):
  """Original CTM navigation Linear -> GLU -> affine LayerNorm block.

  PyTorch Linear's Kaiming-uniform initialization with a=sqrt(5) is uniform
  on +/-1/sqrt(fan_in), including its bias. LayerNorm uses the population
  variance and epsilon 1e-5, unlike the shared Dreamer norm's epsilon 1e-3.
  All operations retain arbitrary leading batch/time axes and float32.
  """

  def __init__(self, neurons):
    self._neurons = neurons

  def __call__(self, inputs):
    inputs = inputs.astype(jnp.float32)
    bound = 1.0 / math.sqrt(inputs.shape[-1])
    kernel = self.get('kernel', _uniform,
                      (inputs.shape[-1], 2 * self._neurons), bound)
    bias = self.get('bias', _uniform, (2 * self._neurons,), bound)
    value, gate = jnp.split(inputs @ kernel + bias, 2, axis=-1)
    activation = value * jax.nn.sigmoid(gate)
    centered = activation - activation.mean(axis=-1, keepdims=True)
    variance = jnp.square(centered).mean(axis=-1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(variance + 1e-5)
    scale = self.get('norm_scale', jnp.ones, (self._neurons,), jnp.float32)
    shift = self.get('norm_bias', jnp.zeros, (self._neurons,), jnp.float32)
    return normalized * scale + shift


class NRSM(nets.RSSM):
  """Multi-tick synchronized recurrent transition compatible with R2I.

  ``units`` is the prior/posterior width. Observation
  encoding stays in R2I's existing modality-specific encoder. The synapse
  uses two original CTM navigation GLU/LayerNorm blocks, each producing
  ``neurons`` coordinates; private NLM and prior/posterior keep ELU/tanh.
  The existing neuronal activity has a learned retention coefficient per neuron
  and a separate NLM write gate. Time constants and the write budget are
  calibrated in environment steps across K internal ticks. The deterministic
  readout preserves raw activity alongside projected activity/synchronization.
  R2I's categorical unimix, distribution API, dyn_loss, and rep_loss are reused.
  """

  state_keys = ('deter', 'stoch', 'logit', 'pre_trace', 'post_trace')

  def __init__(self, neurons=64, memory=8, ticks=4, deter=128, stoch=8,
               classes=8, units=128, nlm_hidden=8, sync_neurons=16,
               unimix=0.01, action_clip=1.0, tick_mode='adaptive',
               alpha_error=1.0, alpha_cert=1.0, sequence_unroll=False,
               time_constant_min=.1, time_constant_max=100.0, write_scale=.1):
    dimensions = dict(neurons=neurons, memory=memory, ticks=ticks, deter=deter,
                      stoch=stoch, classes=classes, units=units,
                      nlm_hidden=nlm_hidden, sync_neurons=sync_neurons)
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
      raise ValueError('NRSM dimensions must be positive integers')
    if classes < 2 or sync_neurons > neurons:
      raise ValueError('Require classes >= 2 and sync_neurons <= neurons')
    if deter <= neurons:
      raise ValueError('Require deter > neurons for raw activity and projected readout')
    if not (math.isfinite(time_constant_min) and math.isfinite(time_constant_max)
            and 0 < time_constant_min < time_constant_max):
      raise ValueError('Require finite 0 < time_constant_min < time_constant_max')
    if not math.isfinite(write_scale) or write_scale <= 0:
      raise ValueError('write_scale must be finite and positive')
    if not 0 <= unimix <= 1:
      raise ValueError('unimix must be in [0, 1]')
    if tick_mode not in ('adaptive', 'uniform', 'off'):
      raise ValueError('tick_mode must be adaptive, uniform, or off')
    if not all(math.isfinite(x) and x > 0 for x in (alpha_error, alpha_cert)):
      raise ValueError('Tick temperatures must be finite and positive')
    if not math.isfinite(action_clip) or action_clip < 0:
      raise ValueError('action_clip must be finite and nonnegative')
    if sequence_unroll is not False:
      raise ValueError('NRSM uses a compiled environment-time scan; sequence_unroll must be False')
    self._neurons = neurons
    self._memory = memory
    self._ticks = ticks
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._units = units
    self._nlm_hidden = nlm_hidden
    self._time_constant_min = time_constant_min
    self._time_constant_max = time_constant_max
    self._write_scale = write_scale
    self._unimix = unimix
    self._action_clip = action_clip
    self._tick_mode = tick_mode
    self._alpha_error = alpha_error
    self._alpha_cert = alpha_cert
    self._unroll = False
    self._nonrecurrent_enc = False
    # Keep architecture indices on the host. During tracing they become HLO
    # constants; a closed-over GPU array would require an implicit readback
    # while compiling/caching policy under JAXAgent's transfer guard.
    # Fixed coverage across the initially ordered time scales includes both
    # endpoints. No sorting or index changes occur as time scales are learned.
    self._sync_indices = (np.asarray([0], np.int32) if sync_neurons == 1 else
                          np.rint(np.linspace(
                              0, neurons - 1, sync_neurons)).astype(np.int32))
    self._pairs = np.asarray([
        (int(i), int(j)) for offset, i in enumerate(self._sync_indices)
        for j in self._sync_indices[offset:]
    ], np.int32)

  def _linear(self, name, units, activation, value):
    return self.get(name, nets.Linear, units, act=activation, norm='none',
                    winit='uniform', fan='avg')(value)

  def initial(self, batch_size):
    return dict(deter=jnp.zeros((batch_size, self._deter), jnp.float32),
                stoch=jnp.zeros((batch_size, self._stoch, self._classes), jnp.float32),
                logit=jnp.zeros((batch_size, self._stoch, self._classes), jnp.float32),
                pre_trace=jnp.zeros(
                    (batch_size, self._neurons, self._memory), jnp.float32),
                post_trace=jnp.zeros(
                    (batch_size, self._neurons, self._memory), jnp.float32))

  def memory_coefficients(self):
    """Return per-neuron (T, rho), with T measured in environment steps.

    The direct old-activity path after K ticks is rho**K = exp(-1/T).
    This calibrates only that path; nonlinear feedback can still change the
    full recurrence Jacobian. Positive bounded time scales do not guarantee
    task memory or dynamical stability.
    """
    raw = self.get('time_constant_raw', _initial_time_constant_raw,
                   (self._neurons,))
    log_time = math.log(self._time_constant_min) + jax.nn.sigmoid(raw) * (
        math.log(self._time_constant_max) - math.log(self._time_constant_min))
    time_constant = jnp.exp(log_time)
    retention = jnp.exp(-1.0 / (self._ticks * time_constant))
    return time_constant, retention

  def _action(self, previous_action, previous_stoch):
    action = previous_action.astype(jnp.float32)
    # Match R2I's flattening convention for factored/array actions.
    if len(action.shape) > len(previous_stoch.shape) - 1:
      action = action.reshape(action.shape[:len(previous_stoch.shape) - 2] + (-1,))
    if self._action_clip:
      action *= jax.lax.stop_gradient(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(action)))
    return action

  def prior_context(self, previous, previous_action):
    """Return the deterministic readout and diagnostics, without an observation."""
    memory, diagnostics = self._transition(previous, previous_action)
    return memory['deter'], diagnostics

  def _transition(self, previous, previous_action):
    """Advance persistent neuronal histories using only previous state/action."""
    z = previous['stoch'].astype(jnp.float32)
    action = self._action(previous_action, z)
    flat_z = z.reshape(z.shape[:-2] + (-1,))
    u = jnp.concatenate((previous['deter'].astype(jnp.float32), flat_z, action), -1)
    history = previous['pre_trace'].astype(jnp.float32)
    post_trace = previous['post_trace'].astype(jnp.float32)
    activity = post_trace[..., -1]
    time_constant, retention = self.memory_coefficients()
    # expm1 avoids cancellation for long time scales. For constant candidate
    # and gate, sum_{j=0}^{K-1} rho**j * calibration = 1 exactly in real
    # arithmetic, decoupling one environment step's write budget from T.
    write_calibration = jnp.expm1(-1.0 / (self._ticks * time_constant)) / (
        jnp.expm1(-1.0 / time_constant))
    decay_raw = self.get('sync_decay_raw', jnp.full, (len(self._pairs),),
                         math.log(.01 / (4.0 - .01)), jnp.float32)
    contexts, activities, synchronies, candidates, write_gates = [], [], [], [], []
    # K is a small static thought budget. Environment time uses nj.scan below.
    for _ in range(self._ticks):
      inputs = jnp.concatenate((activity, u), -1)
      alpha = self.get('synapse_hidden', _NavigationSynapseBlock,
                       self._neurons)(inputs)
      alpha = self.get('synapse_output', _NavigationSynapseBlock,
                       self._neurons)(alpha)
      history = jnp.concatenate((history[..., 1:], alpha[..., None]), -1)
      candidate, write_gate = self.get(
          'nlm', PrivateNLM, self._neurons, self._memory, self._nlm_hidden)(history)
      # Persist the unnormalized additive memory. This is not a convex mixture:
      # activity can exceed [-1, 1]. Bounded time scales and bounded writes give
      # a forward amplitude bound, not stability of nonlinear feedback/gradients.
      activity = (retention * activity + self._write_scale * write_calibration
                  * write_gate * candidate)
      post_trace = jnp.concatenate((post_trace[..., 1:], activity[..., None]), -1)
      sync = window_synchronization(post_trace, decay_raw, self._pairs)
      readout = jnp.concatenate((activity, sync), -1)
      projected = self._linear(
          'sync_projection', self._deter - self._neurons, 'tanh', readout)
      context = jnp.concatenate((activity, projected), -1)
      contexts.append(context)
      activities.append(activity)
      synchronies.append(sync)
      candidates.append(candidate)
      write_gates.append(write_gate)
    memory = dict(deter=context, pre_trace=history, post_trace=post_trace)
    diagnostics = dict(tick_contexts=jnp.stack(contexts, -2),
                       tick_activities=jnp.stack(activities, -2),
                       tick_synchrony=jnp.stack(synchronies, -2),
                       tick_candidates=jnp.stack(candidates, -2),
                       tick_write_gates=jnp.stack(write_gates, -2))
    return memory, diagnostics

  def _prior_stats(self, context):
    hidden = self._linear('prior_hidden', self._units, 'elu', context)
    return self._stats('prior_stats', hidden)

  def _posterior_stats(self, context, embed):
    inputs = jnp.concatenate((context, embed.astype(jnp.float32)), -1)
    hidden = self._linear('posterior_hidden', self._units, 'elu', inputs)
    return self._stats('posterior_stats', hidden)

  def get_stoch(self, deter):
    return self.get_dist(self._prior_stats(deter)).mode()

  def img_step(self, previous, previous_action):
    memory, _ = self._transition(previous, previous_action)
    stats = self._prior_stats(memory['deter'])
    return dict(**memory, stoch=self.get_dist(stats).sample(seed=nj.rng()), **stats)

  def _obs_step(self, previous, previous_action, embed, is_first, with_ticks):
    first = is_first.astype(jnp.bool_)
    initial = self.initial(first.shape[0])
    # New episodes zero both neuronal histories. where() also cleanly cuts
    # gradients from the previous episode for reset rows.
    previous = jax.tree_util.tree_map(
        lambda value, start: jnp.where(
            first.reshape(first.shape + (1,) * (value.ndim - first.ndim)),
            start, value), previous, initial)
    previous_action = self._mask(previous_action, 1.0 - first.astype(jnp.float32))
    memory, diagnostics = self._transition(previous, previous_action)
    context = memory['deter']
    if with_ticks:
      contexts = diagnostics['tick_contexts']
      embeddings = jnp.broadcast_to(embed[..., None, :],
                                    contexts.shape[:-1] + (embed.shape[-1],))
      prior_stats = self._prior_stats(contexts)
      posterior_stats = self._posterior_stats(contexts, embeddings)
      prior_final = {key: value[..., -1, :, :] for key, value in prior_stats.items()}
      post_final = {key: value[..., -1, :, :] for key, value in posterior_stats.items()}
      diagnostics.update(tick_statistics(
          prior_stats['logit'], posterior_stats['logit'], self._tick_mode,
          self._alpha_error, self._alpha_cert))
      diagnostics.update(prior_logits=prior_stats['logit'],
                         posterior_logits=posterior_stats['logit'])
    else:
      prior_final = self._prior_stats(context)
      post_final = self._posterior_stats(context, embed)
    # A real observation draws only the final posterior. The unused prior z is
    # a mode diagnostic; imagination samples the actual prior in img_step().
    post = dict(**memory,
                stoch=self.get_dist(post_final).sample(seed=nj.rng()), **post_final)
    prior = dict(**memory, stoch=self.get_dist(prior_final).mode(), **prior_final)
    return post, prior, diagnostics

  def obs_step(self, prev_state, prev_action, embed, is_first):
    post, prior, _ = self._obs_step(prev_state, prev_action, embed, is_first, False)
    return post, prior

  def observe_with_ticks(self, embed, action, is_first, state=None):
    """Filter [B,T,...] with one compiled scan, returning separate diagnostics."""
    if len(embed.shape) != 3 or len(is_first.shape) != 2:
      raise ValueError('Expected embed [B,T,E] and is_first [B,T]')
    if embed.shape[:2] != action.shape[:2] or embed.shape[:2] != is_first.shape:
      raise ValueError('Observation, action, and reset batch/time axes must agree')
    if embed.shape[1] < 1:
      raise ValueError('Cannot observe an empty sequence')
    state = self.initial(embed.shape[0]) if state is None else state
    swap = lambda value: jnp.swapaxes(value, 0, 1)
    def step(previous, inputs):
      current_embed, current_action, first = inputs
      post, prior, diagnostics = self._obs_step(
          previous, current_action, current_embed, first, True)
      return post, (post, prior, diagnostics)
    _, outputs = nj.scan(step, state, (swap(embed), swap(action), swap(is_first)), unroll=1)
    return jax.tree_util.tree_map(swap, outputs)

  def observe(self, embed, action, is_first, state=None):
    post, prior, _ = self.observe_with_ticks(embed, action, is_first, state)
    return post, prior
