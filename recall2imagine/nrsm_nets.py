"""CT-WM neuronal transitions using the R2I RSSM interface.

The deterministic memory is the pair ``pre_trace, post_trace``. Both finite
neuronal histories persist across environment steps; ``deter`` caches their
synchronization readout for the R2I prediction and policy heads. ``stoch`` is
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


def window_synchronization(post_trace, beta, pairs):
  """Read the finite post-activation window, as in the official CTM RL core.

  Traces have shape [..., neurons, memory], ordered oldest to newest. Each
  selected pair has a learned nonnegative decay in [0, 4]. The newest entry
  always has weight one, including zero-padded episode-start windows; the
  normalization therefore never needs an epsilon or a growing time counter.
  """
  post_trace = post_trace.astype(jnp.float32)
  products = post_trace[..., pairs[:, 0], :] * post_trace[..., pairs[:, 1], :]
  ages = jnp.arange(post_trace.shape[-1] - 1, -1, -1, dtype=jnp.float32)
  decay = jnp.clip(beta.astype(jnp.float32), 0.0, 4.0)
  weights = jnp.exp(-decay[..., None] * ages)
  return (products * weights).sum(-1) / jnp.sqrt(weights.sum(-1))


class PrivateNLM(nj.Module):
  """Independent M -> hidden -> 1 ELU/tanh networks for each neuron."""

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
    hidden = jax.nn.elu(jnp.einsum('...dm,dmh->...dh', history, w1) + b1)
    return jnp.tanh((hidden * w2).sum(-1) + b2)


class NRSM(nets.RSSM):
  """Multi-tick synchronized recurrent transition compatible with R2I.

  ``units`` is the synapse and prior/posterior hidden width. Observation
  encoding stays in R2I's existing modality-specific encoder. All core dense
  layers use CT-WM's ELU/tanh activations and Glorot-uniform initialization.
  R2I's categorical unimix, distribution API, dyn_loss, and rep_loss are reused.
  """

  def __init__(self, neurons=64, memory=8, ticks=4, deter=128, stoch=8,
               classes=8, units=128, nlm_hidden=8, sync_neurons=16,
               unimix=0.01, action_clip=1.0, tick_mode='adaptive',
               alpha_error=1.0, alpha_cert=1.0, sequence_unroll=False):
    dimensions = dict(neurons=neurons, memory=memory, ticks=ticks, deter=deter,
                      stoch=stoch, classes=classes, units=units,
                      nlm_hidden=nlm_hidden, sync_neurons=sync_neurons)
    if any(type(value) is not int or value < 1 for value in dimensions.values()):
      raise ValueError('NRSM dimensions must be positive integers')
    if classes < 2 or sync_neurons > neurons:
      raise ValueError('Require classes >= 2 and sync_neurons <= neurons')
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
    self._pairs = np.asarray([
        (i, j) for i in range(sync_neurons) for j in range(i, sync_neurons)
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
    beta = self.get('sync_beta', jnp.zeros, (len(self._pairs),), jnp.float32)
    contexts, activities, synchronies = [], [], []
    # K is a small static thought budget. Environment time uses nj.scan below.
    for _ in range(self._ticks):
      inputs = jnp.concatenate((activity, u), -1)
      alpha = self._linear('synapse_hidden', self._units, 'elu', inputs)
      alpha = self._linear('synapse_output', self._neurons, 'none', alpha)
      history = jnp.concatenate((history[..., 1:], alpha[..., None]), -1)
      activity = self.get('nlm', PrivateNLM, self._neurons, self._memory,
                           self._nlm_hidden)(history)
      post_trace = jnp.concatenate((post_trace[..., 1:], activity[..., None]), -1)
      sync = window_synchronization(post_trace, beta, self._pairs)
      context = self._linear('sync_projection', self._deter, 'tanh', sync)
      contexts.append(context)
      activities.append(activity)
      synchronies.append(sync)
    memory = dict(deter=context, pre_trace=history, post_trace=post_trace)
    diagnostics = dict(tick_contexts=jnp.stack(contexts, -2),
                       tick_activities=jnp.stack(activities, -2),
                       tick_synchrony=jnp.stack(synchronies, -2))
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
    first = is_first.astype(jnp.float32)
    # Zero sentinels match CT-WM. Ignore distribution cache in the transition.
    previous = jax.tree_util.tree_map(
        lambda value: self._mask(value, 1.0 - first), previous)
    previous_action = self._mask(previous_action, 1.0 - first)
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
