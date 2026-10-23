"""Action-conditioned first-layer readout of CTM activity.

Only the first affine map differs from nets.MLP. Remaining normalization,
activation, hidden layers and output distributions use the original modules.
"""

import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import nets
from . import ninjax as nj


class ActionConditionedLinear(nj.Module):

  def __init__(self, units, neurons, action_dim, discrete, act='none',
               norm='none', bias=True, outscale=1.0, outnorm=False,
               winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._neurons = int(neurons)
    self._action_dim = int(action_dim)
    self._discrete = bool(discrete)
    if self._neurons < 1 or self._action_dim < 1:
      raise ValueError('Reward activity and action widths must be positive.')
    self._act = nets.get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._initializer = nets.Initializer(winit, outscale, fan=fan)

  def __call__(self, activity, shared, action):
    if activity.shape[-1] != self._neurons:
      raise ValueError('Reward activity width does not match CTM neurons.')
    if action.shape != activity.shape[:-1] + (self._action_dim,):
      raise ValueError('Incoming reward action must match feature leading axes and width.')
    if shared.shape[:-1] != activity.shape[:-1]:
      raise ValueError('Reward shared and activity leading axes must agree.')
    width = int(np.prod(self._units))
    # Initialize each bank with the original MLP fan (activity + shared),
    # rather than multiplying fan-in by the number of selectable actions.
    full_shape = (activity.shape[-1] + shared.shape[-1], width)
    init_shared = lambda shape: self._initializer(full_shape)[self._neurons:]
    banks = self._action_dim if self._discrete else self._action_dim + 1
    def init_action(shape):
      return jnp.stack([self._initializer(full_shape)[:self._neurons]
                        for _ in range(banks)], axis=0)
    shared_kernel = self.get('shared_kernel', init_shared,
                             (shared.shape[-1], width))
    action_kernel = self.get('action_kernel', init_action,
                             (banks, self._neurons, width))
    shared_kernel, action_kernel = jaxutils.cast_to_compute(
        (shared_kernel, action_kernel))
    weights = action if self._discrete else jnp.concatenate(
        [jnp.ones(action.shape[:-1] + (1,), action.dtype), action], -1)
    # Weighted sum, not argmax/gather: straight-through discrete samples and
    # continuous actions retain their existing differentiable path.
    x = shared @ shared_kernel + jnp.einsum(
        '...n,knu,...k->...u', activity, action_kernel, weights)
    if self._bias:
      bias = self.get('bias', jnp.zeros, width, np.float32)
      x += jaxutils.cast_to_compute(bias)
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    return self._act(self.get('norm', nets.Norm, self._norm)(x))


class ActionConditionedMLP(nets.MLP):

  def __init__(self, shape, neurons, action_dim, discrete, **kwargs):
    super().__init__(shape, **kwargs)
    if self._layers < 1:
      raise ValueError('Action-conditioned reward requires at least one hidden layer.')
    if self._inputs._keys != ('deter', 'stoch'):
      raise ValueError('Action-conditioned reward inputs must be deter, stoch in that order.')
    self._neurons = int(neurons)
    self._action_dim = int(action_dim)
    self._discrete = bool(discrete)

  def __call__(self, inputs, incoming_action):
    if inputs['deter'].shape[-1] <= self._neurons:
      raise ValueError('Reward deter must contain activity followed by projected features.')
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)
    action = jaxutils.cast_to_compute(incoming_action)
    if action.shape != feat.shape[:-1] + (self._action_dim,):
      raise ValueError('Incoming reward action must match feature leading axes and width.')
    x = x.reshape((-1, x.shape[-1]))
    action = action.reshape((-1, self._action_dim))
    x = self.get('h0', ActionConditionedLinear, self._units, self._neurons,
                 self._action_dim, self._discrete, **self._dense)(
                     x[:, :self._neurons], x[:, self._neurons:], action)
    for i in range(1, self._layers):
      x = self.get(f'h{i}', nets.Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
    if self._shape is None:
      return x
    if isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    if isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    raise ValueError(self._shape)
