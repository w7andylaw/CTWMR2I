import os
import copy

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self._architecture = {
        'ssm_type': config.ssm_type,
        'nrsm': dict(config.nrsm) if config.ssm_type == 'nrsm' else None,
    }
    self.config = config.jax
    self.batch_size = config.batch_size
    self.batch_length = config.batch_length
    self.data_loaders = config.data_loaders
    self.num_buffers = config.num_buffers
    self._setup()
    self.step = step
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    if config.ssm_type == 'nrsm':
      # Record the actual readouts after Agent resolves task/CLI overrides.
      # CTM owns recurrent memory, with independent writing and raw activity
      # access inside deter. Equal-shaped parameters do not imply compatible
      # dynamics, synchronization indices, or readout semantics.
      resolved = self.agent.config
      self._architecture['nrsm_state_schema'] = 'ctm-action-conditioned-reward-v12'
      self._architecture['nrsm_memory_contract'] = {
          'kind': 'persistent_ctm_activity_with_independent_write_gate',
          'state_keys': list(self.agent.wm.rssm.state_keys),
          'activity_storage': 'post_trace[..., -1]',
          'trace_order': 'oldest_to_newest_thought_ticks',
          'episode_reset': 'zero_all_ctm_state_and_incoming_action',
          'cross_step_activity': 'nu[t,0]=nu[t-1,K]',
          'update': 'rho*previous_activity+write_scale*kappa*write_gate*nlm_candidate',
          'rho': 'exp(-1/(ticks*time_constant))',
          'kappa': 'expm1(-1/(ticks*time_constant))/expm1(-1/time_constant)',
          'write_scale': resolved.nrsm.write_scale,
          'write_gate': {
              'input': 'private_nlm_elu_hidden',
              'output': 'sigmoid',
              'parameter_scope': 'private_per_neuron',
              'weights_initialization': 'zeros',
              'bias_initialization': 'zeros',
              'initial_value': 0.5,
          },
          'candidate_activation': 'tanh',
          'constant_gate_and_candidate_environment_step_write': 'write_scale*write_gate*nlm_candidate',
          'time_constant_parameterization': 'bounded_log_sigmoid_per_neuron',
          'time_constant_unit': 'environment_steps',
          'time_constant_min': resolved.nrsm.time_constant_min,
          'time_constant_max': resolved.nrsm.time_constant_max,
          'time_constant_initialization': 'log_spaced_1pct_to_99pct_bounds',
          'activity_layer_norm': False,
          'extra_ssm_state': False,
          'posterior_memory_correction': False,
      }
      self._architecture['nrsm_synapse_contract'] = {
          'kind': 'ctm_navigation_glu_layernorm',
          'blocks': 2,
          'linear_width': 2 * resolved.nrsm.neurons,
          'post_glu_width': resolved.nrsm.neurons,
          'order': ['linear', 'glu', 'layer_norm'],
          'layer_norm': {
              'axis': -1, 'epsilon': 1e-5,
              'elementwise_affine': True, 'variance': 'biased',
          },
          'linear_init': 'fan_in_uniform',
          'dropout': 0.0,
      }
      self._architecture['nrsm_readout_contract'] = {
          'prediction_inputs': {
              'decoder': list(resolved.decoder.inputs),
              'reward': list(resolved.reward_head.inputs),
              'cont': list(resolved.cont_head.inputs),
          },
          'actor_inputs': list(resolved.actor.inputs),
          'critic_inputs': list(resolved.critic.inputs),
          'deter_concatenation': {
              'order': ['current_activity', 'projected_activity_and_synchrony'],
              'raw_activity_width': resolved.nrsm.neurons,
              'raw_activity_transform': 'identity',
              'total_width': resolved.nrsm.deter,
          },
          'synchronization': {
              'index_selection': 'rint(linspace(0,neurons-1,sync_neurons))',
              'indices': self.agent.wm.rssm._sync_indices.tolist(),
              'pairs': self.agent.wm.rssm._pairs.tolist(),
              'pair_order': 'upper_triangle_with_diagonal_in_selected_index_order',
              'indices_remain_fixed_during_learning': True,
          },
          'deter_projection': {
              'input_order': ['current_activity', 'window_synchrony'],
              'input_width': resolved.nrsm.neurons + (
                  resolved.nrsm.sync_neurons * (resolved.nrsm.sync_neurons + 1) // 2),
              'output_width': resolved.nrsm.deter - resolved.nrsm.neurons,
              'activation': 'tanh',
          },
          'direct_extra_memory_bypass': False,
          'direct_action_concatenation': False,
          'reward_first_layer': {
              'kind': 'action_conditioned_activity_readout',
              'activity': 'deter[..., :neurons]',
              'shared': 'concat(deter[..., neurons:], flatten(stoch))',
              'discrete_weights': 'incoming_action_onehot',
              'continuous_weights': 'concat(ones,incoming_action_flat)',
              'action_width': int(np.prod(act_space['action'].shape)),
              'action_discrete': bool(act_space['action'].discrete),
              'action_processing': 'rssm._action_flatten_and_clip',
              'affine': 'shared@U+einsum(activity,W,action_weights)+bias_if_no_norm',
              'norm_activation_and_remaining_layers': 'unchanged_reward_mlp',
              'initialization_fan': 'original_activity_plus_shared_width',
              'distribution_and_loss': 'unchanged',
              'training_action': 'concat(carry_action,replay_action[:,:-1]); zero_on_is_first',
              'imagination_action': 'concat(zero,trajectory.action[:-1])',
              'ephemeral_key': 'reward_action',
              'ephemeral_key_in_recurrent_state': False,
          },
      }
    self.rng = np.random.default_rng(config.seed)

    available = jax.devices(self.config.platform)
    self.policy_devices = [available[i] for i in self.config.policy_devices]
    self.train_devices = [available[i] for i in self.config.train_devices]
    self.single_device = (self.policy_devices == self.train_devices) and (
        len(self.policy_devices) == 1)
    print(f'JAX devices ({jax.local_device_count()}):', available)
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    self._once = True
    self._updates = embodied.Counter()
    self._should_metrics = embodied.when.Every(self.config.metrics_every)
    self._transform()
    self.varibs = self._init_varibs(obs_space, act_space)
    self.sync()

  def policy(self, obs, state=None, mode='train'):
    obs = obs.copy()
    obs = self._convert_inps(obs, self.policy_devices)
    rng = self._next_rngs(self.policy_devices)
    varibs = self.varibs if self.single_device else self.policy_varibs
    if state is None:
      state, _ = self._init_policy(varibs, rng, obs['is_first'])
    else:
      state = tree_map(
          np.asarray, state, is_leaf=lambda x: isinstance(x, list))
      state = self._convert_inps(state, self.policy_devices)
    (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
    outs = self._convert_outs(outs, self.policy_devices)
    # TODO: Consider keeping policy states in accelerator memory.
    state = self._convert_outs(state, self.policy_devices)
    return outs, state

  def train(self, data, state=None):
    rng = self._next_rngs(self.train_devices)
    if state is None:
      state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    outs = self._convert_outs(outs, self.train_devices)
    self._updates.increment()
    if self._should_metrics(self._updates):
      mets = self._convert_mets(mets, self.train_devices)
    else:
      mets = {}
    if self._once:
      self._once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        if count is not None:
          mets[f'params_{name}'] = float(count)
    return outs, state, mets

  def report(self, data):
    rng = self._next_rngs(self.train_devices)
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets, self.train_devices)
    return mets

  def dataset(self, source, shared_memory=True):
    """
    Args:
      sourse: the data source. replay object or generator
      shared_memory: whether to use shared memory based batcher or not.
        shared memory version eats less RAM when sequence length is long.
        be careful - SM is only supported with the disk-based replay buffer.
    """
    if shared_memory:
      batcher = embodied.BatcherSM(
          replay=source,
          workers=self.data_loaders,
          batch_size=self.batch_size,
          batch_sequence_len=self.batch_length,
          postprocess=lambda x: self._convert_inps(x, self.train_devices),
          prefetch_source=4, prefetch_batch=self.num_buffers)
    else:
      batcher = embodied.Batcher(
        sources=[source] * self.batch_size,
        workers=self.data_loaders,
        postprocess=lambda x: self._convert_inps(x, self.train_devices),
        prefetch_source=4, prefetch_batch=1)
    return batcher()

  def save(self):
    if len(self.train_devices) > 1:
      varibs = tree_map(lambda x: x[0], self.varibs)
    else:
      varibs = self.varibs
    varibs = jax.device_get(varibs)
    data = tree_map(np.asarray, varibs)
    return {
        'format': 'ctwm-r2i-v1',
        'architecture': copy.deepcopy(self._architecture),
        'varibs': data,
        'rng_state': copy.deepcopy(self.rng.bit_generator.state),
        'updates': int(self._updates),
    }

  def load(self, state):
    # Also accept upstream parameter-only checkpoints for the upstream models.
    restored_rng = None
    restored_updates = None
    if self._architecture['ssm_type'] == 'nrsm' and state.get('format') != 'ctwm-r2i-v1':
      raise ValueError('NRSM checkpoints require versioned architecture metadata.')
    if state.get('format') == 'ctwm-r2i-v1':
      if state['architecture'] != self._architecture:
        raise ValueError('Checkpoint architecture does not match this run.')
      restored_rng = copy.deepcopy(self.rng)
      restored_rng.bit_generator.state = copy.deepcopy(state['rng_state'])
      restored_updates = int(state['updates'])
      if restored_updates < 0:
        raise ValueError('Checkpoint update count must be nonnegative.')
      state = state['varibs']
    template = self.varibs if len(self.train_devices) == 1 else tree_map(
        lambda x: x[0], self.varibs)
    expected = tree_map(lambda x: (x.shape, str(x.dtype)), template)
    actual = tree_map(lambda x: (x.shape, str(x.dtype)), state)
    if expected != actual:
      raise ValueError('Checkpoint parameter shapes or dtypes do not match.')
    if len(self.train_devices) == 1:
      restored_varibs = jax.device_put(state, self.train_devices[0])
    else:
      restored_varibs = jax.device_put_replicated(state, self.train_devices)
    self.varibs = restored_varibs
    if restored_rng is not None:
      self.rng = restored_rng
      self._updates.load(restored_updates)
    self.sync()

  def sync(self):
    if self.single_device:
      return
    if len(self.train_devices) == 1:
      varibs = self.varibs
    else:
      varibs = tree_map(lambda x: x[0].device_buffer, self.varibs)
    if len(self.policy_devices) == 1:
      self.policy_varibs = jax.device_put(varibs, self.policy_devices[0])
    else:
      self.policy_varibs = jax.device_put_replicated(
          varibs, self.policy_devices)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except ImportError:
      pass  # TensorFlow is optional; JAX executes the model.
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')
    xla_flags = []
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    jax.config.update('jax_transfer_guard', 'disallow')
    if self.config.platform == 'cpu' and self.config.debug:
      jax.config.update('jax_disable_most_optimizations', self.config.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

  def _transform(self):
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    if len(self.train_devices) == 1:
      kw = dict(device=self.train_devices[0])
      self._init_train = nj.jit(self._init_train, **kw)
      self._train = nj.jit(self._train, **kw)
      self._report = nj.jit(self._report, **kw)
    else:
      kw = dict(devices=self.train_devices)
      self._init_train = nj.pmap(self._init_train, 'i', **kw)
      self._train = nj.pmap(self._train, 'i', **kw)
      self._report = nj.pmap(self._report, 'i', **kw)
    if len(self.policy_devices) == 1:
      kw = dict(device=self.policy_devices[0])
      self._init_policy = nj.jit(self._init_policy, **kw)
      self._policy = nj.jit(self._policy, static=['mode'], **kw)
    else:
      kw = dict(devices=self.policy_devices)
      self._init_policy = nj.pmap(self._init_policy, 'i', **kw)
      self._policy = nj.pmap(self._policy, 'i', static=['mode'], **kw)

  def _convert_inps(self, value, devices):
    if len(devices) == 1:
      value = jax.device_put(value, devices[0])
    else:
      check = tree_map(lambda x: len(x) % len(devices) == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {len(devices)} devices: {shapes}')
      # TODO: Avoid the reshape?
      value = tree_map(
          lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value)
      shards = []
      for i in range(len(devices)):
        shards.append(tree_map(lambda x: x[i], value))
      value = jax.device_put_sharded(shards, devices)
    return value

  def _convert_outs(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value, devices):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if len(devices) > 1:
      value = tree_map(lambda x: x[0], value)
    return value

  def _next_rngs(self, devices, mirror=False, high=2 ** 63 - 1):
    if len(devices) == 1:
      return jax.device_put(self.rng.integers(high), devices[0])
    elif mirror:
      return jax.device_put_replicated(
          self.rng.integers(high), devices)
    else:
      return jax.device_put_sharded(
          list(self.rng.integers(high, size=len(devices))), devices)

  def _init_varibs(self, obs_space, act_space):
    varibs = {}
    rng = self._next_rngs(self.train_devices, mirror=True)
    dims = (self.batch_size, self.batch_length)
    data = self._dummy_batch({**obs_space, **act_space}, dims)
    data = self._convert_inps(data, self.train_devices)
    state, varibs = self._init_train(varibs, rng, data['is_first'])
    varibs = self._train(varibs, rng, data, state, init_only=True)
    # obs = self._dummy_batch(obs_space, (1,))
    # state, varibs = self._init_policy(varibs, rng, obs['is_first'])
    # varibs = self._policy(
    #     varibs, rng, obs, state, mode='train', init_only=True)
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
