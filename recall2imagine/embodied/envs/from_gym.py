import functools

import embodied
import gymnasium as gym_
import numpy as np

try:
  import gym
except ImportError:
  gym = None

def flatten(space, onehot=False):
  if isinstance(space, gym_.spaces.Discrete):
    n = space.n
    p = np.eye(n, dtype=np.float32)
    tr = lambda x: p[x]
    space_ = gym_.spaces.Box(0, 1, (n,))
  elif isinstance(space, gym_.spaces.MultiDiscrete):
    p = [np.eye(t, dtype=np.float32) for t in space.nvec]
    tr = lambda x: np.concatenate([eye[j] for eye, j in zip(p, x)], axis=0)
    size = space.nvec.sum()
    space_ = gym_.spaces.Box(0, 1, (size,))
  elif isinstance(space, gym_.spaces.Tuple):
    p = [np.eye(t.n, dtype=np.float32) for t in space.spaces]
    tr = lambda x: np.concatenate([eye[j] for eye, j in zip(p, x)], axis=0)
    space_ = gym_.spaces.Box(0, 1, (sum([len(pp) for pp in p]),))
  else:
    raise TypeError(f'Unsupported categorical observation space: {space!r}')
  return tr, space_

def unmap_action(a, nvec):
  nvec = np.asarray(nvec, np.int64)
  a = int(np.asarray(a))
  if not 0 <= a < int(np.prod(nvec)):
    raise ValueError('Flattened action index outside native action space')
  newa = np.zeros(nvec.size, dtype=np.int64)
  for i in reversed(range(nvec.size)):
    newa[i] = a % int(nvec.flat[i])
    a //= int(nvec.flat[i])
  return newa.reshape(nvec.shape)

class MyObsPopgym(gym_.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.tr, space_ = flatten(self.observation_space)
    self.maction = self.action_space if isinstance(self.action_space, gym_.spaces.MultiDiscrete) else None
    if self.maction is not None:
      self.action_space = gym_.spaces.Discrete(int(np.prod(self.action_space.nvec)))
    self.observation_space = gym_.spaces.Dict({'observation': space_})
    
  def step(self, action):
    if self.maction is not None:
      action = unmap_action(action, self.maction.nvec)
    obs, reward, ter, trun, info = super().step(action)
    obs = self.tr(obs)
    return {'observation': obs}, reward, ter, trun, info

  def reset(self, **kwargs):
    obs, info = super().reset(**kwargs)
    return {'observation': self.tr(obs)}, info

class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', seed=None, **kwargs):
    if isinstance(env, str):
      if 'popgym' in env.lower():
        self._env = gym_.make(env, **kwargs)
        self._env = MyObsPopgym(self._env)
      elif env.startswith('memory_maze:'):
        if gym is None:
          raise ImportError('Memory Maze 1.0.2 requires native legacy gym')
        self._env = gym.make(env, **kwargs)
      else:
        try:
          self._env = gym_.make(env, **kwargs)
        except gym_.error.Error:
          if gym is None:
            raise
          self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = self._is_dict_space(self._env.observation_space)
    self._act_dict = self._is_dict_space(self._env.action_space)
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self._seed = seed

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'step_no': embodied.Space(np.int32),
        # 'ep_no': embodied.Space(np.int32),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      # Seed only the first reset; later episodes advance the native RNG.
      if self._seed is not None:
        try:
          result = self._env.reset(seed=int(self._seed))
        except TypeError:
          self._env.seed(int(self._seed))
          result = self._env.reset()
        self._seed = None
      else:
        result = self._env.reset()
      if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        obs, self._info = result
      else:
        obs, self._info = result, {}
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten({k: v for k, v in action.items() if k != 'reset'})
    else:
      action = action[self._act_key]
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, self._info = result
      self._done = bool(terminated or truncated)
      is_terminal = bool(terminated)
    elif len(result) == 4:
      obs, reward, self._done, self._info = result
      is_terminal = bool(self._info.get(
          'is_terminal', self._done and not self._info.get('TimeLimit.truncated', False)))
    else:
      raise ValueError('Native step must return four (Gym) or five (Gymnasium) values')
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=is_terminal)

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    try:
      image = self._env.render()
    except TypeError:
      image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if self._is_dict_space(value):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if isinstance(space, gym_.spaces.Discrete) or (
        gym is not None and isinstance(space, gym.spaces.Discrete)):
      if getattr(space, 'start', 0) != 0:
        raise ValueError('Discrete spaces must use zero-based action IDs')
      return embodied.Space(np.int32, (), 0, space.n)
    if not hasattr(space, 'low') or not hasattr(space, 'high'):
      raise TypeError(f'Flatten this native space before bridging: {space!r}')
    return embodied.Space(space.dtype, space.shape, space.low, space.high)

  @staticmethod
  def _is_dict_space(space):
    return isinstance(space, gym_.spaces.Dict) or (
        gym is not None and isinstance(space, gym.spaces.Dict))
