"""Native BSuite tasks with explicit parameters, rather than sweep indices."""

from .from_dm import FromDM


class BSuite(FromDM):
  """Use memory_len/N or discounting_chain/DELAY with native task rules."""

  DELAYS = (1, 3, 10, 30, 100)

  def __init__(self, task, seed=0, obs_key='observation', act_key='action'):
    try:
      name, value = task.rsplit('/', 1)
      value = int(value)
    except (ValueError, AttributeError) as exc:
      raise ValueError(
          'BSuite task must be memory_len/N or discounting_chain/DELAY') from exc
    if name == 'memory_len':
      if value < 1:
        raise ValueError('memory_length must be a positive integer')
      from bsuite.experiments.memory_len.memory_len import load
      env = load(memory_length=value, seed=int(seed))
    elif name == 'discounting_chain':
      if value not in self.DELAYS:
        raise ValueError(f'Discounting Chain delay must be one of {self.DELAYS}')
      from bsuite.environments.discounting_chain import DiscountingChain
      env = DiscountingChain(mapping_seed=self.DELAYS.index(value))
    else:
      raise ValueError(f'Unsupported BSuite task: {name!r}')
    self.task = task
    super().__init__(env, obs_key=obs_key, act_key=act_key)
