"""Run bounded native benchmark interactions without training or model imports."""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recall2imagine'))
import embodied
from ruamel.yaml import YAML
from train import make_env


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task', nargs='+', default=[
      'bsuite_memory_len/4', 'bsuite_discounting_chain/10'])
  parser.add_argument('--steps', type=int, default=256,
                      help='Maximum native interaction records per task')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  if args.steps < 1:
    parser.error('--steps must be positive')
  np.random.seed(args.seed)
  defaults = YAML(typ='safe').load(
      (ROOT / 'recall2imagine' / 'configs.yaml').read_text())['defaults']
  for task in args.task:
    config = embodied.Config(defaults).update({'task': task, 'seed': args.seed})
    env = make_env(config)
    try:
      totals = {'task': task, 'records': 0, 'transitions': 0,
                'episodes': 0, 'terminals': 0, 'return_sum': 0.0}
      for step in range(args.steps):
        action = {key: value.sample() for key, value in env.act_space.items()
                  if key != 'reset'}
        action['reset'] = step == 0
        obs = env.step(action)
        for key, value in obs.items():
          if key in env.obs_space and not np.isfinite(np.asarray(value)).all():
            raise RuntimeError(f'Nonfinite observation field {key}')
        totals['records'] += 1
        totals['transitions'] += int(not obs['is_first'])
        totals['episodes'] += int(obs['is_last'])
        totals['terminals'] += int(obs['is_terminal'])
        totals['return_sum'] += float(obs['reward'])
      totals['observation_shapes'] = {k: list(v.shape) for k, v in env.obs_space.items()}
      totals['action_shapes'] = {k: list(v.shape) for k, v in env.act_space.items()}
      print(json.dumps(totals, sort_keys=True))
    finally:
      env.close()


if __name__ == '__main__':
  main()
