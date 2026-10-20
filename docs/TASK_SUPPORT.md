# Native benchmark adapters

The migrated project uses native BSuite, POPGym, Memory Maze, Atari and DMC
tasks. A runnable adapter does not establish that a learning run has converged.
The short environment checks described here do not measure agent performance.

## BSuite

The explicit parameter entry points are:

```bash
python scripts/smoke_env.py --task bsuite_memory_len/4 bsuite_discounting_chain/10
```

`bsuite_memory_len/N` supplies N directly as the native `memory_length` value.
This suffix is **not** a BSuite sweep index or a manually changed episode limit.
The initial cue, native time/query fields, native final reward (+1/-1), and
termination remain unchanged. The native `(1, 3)` observation is flattened to
three floats, and the two native actions are one-hot encoded by R2I's wrapper.

`bsuite_discounting_chain/D` chooses which native delay in {1, 3, 10, 30, 100}
has the 1.1 reward. The other chains still earn 1.0. The first action selects
the chain, later actions have their native behavior, and each episode lasts
100 native transitions. No reward shaping is applied. This task's seed selects
no additional native stochasticity once the bonus delay is fixed.

The legacy `dm_bsuite_memory_len/5` entry point is retained. Its suffix **is**
a native BSuite sweep index and therefore has different semantics.

Training uses the same task IDs, for example
`python -m recall2imagine.train --configs bsuite --task bsuite_memory_len/4`.
Use the project setup and complete run command in README before training;
`smoke_env.py` never starts model training.

BSuite's undiscounted score is distinct from the learner's discounted objective.
At discount approximately .997, the 100-step chain's reward 1.1 can have a lower
discounted value than the immediately paid 1.0. This is a property of that
configuration; an unchanged 100-delay benchmark must not be described as an
unambiguous test of memory capacity alone.

## POPGym

```bash
python scripts/smoke_env.py --task gym_popgym:popgym-RepeatPreviousEasy-v0
```

RepeatPrevious, Autoencode and Concentration support Easy/Medium/Hard through
the native registered Gymnasium IDs. Categorical observations are concatenated
float32 one-hot vectors, as in R2I. A native MultiDiscrete action, if present,
maps bijectively to a single categorical index. Native rewards and the order
required by the native Autoencode implementation are preserved.

Gymnasium `terminated` sets both `is_terminal` and `is_last`; `truncated` sets
only `is_last`, retaining continuation for time-limit bootstrapping. Native
legacy Gym `TimeLimit.truncated` and explicit `is_terminal` metadata are also
respected. The adapter seeds the first reset only, rather than repeating the
same RNG initialization on every episode. Vector workers use seed + worker ID.

## Visual domains

The upstream task paths remain available: `gym_memory_maze:MemoryMaze-9x9-v0`,
`atari_pong`, and `dmc_walker_walk`, with the domain presets controlling their
observation modality and action repetition. Memory Maze 1.0.2 uses native legacy
Gym. ROMs, MuJoCo/rendering and domain-specific packages must be available.
These visual tasks require their own runtime smoke checks; successful BSuite
or POPGym checks do not validate the visual domains.

## Regression checks

```bash
python -m unittest discover -s tests -p test_envs.py -v
```

The checks compare BSuite trajectories with native implementations; verify
all five bonus delays, one-hot/action mappings, automatic reset and terminal
versus truncation semantics; and reset/step the nine POPGym tasks when installed.

The isolated WSL Python 3.11 validation with POPGym 1.0.2 installed passed all
10 tests, without skips. Each of the nine POPGym tasks was checked both through
the direct adapter and through the actual training factory, with space checks
enabled for the factory path. The observations use float32, and the factory's
returned keys agree with its advertised spaces, including the step counter.

A separate bounded 256-record random-policy smoke ran each POPGym task through
the training factory (seed 7). Observation/action widths were:

| Task | Easy | Medium | Hard |
| --- | --- | --- | --- |
| RepeatPrevious | 4 / 4 | 4 / 4 | 4 / 4 |
| Autoencode | 6 / 4 | 6 / 4 | 6 / 4 |
| Concentration | 156 / 52 | 312 / 104 | 728 / 52 |

The widths describe the flattened one-hot observations and one-hot actions,
respectively. All returned values were finite. Concentration's native time
limits ended episodes without incorrectly marking them terminal. The Hard
Autoencode smoke did not reach the end of its longer episode within 256 records;
this check establishes runnable reset/step interfaces, not task solution quality.

The earlier 256-record BSuite smoke completed 42 native Memory Length episodes
and two native Discounting Chain episodes through the same training factory.
Memory Length supplied three observation values/two actions; Discounting Chain
supplied two observation values/five actions. Visual domains were not exercised
by these checks.
