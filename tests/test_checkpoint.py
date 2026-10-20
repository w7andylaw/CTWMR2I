"""Regression for LFS replay's model/step restoration side effects."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import recall2imagine
import embodied


class Value:
  def __init__(self, value):
    self.value = value

  def save(self):
    return self.value

  def load(self, value):
    self.value = value


class Replay(Value):
  def __init__(self, agent, step):
    super().__init__(128)
    self.agent, self.step = agent, step

  def load(self, value):
    self.value = value
    self.agent.load('older replay-prefix model')
    self.step.load(value)


class CheckpointTest(unittest.TestCase):
  def test_explicit_replay_first_restore_keeps_authoritative_model_and_step(self):
    with tempfile.TemporaryDirectory() as tmp:
      agent, step = Value('latest model'), Value(132)
      replay = Replay(agent, step)
      checkpoint = embodied.Checkpoint(pathlib.Path(tmp) / 'state.ckpt',
                                       parallel=False, log=False)
      checkpoint.step, checkpoint.agent, checkpoint.replay = step, agent, replay
      checkpoint.save()
      agent.load('untrained')
      step.load(0)
      checkpoint.load_or_save(keys=['replay', 'agent', 'step'])
      self.assertEqual(agent.value, 'latest model')
      self.assertEqual(step.value, 132)
      self.assertEqual(replay.value, 128)


if __name__ == '__main__':
  unittest.main()
