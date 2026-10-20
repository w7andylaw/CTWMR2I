"""Real Uniform persistence must preserve windows and restore only once."""

import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recall2imagine'))
import embodied
from embodied.replay import chunk as chunklib


def step(index, worker=0):
  return {'observation': np.int32(worker * 1000 + index),
          'reward': np.float32(index), 'is_first': index == 0,
          'is_last': False, 'is_terminal': False}


def windows(replay):
  return sorted(tuple(int(s['observation']) for s in seq)
                for seq in replay.table.values())


class ReplayPersistenceTests(unittest.TestCase):

  def test_restore_is_idempotent_across_all_startup_entry_points(self):
    with tempfile.TemporaryDirectory() as directory:
      original = embodied.replay.Uniform(3, 100, directory, chunks=4)
      for index in range(12):
        original.add(step(index))
      original.save(wait=True)
      restored = embodied.replay.Uniform(3, 100, directory, chunks=4)
      expected = windows(original)
      self.assertEqual(windows(restored), expected)
      for restore in [restored.maybe_restore, restored.load,
                      lambda: restored.load(None)]:
        restore()
        self.assertEqual(len(restored), len(original))
        self.assertEqual(windows(restored), expected)

  def test_multiple_partial_snapshots_do_not_duplicate_experience(self):
    with tempfile.TemporaryDirectory() as directory:
      original = embodied.replay.Uniform(3, 100, directory, chunks=4)
      for index in range(6):
        original.add(step(index))
      original.save(wait=True)
      for index in range(6, 11):
        original.add(step(index))
      original.save(wait=True)
      restored = embodied.replay.Uniform(3, 100, directory, chunks=4)
      self.assertEqual(windows(restored), windows(original))
      self.assertEqual(len(restored), 9)

  def test_parallel_chunk_reads_keep_each_workers_successor_order(self):
    with tempfile.TemporaryDirectory() as directory:
      original = embodied.replay.Uniform(4, 100, directory, chunks=3)
      for index in range(9):
        for worker in [0, 1]:
          original.add(step(index, worker), worker)
      original.save(wait=True)
      load = chunklib.Chunk.load
      oldest = min(chunklib.Chunk.scan(embodied.Path(directory)))

      def delayed_load(filename, load_data=True):
        # Force the first read to finish after its successors. Unordered
        # emission creates discontinuous windows despite correct metadata.
        if load_data and filename == oldest:
          time.sleep(0.05)
        return load(filename, load_data=load_data)

      with mock.patch.object(chunklib.Chunk, 'load', side_effect=delayed_load):
        restored = embodied.replay.Uniform(4, 100, directory, chunks=3)
      self.assertEqual(windows(restored), windows(original))
      self.assertEqual(len(restored), 12)
      for window in windows(restored):
        np.testing.assert_array_equal(np.diff(window), [1, 1, 1])
        self.assertEqual(window[0] // 1000, window[-1] // 1000)


if __name__ == '__main__':
  unittest.main()
