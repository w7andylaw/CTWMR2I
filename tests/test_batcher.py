"""Shared-memory batching must allocate one buffer before publishing rows."""

import importlib.util
import pathlib
import threading
import time
import unittest

import numpy as np


class SharedBufferInitializationTest(unittest.TestCase):

  def test_concurrent_creators_keep_all_rows(self):
    path = (pathlib.Path(__file__).resolve().parents[1] / 'recall2imagine' /
            'embodied' / 'core' / 'batcher.py')
    spec = importlib.util.spec_from_file_location('batcher_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    second_allocation = threading.Event()
    first_written = threading.Event()
    allocation_lock = threading.Lock()
    allocations = []

    class Serializer:
      def batch_buffer(self, prefetch, batch, length):
        with allocation_lock:
          index = len(allocations)
          allocations.append(index)
        if index == 0:
          second_allocation.wait(timeout=0.2)
        else:
          second_allocation.set()
          if not first_written.wait(timeout=2):
            raise RuntimeError('No sample reached first allocation')
        return {'value': np.full((prefetch, batch, length), -999, np.int32)}

    class Replay:
      ready = True
      serializer = Serializer()

      def sample(self, flip, worker, buffers):
        buffers['value'][flip, worker] = 100 + worker
        if worker == 0 and not first_written.is_set():
          first_written.set()
          time.sleep(0.1)
        return True

    batcher = module.BatcherSM(
        Replay(), workers=2, batch_size=2, batch_sequence_len=3,
        prefetch_batch=1)
    batches = []
    consumer = threading.Thread(
        target=lambda: batches.append(next(batcher)), daemon=True)
    try:
      consumer.start()
      consumer.join(timeout=5)
      self.assertTrue(batches, 'Batch construction stalled')
      self.assertEqual(len(allocations), 1)
      np.testing.assert_array_equal(
          batches[0]['value'], np.asarray([[100] * 3, [101] * 3]))
    finally:
      batcher._running = False


if __name__ == '__main__':
  unittest.main()
