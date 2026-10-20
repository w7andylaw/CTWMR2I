import concurrent.futures
from multiprocessing.pool import ThreadPool
from collections import defaultdict, deque
from functools import partial as bind
import numpy as np
from tqdm import tqdm
import embodied

from . import chunk as chunklib


class Saver:

  def __init__(self, directory, chunks=1024):
    self.directory = embodied.Path(directory)
    self.directory.mkdirs()
    self.chunks = chunks
    self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
    self.workers = concurrent.futures.ThreadPoolExecutor(16)
    self.promises = deque()
    self.loading = False

  def add(self, step, worker):
    if self.loading:
      return
    buffer = self.buffers[worker]
    buffer.append(step)
    if buffer.length >= self.chunks:
      self.buffers[worker] = buffer.successor = chunklib.Chunk(self.chunks)
      self.promises.append(self.workers.submit(buffer.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)

  def save(self, wait=False):
    for buffer in self.buffers.values():
      if buffer.length:
        self.promises.append(self.workers.submit(buffer.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, capacity, length):
    filenames = chunklib.Chunk.scan(self.directory)
    if not filenames:
      return
    # Saving a partially filled chunk can leave several snapshots of the same
    # UUID. Only its longest (most complete) snapshot belongs to the stream.
    chunks = {}
    for filename in filenames:
      chunk = chunklib.Chunk.load(filename, load_data=False)
      previous = chunks.get(chunk.uuid)
      if previous is None or chunk.length >= previous.length:
        chunks[chunk.uuid] = chunk
    selected, total = [], 0
    for chunk in sorted(chunks.values(), key=lambda x: x.time, reverse=True):
      if capacity and total >= capacity:
        break
      selected.append(chunk)
      total += max(0, chunk.length - 1)
    chunks = {chunk.uuid: chunk for chunk in selected}
    # Successor links identify consecutive chunks from the same worker. Start
    # at each chain's oldest available chunk, never join unrelated workers.
    successors = {chunk.successor for chunk in selected}
    heads = sorted(
        (chunk for chunk in selected if chunk.uuid not in successors),
        key=lambda x: x.time)
    streamids = {}
    ordered = []
    for head in heads:
      chunk = head
      while chunk is not None:
        if chunk.uuid in streamids:
          raise ValueError('Replay chunks contain overlapping successor chains')
        streamids[chunk.uuid] = head.uuid
        ordered.append(chunk.filename)
        chunk = chunks.get(chunk.successor)
    if len(streamids) != len(chunks):
      raise ValueError('Replay chunks contain a cyclic successor chain')
    threads = min(len(ordered), 32)
    self.loading = True
    try:
      with ThreadPool(threads) as executor:
        # Parallel reads may finish in any order; emit them in stream order.
        for chunk in tqdm(executor.imap(chunklib.Chunk.load, ordered), total=len(ordered)):
          stream = streamids[chunk.uuid]
          for index in range(chunk.length):
            step = {k: v[index] for k, v in chunk.data.items()}
            yield step, stream
          del chunk
    finally:
      self.loading = False
