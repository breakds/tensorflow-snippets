import math
import bisect
import numpy as np

from dictionary import Dictionary

class DocumentSet:
  """This class holds a set of documents, and provides APIs to create feeds."""

  def __init__(self, paths, ascii_only=True):
    # Construct the dictionary first
    self._dictionary = Dictionary(paths, ascii_only)

    # A list of documents. Each document is an integer list of
    # character indices.
    self._docs = []

    for path in paths:
      ids = []
      with open(path, 'r') as f:
        while True:
          character = f.read(1)
          if not character:
            break
          else:
            ids.append(self.dictionary.index(character))
      self._docs.append(np.array(ids))

  @property
  def docs(self):
    return self._docs

  @property
  def size(self):
    return len(self._docs)
  
  @property
  def dictionary(self):
    return self._dictionary

  def FeedBatches(self, sequence_length, batch_size):
    """Generator that produces mini batches which cover all the document."""
    # Feed mini batches for training. Each mini batch consists of
    # inputs, targets and reset_state.
    #
    # sequence_length: number of time steps in each sequence
    # batch_size: the number of sequences in each mini batch

    # Details: randomly pick batch_size start points, where each start
    # points is (total_length/batch_size) away from its neighbor start
    # points. At each yield, collect batch_size sequences with those
    # start points, and move those start points forward.

    # -- I. Create random permutation of docs --
    doc_perm = np.random.permutation(self.size)

    # -- II. Sample start points (doc_ids and offsets) --
    #
    # Each start points is specified by one doc_id and one offset
    # within that document. Note that the doc_id is w.r.t. the
    # permutation doc_perm.
    total_length = sum([len(doc) for doc in self.docs])
    offset_distance = math.ceil(total_length / batch_size)
    offset = []
    doc_id = []

    doc_cumsum = np.cumsum([len(self.docs[ind]) for ind in doc_perm])

    start = np.random.randint(total_length)
    for i in range(batch_size):
      doc_id.append(bisect.bisect_left(doc_cumsum, start))
      offset.append(start - (doc_cumsum[doc_id[-1]] if doc_id[-1] > 0 else 0))
      start = (start + offset_distance) % total_length
    
    # -- III. Generating batches --
    inputs = np.zeros((batch_size, sequence_length), dtype=np.int32)
    targets = np.zeros((batch_size, sequence_length), dtype=np.int32)
    reset_state = [False] * batch_size
    
    number_of_batches = math.ceil(total_length / (batch_size * sequence_length)) + 1
    for batch_id in range(number_of_batches):
      for i in range(batch_size):
        if offset[i] + sequence_length >= len(self.docs[doc_perm[doc_id[i]]]):
          doc_id[i] = (doc_id[i] + 1) % self.size
          offset[i] = 0
          reset_state[i] = True

        end = offset[i] + sequence_length
        inputs[i] = self.docs[doc_perm[doc_id[i]]][offset[i]:end]
        targets[i] = self.docs[doc_perm[doc_id[i]]][(offset[i] + 1) : (end + 1)]

      if batch_id == 0:
        reset_state = [True] * batch_size

      yield inputs, targets, reset_state
