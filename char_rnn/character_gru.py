from collections import namedtuple
from timeit import default_timer as timer
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import *

# Dictionary maps character to its index, and vice versa.
class Dictionary:

  def __init__(self, paths, ascii_only=True):
    self._characters = []
    self._indices = {}
    
    for path in paths:
      with open(path, 'r') as f:
        while True:
          character = f.read(1)
          if not character:
            break
          elif character not in self._indices:
            if ascii_only and ord(character) > 255:
              continue
            self._indices[character] = len(self._characters)
            self._characters.append(character)
    self._size = len(self._characters)

  def character(self, index):
    if index >= self._size:
      return ' '
    return self._characters[index]

  def index(self, character):
    try:
      return self._indices[character]
    except KeyError:
      return self._size

  def Translate(self, indices):
    return ''.join([self.character(index) for index in indices])

  @property
  def size(self):
    return self._size + 1

  def __str__(self):
    return ('[' + str(len(self._characters)) + '] (' +
            ' '.join(self._characters) +')')


class Documents:
  """This class holds a set of documents, and provides APIs to create feeds."""

  def __init__(self, paths, ascii_only=True):
    # Construct the dictionary first
    self._dictionary = Dictionary(paths, ascii_only)

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

  def Feed(self, sequence_length):
    """Generator that keeps yielding next sequence for training."""

    for doc_id in range(len(self._docs)):
      doc_size = len(self._docs[doc_id])

      # TODO(breakds): Does not have to start from 0, in fact it might
      # be a good idea to initialize it from some random number
      # between 0 and sequence_length - 1.
      pos = 0

      reset_state = True

      while True:
        if pos >= doc_size - 1:
          break

        end = min(pos + sequence_length, doc_size - 1)

        # Set ndmin=2 so that inputs and targets become 2 dimensional
        # (with a "batch_size" of 1) instead of 1.
        inputs = np.array(self._docs[doc_id][pos:end], ndmin=2)
        targets = np.array(self._docs[doc_id][(pos + 1):(end + 1)], ndmin=2)

        yield inputs, targets, reset_state

        reset_state = False
        pos = end

  def _Next(self, pos, doc_id, sequence_length, doc_indices=None):
    if not doc_indices:
      doc_indices = range(len(self._docs))

    reset_state = False

    if pos + sequence_length >= len(self._docs[doc_indices[doc_id]]):
      doc_id = (doc_id + 1) % len(self._docs)
      pos = 0
      reset_state = True

    end = pos + sequence_length
    inputs = self._docs[doc_indices[doc_id]][pos:end]
    targets = self._docs[doc_indices[doc_id]][(pos + 1):(end + 1)]

    pos = end

    return inputs, targets, reset_state, pos, doc_id

  def FeedBatchByDocument(self, sequence_length):
    pos = [0 for _ in self._docs]
    doc_id = [i for i in range(len(self._docs))]
    reach_end_once = [False for _ in self._docs]
    reach_end_count = 0

    inputs = np.zeros((len(self._docs), sequence_length), dtype=np.int32)
    targets = np.zeros((len(self._docs), sequence_length), dtype=np.int32)
    
    while reach_end_count < len(self._docs):
      reset_state = [False for _ in self._docs]
      for i in range(len(self._docs)):
        if pos[i] + sequence_length >= len(self._docs[doc_id[i]]):
          doc_id[i] = (doc_id[i] + 1) % len(self._docs)
          pos[i] = 0
          if not reach_end_once[i]:
            reach_end_once[i] = True
            reach_end_count += 1

        if pos[i] == 0:
          reset_state[i] = True

        end = pos[i] + sequence_length

        inputs[i] = self._docs[doc_id[i]][pos[i]:end]
        targets[i] = self._docs[doc_id[i]][(pos[i] + 1):(end + 1)]

        pos[i] = end

      yield inputs, targets, reset_state

  def FeedBatch(self, sequence_length, batch_size):
    total_length = sum([len(doc) for doc in self._docs])
    steps = math.ceil(total_length / (batch_size * sequence_length)) + 1

    # Create random permutation of the documents
    doc_indices = np.random.permutation(len(self._docs))

    pos = [0] * batch_size
    # Here doc_id means the id in doc_indices
    doc_id = [0] * batch_size

    # Initialize pos and doc_id
    accu_length = 0
    accu_doc_id = 0
    current_length = len(self._docs[doc_indices[accu_doc_id]])
    for i in range(batch_size):
      current_overall_pos = math.ceil(i * total_length / batch_size)
      while current_overall_pos >= accu_length + current_length:
        accu_length += current_length
        accu_doc_id += 1
        current_length = len(self._docs[doc_indices[accu_doc_id]])
      doc_id[i] = accu_doc_id
      pos[i] = current_overall_pos - accu_length

    inputs = np.zeros((batch_size, sequence_length), dtype=np.int32)
    targets = np.zeros((batch_size, sequence_length), dtype=np.int32)
    reset_state = [False] * batch_size

    for step in range(steps):
      for i in range(batch_size):
        inputs[i], targets[i], reset_state[i], pos[i], doc_id[i] = (
          self._Next(pos[i], doc_id[i], sequence_length, doc_indices))
      if step == 0:
        reset_state = [False] * batch_size
      yield inputs, targets, reset_state

  @property
  def size(self):
    return len(self._docs)
  
  @property
  def dictionary(self):
    return self._dictionary

# Character Recurrent Neural Network
class CharRNN:
  """Reccurent Network with GRU for character modeling."""

  Properties = namedtuple('Properties', ['dict_size', 'gru_size',
                                         'sequence_length'])
  
  Graph = namedtuple('Graph', ['inputs', 'binary_inputs',
                               'targets', 'initial_state',
                               'rnn_output', 'state', 'logits',
                               'prediction', 'loss', 'train_op'])

  def __init__(self, dictionary,
               # The size of the GRU output state size.
               gru_size,
               # The length of very input sequence.
               sequence_length):
    """Constructor of the RNN Model."""

    self._props = self.Properties(gru_size=gru_size,
                                  sequence_length=sequence_length,
                                  dict_size=dictionary.size)

    # ---------- Graph Construction --------------------

    # The inputs are a batch of indices from the dictionary.
    # Shape: batch_size * time_steps (sequence length)
    inputs = tf.placeholder(tf.int32, [None, sequence_length])
    # The target are a batch of indices from the dictionary.
    # Shape: batch_size * time_steps (sequence length)
    targets = tf.placeholder(tf.int64, [None, sequence_length])
    # The initial state shape: batch_size * gru_size
    initial_state = tf.placeholder(tf.float32, [None, gru_size])


    with tf.device('/cpu:0'):
      # First need to convert the input indices to sparse binary vectors.
      binaries = tf.constant(np.identity(dictionary.size), dtype=tf.float32)
      # binary_inputs shape: batch_size * sequence_length * dictionary_size
      binary_inputs = tf.nn.embedding_lookup(binaries, inputs)

    # According to http://arxiv.org/pdf/1409.2329.pdf, dropout might
    # be helpful here.
    # TODO(breakds): Add dropout.
    #
    # Calling GRUCell does not return an output tensor like other
    # operators does. In fact it returns an operator. Applying the
    # returning operator "gru_op" to other inputs to obtain actual output.
    gru_op = GRUCell(gru_size)
    
    # For the operator rnn, inputs must be a length T list of inputs,
    # each a tensor of shape [batch_size, cell.input_size].
    #
    # By not providing initial state, this RNN will have a zero
    # intiialized state.
    #
    # The split is on the second dimension (dimension 1) since that is
    # the dimension of sequence length.
    input_list = [tf.squeeze(per_time_input, [1]) for per_time_input
                  in tf.split(1, sequence_length, binary_inputs)]

    # both rnn_output and state are lists.
    # rnn_output shape: [batch_size * gru_size] * sequence_length
    # state shape: [batch_size * gru_size] * sequence_length
    rnn_output, state = rnn(gru_op,
                            input_list,
                            initial_state=initial_state,
                            scope="CharRNN")

    # combined_rnn_output shape: (batch_size * sequence_length) * gru_size
    combined_rnn_output = tf.concat(0, rnn_output, name="Combined")

    # logits shape: (batch_size * sequence_length) * dictionary.size
    logits = linear(combined_rnn_output, dictionary.size,
                    True, # has bias,
                    scope='Logits')

    # ---------- Prediction Only ----------
    # probability shape: (batch_size * sequence_length) * dictionary.size
    # Softmax operation preserves the shape.
    probability = tf.nn.softmax(logits, name='Softmax')
    prediction = tf.argmax(probability, 1, name='Prediction')
    # ---------- Prediction Only ----------

    # ---------- Training Only ----------
    # cross_entropies shape: (batch_size * sequence_length)
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, tf.reshape(targets, [-1]))

    # Reduce the cross_entropies to one loss value.
    loss = tf.reduce_mean(cross_entropies)

    self._lr = tf.Variable(0.001, trainable=False)
    train_op = tf.train.AdamOptimizer(self._lr).minimize(loss)
    # ---------- End Training Only ----------
    
    # Organize the graph
    self._graph = self.Graph(
      inputs=inputs,
      binary_inputs=binary_inputs,
      targets=targets,
      initial_state=initial_state,
      rnn_output=rnn_output,
      state=state,
      logits=logits,
      prediction=prediction,
      loss=loss,
      train_op=train_op)

  def assign_learning_rate(self, sess, learning_rate):
    sess.run(tf.assign(self.learning_rate, learning_rate))

  def TrainStep(self, sess, inputs, targets, initial_state):
    """Returns the loss of the training step."""

    loss, state, _ = sess.run([self.graph.loss, self.graph.state,
                               self.graph.train_op],
                              feed_dict={
                                self.graph.inputs: inputs,
                                self.graph.targets: targets,
                                self.graph.initial_state: initial_state
                              })
    return loss, state

  def EvalLoss(self, sess, inputs, targets, initial_state):
    return sess.run(self.graph.loss,
                    feed_dict={
                      self.graph.inputs: inputs,
                      self.graph.targets: targets,
                      self.graph.initial_state: initial_state
                    })

  def Generate(self, sess, initial_index, initial_state, steps):
    # Note that in this case, we expect the sequence length to be 1
    assert self.sequence_length == 1

    result = []

    x = [initial_index]
    state = initial_state
    for i in xrange(steps):
      # Make it 2-dimension to fit.
      inputs = np.array([x])
      x, state = sess.run([self.graph.prediction,
                                    self.graph.state],
                                   feed_dict={
                                     self.graph.inputs: inputs,
                                     self.graph.initial_state: state
                                   })
      result.append(x[0])

    return result, state

  @property
  def sequence_length(self):
    return self._props.sequence_length
  
  @property
  def learning_rate(self):
    return self._lr

  @property
  def dict_size(self):
    return self._props.dict_size

  @property
  def gru_size(self):
    return self._props.gru_size

  @property
  def graph(self):
    return self._graph

class ExponentialMovingAverager:
  def __init__(self, alpha):
    self._alpha = alpha
    self._value = 0.0
    self._first = True

  def Add(self, new_value):
    if self._first:
      self._first = False
      self._value = new_value
    else:
      self._value = new_value * (1.0 - self._alpha) + self._value * self._alpha

  @property
  def value(self):
    return self._value
    

def main(train=True):
  # documents = Documents(['/home/breakds/pf/projects/mh-dex-frontend/app/states/database.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/weapon.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/jewel-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/armor.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/quest-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/quest.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/environment.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/item-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/skill.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/armor-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/search-result.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/weapon-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/avenger.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/item.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/avenger-result.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/skill-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/reducers.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/jewel.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/monster-detail.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/monster.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/common/weapon-db.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/common/paged-list-dispatcher.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/common/dex-db.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/states/common/detail-dispatcher.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/avenger/charm-editor.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/avenger/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/avenger/charm.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/avenger/result.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/item/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/main.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/skill-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/credits/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/armor/columns.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/armor/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/quest-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/monster-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/jewel/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/quest/columns.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/quest/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/utils/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/detail-link.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/detail-panels.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/quest-rank.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/pagination.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/page.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/avenger-panels.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/weapon-widgets.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/slot-symbol.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/data-table.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/filter.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/components/list-table.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/search/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/item-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/item-detail/common.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/item-detail/usage-panel.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/item-detail/acquire-panel.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/root.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/armor-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/home/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/skill/columns.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/skill/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/weapon-detail/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/monster/columns.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/monster/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/weapon/control-panel.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/weapon/table.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/weapon/index.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/weapon/formatter.js',
  #                        '/home/breakds/pf/projects/mh-dex-frontend/app/jewel-detail/index.js'])

  documents = Documents(['/home/breakds/pf/projects/future-racer/experiment/character_model/data/tiny-shakespeare.txt'])

  # documents = Documents(['/home/breakds/pf/projects/future-racer/experiment/character_model/data/simple.txt'])

  print('Dictionary: %d' % documents.dictionary.size)
  
  sequence_length = 24
  gru_size = 32

  max_epochs = 100
  learning_rate_decay = 0.98
  batch_size = 36

  with tf.Graph().as_default(), tf.Session() as sess:

    if train:
      model = CharRNN(documents.dictionary, gru_size, sequence_length)
      learning_rate = 0.1
      model.assign_learning_rate(sess, learning_rate)
      saver = tf.train.Saver()

      sess.run(tf.initialize_all_variables())

      ema = ExponentialMovingAverager(0.95)

      for epoch in range(max_epochs):

        learning_rate *= learning_rate_decay
        model.assign_learning_rate(sess, learning_rate)
        
        start_time = timer()
        state = np.zeros((batch_size, gru_size))
        inner_steps = 0
        for inputs, targets, reset_state in documents.FeedBatch(sequence_length, batch_size):
          # Handle reset state
          for i in range(batch_size):
            if reset_state[i]:
              state[i] = np.zeros(gru_size)

          loss, state = model.TrainStep(sess, inputs, targets, state)

          ema.Add(loss)
          
          inner_steps += 1
          if inner_steps % 20 == 0:
            print('=', end='', flush=True)
        end_time = timer()
        print('\nEpoch %05d: %.4f, training cost: %.4f sec' % (epoch, ema.value,
                                                               end_time - start_time))
        
        saver.save(sess, '/home/breakds/tmp/char_gru/model.chpt')        
    else:
      model = CharRNN(documents.dictionary, gru_size, 1)
      saver = tf.train.Saver()
      sess.run(tf.initialize_all_variables())

      saver.restore(sess, '/home/breakds/tmp/char_gru/model.chpt')

      with open('/home/breakds/tmp/output.txt', 'w') as output:
        state = np.zeros((1, gru_size))
        index = round(np.random.uniform(0, 80))
        for i in xrange(100):
          indices, state = model.Generate(sess, index, state, 36)
          print(indices)
          index = indices[-1]
          output.write(documents.dictionary.Translate(indices))

      
      
      
