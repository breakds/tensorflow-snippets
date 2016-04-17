from collections import namedtuple
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import *

class ExponentialMovingAverager:
  """Maintains an exponential moving average value."""
  
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

class CharRNN:
  """Reccurent Network with GRU for character modeling."""

  Properties = namedtuple('Properties',
                          ['dictionary',
                           'gru_size',
                           'sequence_length',
                           'embedding_size'])
  
  Graph = namedtuple('Graph', ['inputs',
                               'embedded_inputs',
                               'targets',
                               'initial_state',
                               'rnn_output',
                               'state',
                               'logits',
                               'prediction',
                               'loss',
                               'train_op'])
  
  class Parameters:
    def __init__(self):
      self.max_epochs = 10
      self.batch_size = 100
      self.restore_path = None
      self.save_path = None
      self.learning_rate = 0.01
      self.learning_rate_decay = None
      self.loss_ema_decay = 0.95

  def __init__(self, dictionary,
               gru_size, # The size of the GRU output state size.
               sequence_length, # The length of very input sequence.
               # when embedding_size is None, use onehot input
               # instead.
               embedding_size=None):
    """Constructor of the RNN Model."""

    # -- I. Save configurations --

    self._props = self.Properties(
      gru_size=gru_size,
      sequence_length=sequence_length,
      dictionary=dictionary,
      embedding_size = embedding_size)

    # -- II. Network inputs --

    # The inputs are a batch of indices from the dictionary.
    # Shape: batch_size * time_steps (sequence length)
    inputs = tf.placeholder(tf.int32, [None, sequence_length])
    # The target are a batch of indices from the dictionary. Shape:
    # batch_size * time_steps (sequence length)
    targets = tf.placeholder(tf.int64, [None, sequence_length])
    # The initial state shape: batch_size * gru_size
    initial_state = tf.placeholder(tf.float32, [None, gru_size])

    # -- III. Handle Embedding if required --

    if embedding_size:
      # Create embedding matrix.
      # Shape: dictionary_size * embedding_size
      embedding_matrix = tf.get_variable('embedding_matrix',
                                         (dictionary.size, embedding_size),
                                         tf.float32,
                                         tf.truncated_normal_initializer(mean=0.0,
                                                                         stddev=0.1,
                                                                         dtype=tf.float32))
      embedding_bias = tf.get_variable('embedding_bias', (embedding_size), tf.float32,
                                       tf.truncated_normal_initializer(mean=0.0,
                                                                       stddev=0.1,
                                                                       dtype=tf.float32))
      # embedded_inputs shape: batch_size * sequence_length * embedding_size
      embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs) + embedding_bias
    else:
      # Otherwise, use onehot inputs instead.
      identity = tf.constant(np.identity(dictionary.size, dtype=np.float32), dtype=tf.float32)
      embedded_inputs = tf.nn.embedding_lookup(identity,  inputs)

    # -- IV. Recurrent neural network units --

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
                  in tf.split(1, sequence_length, embedded_inputs)]

    # both rnn_output and state are lists.
    # rnn_output shape: [batch_size * gru_size] * sequence_length
    # state shape: [batch_size * gru_size] * sequence_length
    rnn_output, state = rnn(gru_op,
                            input_list,
                            initial_state=initial_state,
                            scope="CharRNN")

    # -- V. Linear transformation on RNN outputs --

    # combined_rnn_output shape: (batch_size * sequence_length) * gru_size
    combined_rnn_output = tf.concat(0, rnn_output, name="Combined")

    # logits shape: (batch_size * sequence_length) * dictionary.size
    logits = linear(combined_rnn_output, dictionary.size,
                    True, # has bias,
                    scope='Logits')

    # -- VI. Prediction operations --
    # probability shape: (batch_size * sequence_length) * dictionary.size
    # Softmax operation preserves the shape.
    probability = tf.nn.softmax(logits, name='Softmax')
    prediction = tf.argmax(probability, 1, name='Prediction')

    # -- VII. Training operatings --
    # cross_entropies shape: (batch_size * sequence_length)
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, tf.reshape(tf.transpose(targets), [-1]))

    # Reduce the cross_entropies to one loss value.
    loss = tf.reduce_mean(cross_entropies)

    self._lr = tf.Variable(0.001, trainable=False)
    train_op = tf.train.AdamOptimizer(self._lr).minimize(loss)
    
    # -- VIII. Save the graph --
    self._graph = self.Graph(
      inputs=inputs,
      embedded_inputs=embedded_inputs,
      targets=targets,
      initial_state=initial_state,
      rnn_output=rnn_output,
      state=state,
      logits=logits,
      prediction=prediction,
      loss=loss,
      train_op=train_op)

  @property
  def learning_rate(self):
    return self._lr

  @property
  def sequence_length(self):
    return self._props.sequence_length

  @property
  def gru_size(self):
    return self._props.gru_size

  @property
  def dictionary_size(self):
    return self._props.dictionary.size

  @property
  def dictionary(self):
    return self._props.dictionary

  @property
  def graph(self):
    return self._graph

  def assign_learning_rate(self, sess, learning_rate):
    "Set the learning rate to a new value."
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

  def Train(self, sess, documents, parameters):
    """Train the model on the specified document."""
    
    print('[info] Training Started')
    print('[info] Dictionary: %d' % documents.dictionary.size)
    print('[info] Document Set: %d' % documents.size)

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    if parameters.restore_path:
      saver.restore(sess, parameters.restore_path)

    self.assign_learning_rate(sess, parameters.learning_rate)

    ema = ExponentialMovingAverager(parameters.loss_ema_decay)

    state = np.zeros((parameters.batch_size, self.gru_size))
    for epoch in range(parameters.max_epochs):

      print('[info] Start epoch %05d ... ' % epoch)

      if parameters.learning_rate_decay:
        self.assign_learning_rate(sess, self.learning_rate * parameters.learning_rate_decay)

      start_time = timer()
      inner_steps = 0

      for inputs, targets, reset_state in documents.FeedBatches(self.sequence_length,
                                                                parameters.batch_size):
        for i in range(parameters.batch_size):
          if reset_state[i]:
            state[i] = np.zeros(self.gru_size)

        loss, state = self.TrainStep(sess, inputs, targets, state)

        ema.Add(loss)
        
        inner_steps += 1
        if inner_steps % 20 == 0:
          print('=', end='', flush=True)
      end_time = timer()
      
      print('\n[ ok ]Epoch %05d: %.4f, training cost: %.4f sec' % (epoch, ema.value,
                                                                   end_time - start_time))

      saver.save(sess, parameters.save_path)

  def Write(self, sess, model_path, output_path):
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    saver.restore(sess, model_path)

    with open(output_path, 'w') as output:
      state = np.zeros((1, self.gru_size))
      index = round(np.random.randint(self.dictionary_size))
      for i in xrange(100):
        indices, state = self.Generate(sess, index, state, 36)
        output.write(self.dictionary.Translate(indices))
      



    


    
  
