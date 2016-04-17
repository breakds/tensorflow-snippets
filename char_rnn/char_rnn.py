#!/user/bin/python3
import importlib
from dictionary import Dictionary
from document import DocumentSet
from rnn_model import CharRNN

import tensorflow as tf

def TrainModel(document_paths=None):
  if document_paths == None:
    document_paths = ['/home/breakds/dataset/char_rnn/shakes/tiny-shakespeare.txt']
  documents = DocumentSet(document_paths)

  with tf.Graph().as_default(), tf.Session() as sess:
    model = CharRNN(documents.dictionary,
                    gru_size=64,
                    sequence_length=25,
                    embedding_size=None)

    parameters = CharRNN.Parameters()
    parameters.save_path = '/home/breakds/tmp/char_rnn/model/model.chpt'

    model.Train(sess, documents, parameters)

def Generate(document_paths=None):
  if document_paths == None:
    document_paths = ['/home/breakds/dataset/char_rnn/shakes/tiny-shakespeare.txt']
  documents = DocumentSet(document_paths)
  with tf.Graph().as_default(), tf.Session() as sess:
    model = CharRNN(documents.dictionary,
                    gru_size=64,
                    sequence_length=1,
                    embedding_size=None)

    model.Write(sess,
                '/home/breakds/tmp/char_rnn/model/model.chpt',
                '/home/breakds/tmp/char_rnn/output.txt')

if __name__ == '__main__':
  # TrainModel()
  Generate()

