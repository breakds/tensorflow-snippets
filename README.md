# tensorflow-snippets
Complementary code snippets to official Tensorflow API documentation. I found that the most frustrating thing to learn a new framework such as [Tensorflow](https://www.tensorflow.org/versions/r0.7/api_docs/index.html) is to learn and remember all the APIs. The documentation is very helpful, but sometimes it is necessary to try some simple examples to make sure that my understanding is correct to gain some confidence of their behavior. This repo is merely a result of compiling those tiny simple experimentations into ipython notebooks.

I am just bootstrapping this effort. Please feel free to add whatever you want via pull requests or forks.

Thanks.

## Contents

### Tensor and Matrix Transformation

1. concat
   * [Notebook File](https://github.com/breakds/tensorflow-snippets/tree/master/snippets/concat.ipynb)
   * [Official Documentation](https://www.tensorflow.org/versions/r0.7/api_docs/python/array_ops.html#concat)

### Dealing with IDs and Embeddings

1. embedding_lookup
   * [Notebook File](https://github.com/breakds/tensorflow-snippets/tree/master/snippets/embedding_lookup.ipynb)
   * [Official Documentation](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#embedding_lookup)

### Operators for Loss Functions

1. Softmax and Cross Entropy
   *   [Notebook File](https://github.com/breakds/tensorflow-snippets/tree/master/snippets/softmax_and_cross_entropy.ipynb)
   *   Official Documentation
       * [softmax](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#softmax)
       * [softmax_cross_entropy_with_logits](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#softmax_cross_entropy_with_logits)
       * [sparse_softmax_cross_entropy_with_logits](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#softmax_cross_entropy_with_logits)
2. Softmax and Cross Entropy over time steps (for RNNs)
   *   [Notebook File](https://github.com/breakds/tensorflow-snippets/tree/master/snippets/sequence_loss_by_example.ipynb)
   *   Defined in [seq2seq.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py)

