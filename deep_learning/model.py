import functools
import numpy as np
import tensorflow as tf

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class StocksPredictorModel:

    def __init__(self, data, target, dropout, learning_rate=0.001, num_hidden=8, num_layers=1):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.learning_rate = learning_rate
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.LSTMCell(self._num_hidden)
        network = tf.contrib.rnn.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)

        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        out_size = int(self.target.get_shape()[1])
        weight, bias = self._weight_and_bias(self._num_hidden, out_size)

        return tf.matmul(last, weight) + bias

    @lazy_property
    def cost(self):
        return tf.nn.l2_loss(tf.subtract(self.prediction, self.target))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(num_hidden, out_size):
        weight = tf.Variable(tf.truncated_normal([num_hidden, out_size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return weight, bias

    @staticmethod
    def accuracy(predictions, labels):
      err = np.sum( np.isclose(predictions, labels, 0.0, 0.005) ) / (predictions.shape[0] * predictions.shape[1])
      return (100.0 * err)
