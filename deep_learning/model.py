import functools
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

    def __init__(self, data, target, dropout, learning_rate=0.001, num_hidden=128, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._learning_rate = learning_rate
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

        companies_num, classes_num = int(self.target.get_shape()[1]), int(self.target.get_shape()[2])
        out_size = companies_num * classes_num
        weight, bias = self._weight_and_bias(self._num_hidden, out_size)

        prediction = tf.matmul(last, weight) + bias
        prediction = tf.reshape(prediction, [-1, companies_num, classes_num])
        return tf.nn.softmax(prediction)

    @lazy_property
    def cost(self):
        return -tf.reduce_sum(self.target * tf.log(self.prediction))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(num_hidden, out_size):
        weight = tf.Variable(tf.truncated_normal([num_hidden, out_size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return weight, bias
