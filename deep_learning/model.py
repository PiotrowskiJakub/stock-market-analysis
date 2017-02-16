
import functools
import tensorflow as tf

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model:

    def __init__(self):
        self.data = tf.placeholder(tf.float32, [None, 6, 1])  # [BATCHSIZE, SEQLEN]
        self.target = tf.placeholder(tf.float32, [None, 3]) # [BATCHSIZE, PREDLEN]
        self.num_hidden = 24
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        val, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)
        # return val ?
        weight = tf.Variable(tf.truncated_normal([self.num_hidden, int(self.target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))
        incoming = tf.matmul(val, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.prediction,1e-10,1.0)))
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def gru():
        ALPHASIZE = 98
        CELLSIZE = 512
        NLAYERS = 3
        SEQLEN = 30

        cell = tf.contrib.rnn.GRUCell(CELLSIZE)
        mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS, state_is_tuple = False)
        Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state = Hin)
        # Hr [BATCHSIZE, SQLEN, CELLSIZE]
        Hf = tf.reshape(Hr, [-1, CELLSIZE]) # [BATCHSIZE * SEQLEN, CELLSIZE]
        Ylogits = layers.linear(Hf, ALPHASIZE) # [BATCHSIZE * SEQLEN, ALPHASIZE]
        Y = tf.nn.softmax(Ylogits) # [BATCHSIZE * SEQLEN, ALPHASIZE]
        loss = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y)
