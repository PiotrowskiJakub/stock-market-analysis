
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
        self.data = tf.placeholder(tf.float32, [1, 4, 1])  # [batch_size, companies_number * 2, input_dimension]
        self.target = tf.placeholder(tf.float32, [1, 2]) # [batch_size, companies_number]
        self.num_hidden = 8
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
        val, _ = tf.nn.dynamic_rnn(cell, self.data, dtype = tf.float32) # [batch_size, max_time, cell.output_size]
        val = tf.reshape(val, [-1, self.num_hidden]) # [batch_size * max_time, cell.output_size] - convert to 2D
        weight = tf.Variable(tf.truncated_normal([self.num_hidden, int(self.target.get_shape()[1])]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.target.get_shape()[1]]))
        incoming = tf.matmul(val, weight) + bias
        return incoming # try with RELU - tf.nn.relu(incoming)

    @lazy_property
    def optimize(self):
        loss = tf.nn.l2_loss(tf.sub(self.prediction, self.target))
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(loss)

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
