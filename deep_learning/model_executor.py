# Model creation
import tensorflow as tf
import stocks_reader 
from model import StocksPredictorModel
companies_number = len(stocks_reader.COMPANIES)

train_input, train_output, valid_input, valid_output, test_input, test_output = stocks_reader.read_data()

data = tf.placeholder(tf.float32, [None, companies_number, 2])
target = tf.placeholder(tf.float32, [None, companies_number])
dropout = tf.placeholder(tf.float32)

model = StocksPredictorModel(data, target, dropout, 0.001)

# Model execution
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 1
no_of_batches = int(int(len(train_input)) / batch_size)
epoch = 10000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        _, cost, predictions = sess.run([model.optimize, model.cost, model.prediction],
                                     {data: inp, target: out, dropout: 1.0})
    acc = model.accuracy(predictions, out)
    print('Minibatch loss at step %d: %f' % (i, cost))
    print('Minibatch accuracy: %.1f%%' % acc)
 
train_err = sess.run(model.error,{data: train_input, target: train_output, dropout: 1.0})
print('Training error {:3.1f}%'.format(i + 1, 100 * train_err))

valid_acc = model.accuracy(sess.run(model.prediction,{data: valid_input, dropout: 1.0}), valid_output)
print('Validation accuracy: %.1f%%' % valid_acc)

test_acc = model.accuracy(sess.run(model.prediction,{data: test_input, dropout: 1.0}), test_output)
print('Test accuracy: %.1f%%' % test_acc)

sess.close()
