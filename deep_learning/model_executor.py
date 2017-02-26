# Model creation
import tensorflow as tf
import stocks_reader 
from model import StocksPredictorModel
companies_number = len(stocks_reader.COMPANIES)

X_train, y_train, X_valid, y_valid, X_test, y_test = stocks_reader.read_data()

data = tf.placeholder(tf.float32, [None, companies_number, 2])
target = tf.placeholder(tf.float32, [None, companies_number])
dropout = tf.placeholder(tf.float32)

model = StocksPredictorModel(data, target, dropout, 0.001)

# Model execution
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 1
no_of_batches = int(int(len(X_train)) / batch_size)
epoch = 10000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
        ptr+=batch_size
        _, cost, predictions = sess.run([model.optimize, model.cost, model.prediction],
                                     {data: inp, target: out, dropout: 1.0})
    acc = model.accuracy(predictions, out)
    print('Minibatch loss at step %d: %f' % (i, cost))
    print('Minibatch accuracy: %.1f%%' % acc)
 
train_err = sess.run(model.error,{data: X_train, target: y_train, dropout: 1.0})
print('Training error {:3.1f}%'.format(i + 1, 100 * train_err))

valid_acc = model.accuracy(sess.run(model.prediction,{data: X_valid, dropout: 1.0}), y_valid)
print('Validation accuracy: %.1f%%' % valid_acc)

test_acc = model.accuracy(sess.run(model.prediction,{data: X_test, dropout: 1.0}), y_test)
print('Test accuracy: %.1f%%' % test_acc)

sess.close()
