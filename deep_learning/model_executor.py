# Model creation
import time
import tensorflow as tf
import stocks_reader
from model import StocksPredictorModel
companies_num = len(stocks_reader.COMPANIES)

X_train, y_train, X_valid, y_valid, X_test, y_test = stocks_reader.read_data()

input_shape = X_train.shape[2]
output_shape = y_train.shape[2]

data = tf.placeholder(tf.float32, [None, 1, input_shape])
target = tf.placeholder(tf.float32, [None, 1, output_shape])
dropout = tf.placeholder(tf.float32)

num_hidden = 120
model = StocksPredictorModel(data, target, dropout, num_hidden, 5, learning_rate=0.001)

# Model execution
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restore = False
save = True
checkpoint_path = './checkpoints/model.ckpt'

if(restore):
    print('Restoring model...')
    saver.restore(sess, checkpoint_path)
    
epoch = 100000

for i in range(epoch):
    _, cost = sess.run([model.optimize, model.cost],
                                 {data: X_train, target: y_train, dropout: 0.5})
    print(time.strftime('%X') + ' - Minibatch loss at epoch %d: %f' % (i, cost.mean()))
    if i % 1000 == 0:
        train_err = sess.run(model.error,{data: X_valid, target: y_valid, dropout: 1.0})
        print('Validation error {:3.1f}%'.format(100 * train_err))

if(save):
    print('Saving model...')
    save_path = saver.save(sess, checkpoint_path)

train_err = sess.run(model.error,{data: X_train, target: y_train, dropout: 1.0})
print('Training error {:3.1f}%'.format(100 * train_err))

valid_err = sess.run(model.error,{data: X_valid, target: y_valid, dropout: 1.0})
print('Validation error {:3.1f}%'.format(100 * valid_err))

test_err = sess.run(model.error,{data: X_test, target: y_test, dropout: 1.0})
print('Test error {:3.1f}%'.format(100 * test_err))

predictions = sess.run(model.prediction,{data: X_valid, dropout: 1.0})

sess.close()
