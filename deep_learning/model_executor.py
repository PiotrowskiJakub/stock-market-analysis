# Model creation
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

num_hidden = companies_num
model = StocksPredictorModel(data, target, dropout, num_hidden, learning_rate=0.003)

# Model execution
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restore = False
save = False
checkpoint_path = './checkpoints/model.ckpt'

batch_size = 10
batches_num = int(int(len(X_train)) / batch_size)
epoch = 100

if(restore):
    print('Restoring model...')
    saver.restore(sess, checkpoint_path)

for i in range(epoch):
    ptr = 0
    for j in range(batches_num):
        inp, out = X_train[ptr:ptr+batch_size], y_train[ptr:ptr+batch_size]
        ptr+=batch_size
        _, cost = sess.run([model.optimize, model.cost],
                                     {data: inp, target: out, dropout: 0.5})
        if j % 1000 == 0:
            print('[Epoch %d] loss at batch %d: %f' % (i, j, cost[0][0]))
    if i % 10 == 0:
        train_err = sess.run(model.error,{data: X_train, target: y_train, dropout: 1.0})
        print('Training error {:3.1f}%'.format(100 * train_err))

if(save):
    print('Saving model...')
    save_path = saver.save(sess, checkpoint_path)

train_err = sess.run(model.error,{data: X_train, target: y_train, dropout: 1.0})
print('Training error {:3.1f}%'.format(100 * train_err))

valid_err = sess.run(model.error,{data: X_valid, target: y_valid, dropout: 1.0})
print('Validation error {:3.1f}%'.format(100 * valid_err))

test_err = sess.run(model.error,{data: X_test, target: y_test, dropout: 1.0})
print('Test error {:3.1f}%'.format(100 * test_err))

predictions = sess.run(model.prediction,{data: X_train, dropout: 1.0})

sess.close()
