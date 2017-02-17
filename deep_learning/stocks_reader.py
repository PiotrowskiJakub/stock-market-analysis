import quandl
import numpy as np
from datetime import date

quandl.ApiConfig.api_key = 'XXX'

companies = [
        'WIKI/MSFT',
        'WIKI/AAPL'
        ]
start_date = date(2016, 1, 31)
end_date = date(2017, 2, 13)

data = quandl.get(companies, start_date=start_date, end_date=end_date)

# There is no data for weekends, so end_date - start_date isn't best thing to do here
# Instead take first dimension from API data (days x metrics)
# -1 because there is no prediction for last day
num_days = data.shape[0] - 1
num_stocks = len(companies)

stock_data = [[] for _ in range(num_days)]
output_data = [[] for _ in range(num_days)]
factors_price = np.ndarray(shape=(num_stocks), dtype=np.float32)
factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)

# For each company get first price and volume to normalize data
# Store it for de-normalization
for idx, company in enumerate(companies):
    factors_price[idx] = data[company + ' - Adj. Close'][0]
    factors_volume[idx] = data[company + ' - Volume'][0]

# Use python arrays to collect data and then convert to numpy ndarray
for day_idx in range(num_days):
    for company_idx, company in enumerate(companies):
        stock_data[day_idx].append([data[company + ' - Adj. Close'][day_idx] / factors_price[company_idx]])
        stock_data[day_idx].append([data[company + ' - Volume'][day_idx] / factors_volume[company_idx]])

for day_idx in range(num_days):
    for company_idx, company in enumerate(companies):
        output_data[day_idx].append(data[company + ' - Adj. Close'][day_idx + 1] / factors_price[company_idx])

#TODO as parameter
train_split = int(0.6 * num_days)
valid_split = int(0.2 * num_days) + train_split
test_split = int(0.2 * num_days) + valid_split

stock_data = np.array(stock_data, dtype=np.float32)
output_data = np.array(output_data, dtype=np.float32)

train_data = stock_data[:train_split, :]
valid_data = stock_data[train_split:valid_split, :]
test_data = stock_data[valid_split:test_split, :]

train_output = output_data[:train_split, :]
valid_output = output_data[train_split:valid_split, :]
test_output = output_data[valid_split:test_split, :]

# Model creation
import tensorflow as tf
companies_number = len(companies)
data = tf.placeholder(tf.float32, [None, companies_number * 2, 1]) # [batch_size, companies_number * 2, input_dimension]
target = tf.placeholder(tf.float32, [None, companies_number]) # [batch_size, companies_number]

num_hidden = companies_number * 4
cell = tf.contrib.rnn.LSTMCell(num_hidden)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32) # [batch_size, max_time, cell.output_size]
val = tf.reshape(val, [-1, num_hidden]) # [batch_size * max_time, cell.output_size] - convert to 2D

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.matmul(val, weight) + bias
                      
loss = tf.nn.l2_loss(tf.subtract(prediction, target))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Model execution
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 1
no_of_batches = int(int(len(train_data)) / batch_size)
epoch = 500
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_data[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print("Epoch ",str(i))
    
incorrect = sess.run(error,{data: valid_data, target: valid_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
