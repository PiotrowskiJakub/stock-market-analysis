import quandl
import numpy as np
from datetime import date

quandl.ApiConfig.api_key = 'jPGm5gjF1imaezGU9QMU'

companies = [
        'WIKI/MSFT',
        'WIKI/AAPL',
        'WIKI/FB',
        'WIKI/AXP',
        'WIKI/BA',
        'WIKI/CAT',
        'WIKI/CSCO',
        'WIKI/CVX',
        'WIKI/DD',
        'WIKI/XOM',
        'WIKI/GE',
        'WIKI/GS',
        'WIKI/HD',
        'WIKI/IBM',
        'WIKI/INTC',
        'WIKI/JNJ',
        'WIKI/KO',
        'WIKI/JPM'
        ]
start_date = date(2017, 2, 1)
end_date = date(2017, 2, 19)

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
        stock_data[day_idx].append([data[company + ' - Adj. Close'][day_idx] / factors_price[company_idx],
                                   data[company + ' - Volume'][day_idx] / factors_volume[company_idx]])

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
from model import StocksPredictorModel
companies_number = len(companies)

data = tf.placeholder(tf.float32, [None, companies_number, 2])
target = tf.placeholder(tf.float32, [None, companies_number])
dropout = tf.placeholder(tf.float32)

model = StocksPredictorModel(data, target, dropout, 0.001)

# Model execution
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 5
no_of_batches = int(int(len(train_data)) / batch_size)
epoch = 100000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_data[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        _, cost, predictions = sess.run([model.optimize, model.cost, model.prediction],
                                     {data: inp, target: out, dropout: 1.0})
    acc = model.accuracy(predictions, out)
    print('Minibatch loss at step %d: %f' % (i, cost))
    print('Minibatch accuracy: %.1f%%' % acc)
 
valid_acc = model.accuracy(sess.run(model.prediction,{data: valid_data, dropout: 1.0}), valid_output)
print('Validation accuracy: %.1f%%' % valid_acc)

test_acc = model.accuracy(sess.run(model.prediction,{data: test_data, dropout: 1.0}), test_output)
print('Test accuracy: %.1f%%' % test_acc)

sess.close()