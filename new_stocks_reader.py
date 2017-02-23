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
# -batch_size because there should be enough data to make predictions
batch_size = 5
num_days = data.shape[0] - batch_size
num_stocks = len(companies)
no_of_batches = int(int(num_days) / batch_size)

stock_data = [[] for _ in range(no_of_batches)]
output_data = [[] for _ in range(no_of_batches)]
factors_price = np.ndarray(shape=(num_stocks), dtype=np.float32)
factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)

# For each company get first price and volume to normalize data
# Store it for de-normalization
for idx, company in enumerate(companies):
    factors_price[idx] = data[company + ' - Adj. Close'][0]
    factors_volume[idx] = data[company + ' - Volume'][0]

# Use python arrays to collect data and then convert to numpy ndarray
ptr = 0
for batch_no in range(no_of_batches):
    for batch_idx in range(batch_size):
        stock_data[batch_no].append([])
        for company_idx, company in enumerate(companies):
            stock_data[batch_no][batch_idx].append([data[company + ' - Adj. Close'][ptr+batch_idx] / factors_price[company_idx],
                                                    data[company + ' - Volume'][ptr+batch_idx] / factors_volume[company_idx]])
    ptr += batch_size

# for day_idx in range(num_days):
    # for company_idx, company in enumerate(companies):
        # output_data[day_idx].append(data[company + ' - Adj. Close'][day_idx + 1] / factors_price[company_idx])

# #TODO as parameter
# train_split = int(0.6 * num_days)
# valid_split = int(0.2 * num_days) + train_split
# test_split = int(0.2 * num_days) + valid_split

# stock_data = np.array(stock_data, dtype=np.float32)
# output_data = np.array(output_data, dtype=np.float32)

# train_data = stock_data[:train_split, :]
# valid_data = stock_data[train_split:valid_split, :]
# test_data = stock_data[valid_split:test_split, :]

# train_output = output_data[:train_split, :]
# valid_output = output_data[train_split:valid_split, :]
# test_output = output_data[valid_split:test_split, :]


