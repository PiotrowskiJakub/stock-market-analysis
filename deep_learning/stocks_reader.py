import quandl
import numpy as np
from datetime import date

quandl.ApiConfig.api_key = 'XXX'

companies = [
        'WIKI/MSFT',
        'WIKI/AAPL'
        ]
start_date = date(2017, 1, 31)
end_date = date(2017, 2, 13)

data = quandl.get(companies, start_date=start_date, end_date=end_date)

# There is no data for weekends, so end_date - start_date isn't best thing to do here
# Instead take first dimension from API data (days x metrics)
num_days = data.shape[0]
num_stocks = len(companies)

"""ndarray solution
How many metrics per company we want to push. Right now it's 2: price difference and volume
num_metrics = 2
stock_data = np.ndarray(shape=(num_days, num_stocks * num_metrics), dtype=np.float32)
"""

stock_data = [[] for _ in range(num_days)]
factors_price = np.ndarray(shape=(num_stocks), dtype=np.float32)
factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)

# For each company get first price and volume to normalize data
# Store it for de-normalization
for idx, company in enumerate(companies):
    factors_price[idx] = data[company + ' - Adj. Close'][0]
    factors_volume[idx] = data[company + ' - Volume'][0]
    
""" ndarray solution, this is really fucked up.
# For each day create array containing stocks and volumes for each company
for day_idx in range(num_days):
    for company_idx, company in enumerate(companies):
        # Note here because it's not obvious. Why company_idx * num_metrics:
        # fuck this, append would be much easier.
        stock_data[day_idx][company_idx * num_metrics] = (data[company + ' - Close'][day_idx] - data[company + ' - Open'][day_idx]) / factors_price[company_idx]
        stock_data[day_idx][(company_idx * num_metrics) + 1] = (data[company + ' - Volume'][day_idx] / factors_volume[company_idx])
"""

# Use python arrays to collect data and then convert to numpy ndarray
for day_idx in range(num_days):
    for company_idx, company in enumerate(companies):
        stock_data[day_idx].append(data[company + ' - Adj. Close'][day_idx] / factors_price[company_idx])
        stock_data[day_idx].append(data[company + ' - Volume'][day_idx] / factors_volume[company_idx])

#TODO as parameter
train_split = int(0.6 * num_days)
valid_split = int(0.2 * num_days) + train_split
test_split = int(0.2 * num_days) + valid_split

stock_data = np.array(stock_data, dtype=np.float32)

train_data = stock_data[:train_split, :]
valid_data = stock_data[train_split:valid_split, :]
test_data = stock_data[valid_split:test_split, :]
