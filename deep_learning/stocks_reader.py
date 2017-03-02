import quandl
import numpy as np
from datetime import date

quandl.ApiConfig.api_key = 'jPGm5gjF1imaezGU9QMU'

FORECASTS_NUM = 5   # Number of days to forecast
CHANGE_THRESHOLD = 0.3

start_date = date(2016, 9, 1)
end_date = date(2017, 2, 19)

COMPANIES = ['WIKI/MSFT', 'WIKI/AAPL']
'''
COMPANIES = [
        'WIKI/MSFT',
        'WIKI/AAPL',
        'WIKI/FB',
        'WIKI/AXP',
        'WIKI/CAT',
        'WIKI/CSCO',
        'WIKI/GE',
        'WIKI/GS',
        'WIKI/HD',
        'WIKI/IBM',
        'WIKI/INTC',
        'WIKI/JNJ',
        'WIKI/KO',
        'WIKI/JPM'
        ]
'''

    
def change_to_vector(change_percentage):
    """Creates a vector that shows price changes.
    [1 0 0] - the price fell by more than 3%
    [0 1 0] - the price has not changed significantly
    [0 0 1] - the price rose by more than 3%
    """
    change_vector = ([0] * 3)
    i = 1
    if change_percentage < -CHANGE_THRESHOLD:
        i = 0
    elif change_percentage > CHANGE_THRESHOLD:
        i = 2
    change_vector[i] = 1
    return change_vector

def read_data():
    data = quandl.get(COMPANIES, start_date=start_date, end_date=end_date)
    
    # There is no data for weekends, so end_date - start_date isn't best thing to do here
    # Instead take first dimension from API data (days x metrics)
    # -FORECASTS_NUM because there is no prediction for FORECASTS_NUM last days
    num_days = data.shape[0] - FORECASTS_NUM
    num_stocks = len(COMPANIES)
    
    stock_data = [[] for _ in range(num_days)]
    output_data = [[] for _ in range(num_days)]
    factors_price = np.ndarray(shape=(num_stocks), dtype=np.float32)
    factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)
    
    # For each company get first price and volume to normalize data
    # Store it for de-normalization
    for idx, company in enumerate(COMPANIES):
        factors_price[idx] = data[company + ' - Adj. Close'][0]
        factors_volume[idx] = data[company + ' - Volume'][0]
    
    # Use python arrays to collect data and then convert to numpy ndarray
    for day_idx in range(num_days):
        for company_idx, company in enumerate(COMPANIES):
            stock_data[day_idx].append([data[company + ' - Adj. Close'][day_idx] / factors_price[company_idx],
                                       data[company + ' - Volume'][day_idx] / factors_volume[company_idx]])
    
    for day_idx in range(num_days):
        for company_idx, company in enumerate(COMPANIES):
            next_prices = data[company + ' - Adj. Close'].values
            next_prices = next_prices[day_idx + 1:day_idx + 1 + FORECASTS_NUM] / factors_price[company_idx]
            change_percentage = (np.max(next_prices)/stock_data[day_idx][company_idx][0]) - 1
            change_percentage = np.clip(change_percentage * 10, -1.0, 1.0)
            output_data[day_idx].append(change_to_vector(change_percentage))
    
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
    
    return train_data, train_output, valid_data, valid_output, test_data, test_output
