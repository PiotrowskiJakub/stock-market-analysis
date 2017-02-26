import quandl
import numpy as np
from datetime import date

quandl.ApiConfig.api_key = 'jPGm5gjF1imaezGU9QMU'

COMPANIES = [
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

def read_data():
    data = quandl.get(COMPANIES, start_date=start_date, end_date=end_date)
    
    # There is no data for weekends, so end_date - start_date isn't best thing to do here
    # Instead take first dimension from API data (days x metrics)
    # -num_forecasts because there is no prediction for 'num_forecasts' last days
    num_forecasts = 5
    num_days = data.shape[0] - num_forecasts
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
            next_prices = next_prices[day_idx + 1:day_idx + 1 + num_forecasts] / factors_price[company_idx]
            change_percentage = (np.max(next_prices)/stock_data[day_idx][company_idx][0]) - 1
            change_percentage = np.clip(change_percentage * 10, -1.0, 1.0)
            output_data[day_idx].append(change_percentage)
    
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
