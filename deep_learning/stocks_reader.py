#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:25:18 2017

@author: soso

Work in progress file for getting stock data
"""
import quandl
import tensorflow as tf
import numpy as np

quandl.ApiConfig.api_key = 'x'

companies = [
        'WIKI/MSFT',
        'WIKI/AAPL'
        ]
num_stocks = len(companies)
start_date = '2017-02-12'
end_date = '2017-02-13'

data = quandl.get(companies, start_date=start_date, end_date=end_date)

stock_data = []
factors_prices = np.ndarray(shape=(num_stocks), dtype=np.float32)
factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)

"""
data should look like this
[[foo, bar], [foo, bar], ...]
"""
for company in companies:
    delta = float(data[company + ' - Open'] - data[company + ' - Close'])
    volume = float(data[company + ' - Volume'])
    print(delta)