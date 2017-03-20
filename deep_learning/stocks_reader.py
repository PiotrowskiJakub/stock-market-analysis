import quandl
import os
import math
import numpy as np
from datetime import date
from six.moves import cPickle as pickle
import pandas as pd
import pandas_datareader as pdr

quandl.ApiConfig.api_key = 'jPGm5gjF1imaezGU9QMU'

FORECASTS_NUM = 10   # Number of days to forecast
CHANGE_THRESHOLD_BOUNDARIES = [0.05, 0.03] # Price change boundaries

start_date = date(2010, 1, 1)
end_date = date(2017, 3, 15)

COMPANIES = ["^IXIC", "AMAT", "AAPL", "QSII", "CAMP", "IDTI", "LRCX", "MGRC", "MENT", "JKHY", "ADBE", "CERN", "CY", "FISV", "LLTC", "MSFT", "SIGM", "TECD", "PLAB", "MXIM", "CRUS", "DGII", "SYMC", "CSCO", "XLNX", "PRGS", "QCOM", "ZBRA", "EFII", "KOPN", "SPNS", "SNPS", "CREE", "INTU", "MCHP", "PRKR", "SANM", "UTEK", "DSPG", "MIND", "SSYS", "VECO", "BRKS", "CTXS", "HLIT", "IVAC", "KFRC", "NATI", "NTAP", "RSYS", "RCII", "ANSS", "CHKP", "CSGS", "KVHI", "PEGA", "SEAC", "SYKE", "TTEC", "VSAT", "YHOO", "OSIS", "POWI", "RMBS", "RNWK", "SYNT", "TTWO", "AMKR", "CTSH", "MANH", "MSTR", "ULTI", "VRSN", "EPAY", "BRCD", "EGAN", "EXTR", "FFIV", "FNSR", "HSII", "IMMR", "INAP", "JCOM", "NTCT", "NVDA", "PCTI", "PRFT", "QUIK", "ACLS", "CCMP", "HSTM", "ISSC", "LPSN", "MRVL", "SLAB", "SPRT", "TTMI", "MOSY", "OMCL", "PDFS", "CPSI", "STX", "SYNA", "VRNT", "CALD", "FORM", "BLKB", "INTX", "MPWR", "UCTT", "BIDU", "SPWR", "CVLT", "FSLR", "GUID", "IPGP", "SNCR", "CAVM", "ENOC", "GLUU", "GSIT", "TYPE", "RBCN", "SMCI", "VRTU", "ERII", "AVGO", "FTNT", "MDSO", "VRSK"]
companies_count = len(COMPANIES)
# COMPANIES = ['WIKI/' + company for company in COMPANIES_SYMBOLS]

def save_pickle(data, filename):
    print('Pickling %s.' % filename)
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', filename, ':', e)
    return data

def load_pickle(filename):
    if os.path.exists(filename):
        print('Loading %s.' % filename)
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print('Unable to process data from', filename, ':', e)
            raise
    else:
        return None

def change_to_vector(change_percentage):
    """Creates a vector that shows price changes.
    [1 0 0 0 0] - the price fell by more than 5%
    [0 1 0 0 0] - the price fell by more than 3% but less than 5%
    [0 0 1 0 0] - the price has not changed significantly
    [0 0 0 1 0] - the price rose by more than 3% but less than 5%
    [0 0 0 0 1] - the price rose by more than 5%
    """
    vector_length = len(CHANGE_THRESHOLD_BOUNDARIES) * 2 + 1
    change_vector = ([0] * vector_length)

    for idx, threshold in enumerate(CHANGE_THRESHOLD_BOUNDARIES):
        if change_percentage < -threshold:
            change_vector[idx] = 1
            return change_vector
        elif change_percentage > threshold:
            change_vector[vector_length - 1 - idx] = 1
            return change_vector

    change_vector[len(CHANGE_THRESHOLD_BOUNDARIES)] = 1
    return change_vector

def read_data():
    data_filename = './data.pickle'
    data = load_pickle(data_filename)
    if(data is None):
        # data = save_pickle(quandl.get(COMPANIES, start_date=start_date, end_date=end_date), data_filename)
        data = save_pickle(pdr.data.DataReader(COMPANIES, 'yahoo', start_date, end_date), data_filename)

    adj_closes = pd.DataFrame(data.ix['Adj Close'])
    volumes = pd.DataFrame(data.ix['Volume'])
    # There is no data for weekends, so end_date - start_date isn't best thing to do here
    # Instead take first dimension from API data (days x metrics)
    # -FORECASTS_NUM because there is no prediction for FORECASTS_NUM last days
    num_days = data.shape[1] - FORECASTS_NUM
    num_stocks = len(COMPANIES)

    stock_data = []
    output_data = []
    factors_price = np.ndarray(shape=(num_stocks), dtype=np.float32)
    factors_volume = np.ndarray(shape=(num_stocks), dtype=np.float32)

    # For each company get first price and volume to normalize data
    # Store it for de-normalization
    for idx, company in enumerate(COMPANIES):
        adj_close = adj_closes[company]
        volume = volumes[company]

        if(adj_close.empty or volume.empty):
            print("Warning! Empty data for " + company)
        else:
            factors_price[idx] = adj_close[0]
            factors_volume[idx] = volume[0]

    # Use python arrays to collect data and then convert to numpy ndarray
    for company_idx, company in enumerate(COMPANIES):
        if company_idx == 0:    # Drop NASDAQ index
            continue
        for day_idx in range(num_days):
            price = adj_closes[company][day_idx] / factors_price[company_idx]
            volume = volumes[company][day_idx] / factors_volume[company_idx]
            if(math.isnan(price) or math.isnan(volume)):
                print('Warning! NaN values found for company %s and day %s' % (company, day_idx))
            stock_data.append([price, volume])

    for company_idx, company in enumerate(COMPANIES):
        if company_idx == 0:    # Drop NASDAQ index
            continue
        for day_idx in range(num_days):
            next_prices = adj_closes[company].values
            next_prices = next_prices[day_idx + 1:day_idx + 1 + FORECASTS_NUM] / factors_price[company_idx]
            current_price = adj_closes[company].values[day_idx] / factors_price[company_idx]
            change_percentage = (np.max(next_prices)/current_price) - 1
            output_data.append(change_to_vector(change_percentage))

    #TODO as parameter
    train_split = int(0.6 * num_days * companies_count)
    valid_split = int(0.2 * num_days * companies_count) + train_split
    test_split = int(0.2 * num_days * companies_count) + valid_split

    stock_data = np.array(stock_data, dtype=np.float32)
    output_data = np.array(output_data, dtype=np.float32)
    
    stock_data = stock_data.reshape(stock_data.shape[0], 1, stock_data.shape[1])
    output_data = output_data.reshape(output_data.shape[0], 1, output_data.shape[1])

    train_data = stock_data[:train_split, :]
    valid_data = stock_data[train_split:valid_split, :]
    test_data = stock_data[valid_split:test_split, :]

    train_output = output_data[:train_split, :]
    valid_output = output_data[train_split:valid_split, :]
    test_output = output_data[valid_split:test_split, :]

    return train_data, train_output, valid_data, valid_output, test_data, test_output
