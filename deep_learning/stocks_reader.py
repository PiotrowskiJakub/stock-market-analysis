import quandl
import os
import math
import numpy as np
from datetime import date
from six.moves import cPickle as pickle

quandl.ApiConfig.api_key = 'jPGm5gjF1imaezGU9QMU'

FORECASTS_NUM = 10   # Number of days to forecast
CHANGE_THRESHOLD_BOUNDARIES = [0.05, 0.03] # Price change boundaries

start_date = date(2010, 1, 1)
end_date = date(2017, 3, 10)

COMPANIES_SYMBOLS = ["AMAT", "AAPL", "QSII", "CAMP", "IDTI", "LRCX", "MGRC", "MENT", "JKHY", "ADBE", "CERN", "CY", "FISV", "LLTC", "MSFT", "SIGM", "TECD", "PLAB", "MXIM", "CRUS", "DGII", "SYMC", "CSCO", "XLNX", "PRGS", "QCOM", "ZBRA", "EFII", "KOPN", "SPNS", "SNPS", "AVID", "CREE", "INTU", "MCHP", "PRKR", "SANM", "UTEK", "DSPG", "MIND", "SSYS", "VECO", "BRKS", "CTXS", "HLIT", "IVAC", "KFRC", "NATI", "NTAP", "RSYS", "RCII", "ANSS", "CHKP", "CSGS", "KVHI", "PEGA", "SEAC", "SYKE", "TTEC", "VSAT", "YHOO", "OSIS", "POWI", "RMBS", "RNWK", "SYNT", "TTWO", "AMKR", "CTSH", "MANH", "MSTR", "ULTI", "VRSN", "EPAY", "BRCD", "EGAN", "EXTR", "FFIV", "FNSR", "HSII", "IMMR", "INAP", "JCOM", "NTCT", "NVDA", "PCTI", "PRFT", "QUIK", "ACLS", "CCMP", "HSTM", "ISSC", "LPSN", "MRVL", "SLAB", "SPRT", "TTMI", "MOSY", "OMCL", "PDFS", "CPSI", "STX", "SYNA", "VRNT", "CALD", "FORM", "BLKB", "INTX", "MPWR", "UCTT", "BIDU", "SPWR", "CVLT", "FSLR", "GUID", "IPGP", "SNCR", "CAVM", "ENOC", "GLUU", "GSIT", "TYPE", "RBCN", "SMCI", "VRTU", "ERII", "AVGO", "FTNT", "MDSO", "VRSK"]
COMPANIES = ['WIKI/' + company for company in COMPANIES_SYMBOLS]

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
        data = save_pickle(quandl.get(COMPANIES, start_date=start_date, end_date=end_date), data_filename)

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
        adj_close = data.filter(regex=company + ' - Adj. Close')
        volume = data.filter(regex=company + ' - Volume')

        if(adj_close.empty or volume.empty):
            print("Warning! Empty data for " + company)
        else:
            factors_price[idx] = adj_close.iloc[0][0]
            factors_volume[idx] = volume.iloc[0][0]

    # Use python arrays to collect data and then convert to numpy ndarray
    for day_idx in range(num_days):
        for company_idx, company in enumerate(COMPANIES):
            price = data.filter(regex=company + ' - Adj. Close').iloc[day_idx][0] / factors_price[company_idx]
            volume = data.filter(regex=company + ' - Volume').iloc[day_idx][0] / factors_volume[company_idx]
            if(math.isnan(price) or math.isnan(volume)):
                print('Warning! NaN values found for company ' + company)
            stock_data[day_idx].append([price, volume])

    for day_idx in range(num_days):
        for company_idx, company in enumerate(COMPANIES):
            next_prices = data.filter(regex=company + ' - Adj. Close').values
            next_prices = next_prices[day_idx + 1:day_idx + 1 + FORECASTS_NUM] / factors_price[company_idx]
            change_percentage = (np.max(next_prices)/stock_data[day_idx][company_idx][0]) - 1
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
