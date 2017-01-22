# Simple Linear Regression - estimation of a stock trend

# Importing the libraries
import quandl
import numpy as np
import matplotlib.pyplot as plt

# Parameters
analysis_days_number = 180
quandl_dataset = "WIKI/FB"

# Importing the dataset
dataset = quandl.get(quandl_dataset)
dataset = dataset.reset_index()

# Converting the dates to an integer number of days since the start of the data
dataset['Date'] = (dataset['Date'] - dataset['Date'].min())  / np.timedelta64(1,'D')
X = dataset.iloc[-analysis_days_number:, 0:1].values
y = dataset.iloc[-analysis_days_number:, 4].values

# Fitting to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Getting model coefficients
coef = regressor.coef_

def show_plot(X, y, regressor):
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Stock market polynomial regression')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
# Visualising the Linear Regression model
show_plot(X, y, regressor)
