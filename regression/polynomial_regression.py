# POLYNOMIAL REGRESSION

# Importing the libraries
import quandl
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
polynomial_degree = 4
test_size = 0.05
quandl_dataset = "WIKI/FB"

# Importing the dataset
dataset = quandl.get(quandl_dataset)
dataset = dataset.reset_index()

# Converting the dates to an integer number of days since the start of the data
dataset['Date'] = (dataset['Date'] - dataset['Date'].min())  / np.timedelta64(1,'D')
#dataset['Date'] = [date.timetuple().tm_yday for date in dataset['Date']]
#dataset['Date'] = [date.isocalendar()[1] for date in dataset['Date']]
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
split_index = int(math.floor(len(X) * (1 - test_size)))
X_train = X[0:split_index, :]
y_train = y[0:split_index]
X_test = X[split_index:, :]
y_test = y[split_index:]

# Fitting to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = polynomial_degree)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Predicting a new result
transformed_array = poly_reg.fit_transform(X_test)

predictions = regressor.predict(transformed_array)
score = regressor.score(transformed_array, y_test)


# Only for one dimensional matrix
def show_plot(X, y, regressor, poly_reg):
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
    plt.title('Stock market polynomial regression')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
# Visualising the train results
show_plot(X_train, y_train, regressor, poly_reg)

# Visualising the test results
show_plot(X_test, y_test, regressor, poly_reg)
