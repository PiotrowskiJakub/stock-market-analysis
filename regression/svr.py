# SVR

# Importing the libraries
import quandl
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
test_size = 0.2
quandl_dataset = "WIKI/FB"

# Importing the dataset
dataset = quandl.get(quandl_dataset)
dataset = dataset.reset_index()

# Converting the dates to an integer number of days since the start of the data
dataset['Date'] = (dataset['Date'] - dataset['Date'].min())  / np.timedelta64(1,'D')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
split_index = int(math.floor(len(X) * (1 - test_size)))
X_train = X[0:split_index, :]
y_train = y[0:split_index]
X_test = X[split_index:, :]
y_test = y[split_index:]

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting a new result
predictions = sc_y.inverse_transform(regressor.predict(X_test))
score = regressor.score(X_test, y_test)

# Only for one dimensional matrix
def show_plot(X, y, regressor):
    plt.scatter(X, y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Stock market SVR')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
# Visualising the train results
show_plot(X_train, y_train, regressor)

# Visualising the test results
show_plot(X_test, y_test, regressor)
