import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from pylab import rcParams 
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
# needed to pip install pmdarima - which also fetched statsmodels
from pmdarima.arima import auto_arima     
from sklearn.metrics import mean_squared_error, mean_absolute_error

# inserts an entry into list of warning filters
warnings.filterwarnings('ignore')   

# style of graph being plotted - font style and related preferences (can be changed)
plt.style.use('dark_background')

# changing default figsize parameter
rcParams['figure.figsize'] = 10, 6

data_frame = pd.read_csv('AAPL_Jun_2019_2020.csv')    # modifed from example as path was not being picked up

#print(data_frame['Date'])

# gets Date column data
con = data_frame['Date']
# converts date column to a Python DateTime object
data_frame['Date'] = pd.to_datetime(data_frame['Date'])

# sets data_frame index to Date column
data_frame.set_index('Date', inplace=True)


#print(data_frame.index)

# get Year, Month and Day columns
data_frame['year'] = data_frame.index.year
data_frame['month'] = data_frame.index.month
data_frame['day'] = data_frame.index.day

# Display a random sample of 5 rows - year, month and day columsn are added
#print(data_frame.sample(5, random_state=0))

# groups data by date and close - x axis is date, y axis is mean closing price
temp = data_frame.groupby(['Date']) ['Close'].mean()

# plots line graph of mean closing price vs date - WORKING NOW
temp.plot(figsize=(15, 5), kind='line', title='Closing Prices(Monthwise)', fontsize=14)
plt.show()
# plots bar graph of mean closing price vs month - WORKING NOW
new_temp = data_frame.groupby('month')['Close'].mean().plot.bar()
plt.show()
# split dataset into testing and training sets - This wasn't working,now has been fixed
test = data_frame[int(0.8*data_frame.size):] # last 20% of values
train = data_frame[:int(0.8*data_frame.size)] # first 80% of values

#TODO: display test and train on the same graph as shown in the medium doc

def test_stationarity(timeseries):
    # Determining rolling statistics
    roll_mean = timeseries.rolling(12).mean()
    roll_std = timeseries.rolling(12).std()

    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()  # using block = False stops graph from showing, but block = True seems to be the same as not including it at all

    # running Augmented Dickey Fuller Test
    print("Results of Dickey Fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    
    # adfuller gives output without labels - need to manually write what values mean using for loop
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
        print(output)

test_stationarity(train['Close'])

# taking ln of training and testing closing values
train_log = np.log(train['Close'])
test_log = np.log(test['Close'])

# plotting moving average
moving_avg = train_log.rolling(24).mean()
plt.plot(train_log)
plt.plot(moving_avg, color='red')
plt.show()

# removing trend to make time series stationary
train_log_moving_avg_diff = train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace=True)  # took average of first 24 values, so rolling mean is not defined for first 23 - drop values
test_stationarity(train_log_moving_avg_diff)

# stabilise mean of time series - required for stationary time series
train_log_diff = train_log - train_log.shift(1)
test_stationarity(train_log_diff.dropna())

# check stationarity of residuals - confirms if there is seasonality
#train_log_decompose = pd.DataFrame(residual)    # residual is not defined
#train_log_decompose['date'] = train_log.index
#train_log_decompose.set_index('date', inplace=True)
#train_log_decompose.dropna(inplace=True)
#test_stationarity(train_log_decompose[0])

# fit auto ARIMA - fit model on the univariate series
model = auto_arima(train_log, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train_log)

# predict values on validation set
#forecast = model.predict(n_periods=len(test))  # n_periods creates errors
#forecast = pd.DataFrame(forecast, index=test_log.index, columns=['Prediction'])

# check performance of model using RMSE as metric
plt.plot(train_log, label='Train')
plt.plot(test_log, label='Test')
#plt.plot(forecast, label='Prediction')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()