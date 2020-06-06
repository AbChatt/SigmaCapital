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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# needed to pip install pmdarima - which also fetched statsmodels
from pmdarima.arima import auto_arima     
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Define the buy/sell rating as appropriate
def closeRating(prediction):
    number_of_days = len(prediction) # How many days are in our test set
    initial_price = prediction['Prediction'].iloc[0] # Variable names for these
    final_price = prediction['Prediction'].iloc[-1] # are self explanatory
    model_slope = (prediction['Prediction'].iloc[-1] - initial_price)/number_of_days # Basic linear regression 
    
    if model_slope > 0:
        return ["BUY", model_slope]
    else:
        return ["SELL", model_slope]
    
# Determine if timeseries is stationary
def test_stationarity(timeseries):
    # Determining rolling statistics
    roll_mean = timeseries.rolling(12).mean()
    roll_std = timeseries.rolling(12).std()

    # Plot rolling statistics
    # plt.plot(timeseries, color='blue', label='Original')
    # plt.plot(roll_mean, color='red', label='Rolling Mean')
    # plt.plot(roll_std, color='white', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean and Standard Deviation')
    # plt.show()  # using block = False stops graph from showing, but block = True seems to be the same as not including it at all

    # running Augmented Dickey Fuller Test
    # print("Results of Dickey Fuller test")
    adft = adfuller(timeseries, autolag='AIC')
    
    # adfuller gives output without labels - need to manually write what values mean using for loop
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values
    
    test_stats = output[0] # Test statistics
    critical_values = output[4] # Critical values
    
    if test_stats > critical_values: # If the test statistics are more than the critical values, the null hypothesis passes and it is not stationary
        return False
    else:
        return True # The null hypothesis IS stationary
        
def generateCloseModel(data, ticker):
    
    # inserts an entry into list of warning filters
    warnings.filterwarnings('ignore')   
    
    # style of graph being plotted - font style and related preferences (can be changed)
    plt.style.use('dark_background')
    
    # changing default figsize parameter
    rcParams['figure.figsize'] = 10, 6
    
    data_frame = pd.read_csv(data)    # modified from example as path was not being picked up
    
    # gets Date column data
    con = data_frame['Date']
    # converts date column to a Python DateTime object
    data_frame['Date'] = pd.to_datetime(data_frame['Date'])
    
    # sets data_frame index to Date column
    data_frame.set_index('Date', inplace=True)
    
    # get Year, Month and Day columns
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    
    # Display a random sample of 5 rows - year, month and day columsn are added
    #print(data_frame.sample(5, random_state=0))
    
    # groups data by date and close - x axis is date, y axis is mean closing price
    temp = data_frame.groupby(['Date']) ['Close'].mean()
    
    # plots line graph of mean closing price vs date - WORKING NOW
    temp.plot(figsize=(15, 5), kind='line', title=f'{ticker} Closing Prices(Monthwise)', fontsize=14)
    plt.show()
    
    # plots bar graph of mean closing price vs month - WORKING NOW
    # new_temp = data_frame.groupby('month')['Close'].mean().plot.bar()
    # plt.title('Average Closing Price By Month')
    # plt.xlabel('Month')
    # plt.ylabel('Stock Price')
    # plt.show()
    
    # split dataset into testing and training sets - This wasn't working, now has been fixed
    limit = int(0.8*data_frame.shape[0])
    test = data_frame[limit:] # last 20% of values
    train = data_frame[:limit] # first 80% of values
    # Plot into graph
    plt.plot(train.groupby(['Date']) ['Close'].mean(), color='blue', label='Train')
    plt.plot(test.groupby(['Date']) ['Close'].mean(), color='red', label='Test')
    plt.legend(loc='best')
    plt.title('Training and Test Data')
    plt.show()
    
    is_stationary = test_stationarity(train['Close'])

    #-- SUMMARY OF LINES 118 - 194 -------------
    # If the stock prices are not stationary, we remove the trends and seasonality
    
    if not is_stationary:
    # taking ln of training and testing closing values
        train_log = np.log(train['Close'])
        test_log = np.log(test['Close'])
    
        # plotting moving average
        moving_avg = train_log.rolling(24).mean()
        # plt.plot(train_log, label='Original')
        # plt.plot(moving_avg, color='red', label='Moving Average')
        # plt.title('Moving Average')
        # plt.show()
    
        # removing trend to make time series stationary
        train_log_moving_avg_diff = train_log - moving_avg
        train_log_moving_avg_diff.dropna(inplace=True)  # took average of first 24 values, so rolling mean is not defined for first 23 - drop values
    
    # test_stationarity(train_log_moving_avg_diff)
    
        # stabilise mean of time series - required for stationary time series
        train_log_diff = train_log - train_log.shift(1)
        test_stationarity(train_log_diff.dropna())
    

    # fit auto ARIMA - fit model on the univariate series
    
        model = auto_arima(train_log, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train_log)
    
        # predict values on validation set
        forecast = model.predict(n_periods=len(test))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test_log.index, columns=['Prediction'])
        forecast = np.exp(forecast) # Adjust for non-log values
    
        # Graph forecasts vs actual
        plt.plot(train.groupby(['Date']) ['Close'].mean(), label='Train')
        plt.plot(test.groupby(['Date']) ['Close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
    
        # check performance of model using RMSE as metric

        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))

    else:
        model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train)
        
        # predict values on validation set
        forecast = model.predict(n_periods=len(test))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
        
        # Graph forecasts vs actual
        plt.plot(train.groupby(['Date']) ['Close'].mean(), label='Train')
        plt.plot(test.groupby(['Date']) ['Close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
        # check performance of model using RMSE as metric
     
        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))
        
    result = closeRating(forecast)
    result.append(rms)
        
    return tuple(result)
    
        
    # print(f" You should {rating(forecast)} this stock")

def HoltExp(data, ticker):
    # inserts an entry into list of warning filters
    warnings.filterwarnings('ignore')   
    
    # style of graph being plotted - font style and related preferences (can be changed)
    plt.style.use('dark_background')
    
    # changing default figsize parameter
    rcParams['figure.figsize'] = 10, 6
    
    data_frame = pd.read_csv(data)    # modified from example as path was not being picked up
    
    # gets Date column data
    con = data_frame['Date']
    # converts date column to a Python DateTime object
    data_frame['Date'] = pd.to_datetime(data_frame['Date'])
    
    # sets data_frame index to Date column
    data_frame.set_index('Date', inplace=True)
    
    # get Year, Month and Day columns
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    
    # Display a random sample of 5 rows - year, month and day columsn are added
    #print(data_frame.sample(5, random_state=0))
    
    # groups data by date and close - x axis is date, y axis is mean closing price
    temp = data_frame.groupby(['Date']) ['Close'].mean()
    
    # plots line graph of mean closing price vs date - WORKING NOW
    temp.plot(figsize=(15, 5), kind='line', title=f'{ticker} Closing Prices(Monthwise)', fontsize=14)
    plt.show()
    
    # plots bar graph of mean closing price vs month - WORKING NOW
    # new_temp = data_frame.groupby('month')['Close'].mean().plot.bar()
    # plt.title('Average Closing Price By Month')
    # plt.xlabel('Month')
    # plt.ylabel('Stock Price')
    # plt.show()
    
    # split dataset into testing and training sets - This wasn't working, now has been fixed
    limit = int(0.8*data_frame.shape[0])
    test = data_frame[limit:] # last 20% of values
    train = data_frame[:limit] # first 80% of values
    # Plot into graph
    plt.plot(train.groupby(['Date']) ['Close'].mean(), color='blue', label='Train')
    plt.plot(test.groupby(['Date']) ['Close'].mean(), color='red', label='Test')
    plt.legend(loc='best')
    plt.title('Training and Test Data')
    plt.show()
    
    is_stationary = test_stationarity(train['Close'])
    
    #-- SUMMARY OF LINES 118 - 194 -------------
    # If the stock prices are not stationary, we remove the trends and seasonality
    
    if not is_stationary:
        # taking ln of training and testing closing values
        train_log = np.log(train['Close'])
        test_log = np.log(test['Close'])
        
        # plotting moving average
        moving_avg = train_log.rolling(24).mean()
        # plt.plot(train_log, label='Original')
        # plt.plot(moving_avg, color='red', label='Moving Average')
        # plt.title('Moving Average')
        # plt.show()
        
        # removing trend to make time series stationary
        train_log_moving_avg_diff = train_log - moving_avg
        train_log_moving_avg_diff.dropna(inplace=True)  # took average of first 24 values, so rolling mean is not defined for first 23 - drop values
        
        # test_stationarity(train_log_moving_avg_diff)
        
        # stabilise mean of time series - required for stationary time series
        train_log_diff = train_log - train_log.shift(1)
        test_stationarity(train_log_diff.dropna())
        
        
        # fit Exponential Smoothing - fit model on the univariate series
        
        model = ExponentialSmoothing(np.array(train_log['Close']), trend='mul', seasonal=None, damped=True)
        model.fit(train_log)

        # predict values on validation set
        forecast = model.forecast(np.asarray(len(test)))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test_log.index, columns=['Prediction'])
        forecast = np.exp(forecast) # Adjust for non-log values
        
        # Graph forecasts vs actual
        plt.plot(train.groupby(['Date']) ['Close'].mean(), label='Train')
        plt.plot(test.groupby(['Date']) ['Close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
        # check performance of model using RMSE as metric
        
        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))
        
    else:
        model = ExponentialSmoothing(np.array(train['Close']), trend='mul', seasonal=None, damped=True)
        model.fit(train)

        # predict values on validation set
        forecast = model.forecast(len(test))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
        
        # Graph forecasts vs actual
        plt.plot(train.groupby(['Date']) ['Close'].mean(), label='Train')
        plt.plot(test.groupby(['Date']) ['Close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
        # check performance of model using RMSE as metric
        
        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))
        
    result = HoltRating(forecast)
    result.append(rms)
        
    return tuple(result)
        
def modelSelection(data, ticker):
    close = generateCloseModel(data, ticker)
    return close
    



# --- NEXT TASKS ---#
# TODO: Develop other models based on volume, etc
# TODO: Develop model evaluation
# TODO: Adopt and migrate to stocks API
# TODO: Present status using InitialState
