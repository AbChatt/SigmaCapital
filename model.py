import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import asyncio

from pylab import rcParams
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# needed to pip install pmdarima - which also fetched statsmodels
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import API
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.async_support.timeseries import TimeSeries



# Define the buy/sell rating as appropriate
def rating(prediction):
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

def generateCloseModel(data_frame, ticker):

    # split dataset into testing and training sets - This wasn't working, now has been fixed
    limit = int(0.9*data_frame.shape[0])
    test = data_frame[limit:] # last 20% of values
    train = data_frame[:limit] # first 80% of values
    # Plot into graph
    plt.plot(train.groupby(['date']) ['4. close'].mean(), color='blue', label='Train')
    plt.plot(test.groupby(['date']) ['4. close'].mean(), color='red', label='Test')
    plt.legend(loc='best')
    plt.title('Training and Test Data')
    plt.show()

    is_stationary = test_stationarity(train['4. close'])

    #-- SUMMARY OF LINES 118 - 194 -------------
    # If the stock prices are not stationary, we remove the trends and seasonality

    if not is_stationary:
    # taking ln of training and testing closing values
        train_log = np.log(train['4. close'])
        test_log = np.log(test['4. close'])

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
        plt.plot(train.groupby(['date']) ['4. close'].mean(), label='Train')
        plt.plot(test.groupby(['date']) ['4. close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # check performance of model using RMSE as metric

        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))

    else:
        model = auto_arima(train['4. close'], trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train['4. close'])

        # predict values on validation set
        forecast = model.predict(n_periods=len(test))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

        # Graph forecasts vs actual
        plt.plot(train.groupby(['date']) ['4. close'].mean(), label='Train')
        plt.plot(test.groupby(['date']) ['4. close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # check performance of model using RMSE as metric

        rms = np.sqrt(mean_squared_error(test,forecast))

    result = rating(forecast)
    result.append(rms)
    result.append(ticker)
    return tuple(result)


    # print(f" You should {rating(forecast)} this stock")

def HoltExp(data_frame, ticker):

    # split dataset into testing and training sets - This wasn't working, now has been fixed
    limit = int(0.9*data_frame.shape[0])
    test = data_frame[limit:] # last 20% of values
    train = data_frame[:limit] # first 80% of values
    # Plot into graph
    plt.plot(train.groupby(['date']) ['4. close'].mean(), color='blue', label='Train')
    plt.plot(test.groupby(['date']) ['4. close'].mean(), color='red', label='Test')
    plt.legend(loc='best')
    plt.title('Training and Test Data')
    plt.show()

    is_stationary = test_stationarity(train['4. close'])

    #-- SUMMARY OF LINES 118 - 194 -------------
    # If the stock prices are not stationary, we remove the trends and seasonality

    if not is_stationary:
        # taking ln of training and testing closing values
        train_log = np.log(train['4. close'])
        test_log = np.log(test['4. close'])

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

        model = ExponentialSmoothing(np.array(train_log), trend='mul', seasonal=None, damped=True)
        fit = model.fit()

        # predict values on validation set
        forecast = fit.forecast(np.asarray(len(test)))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test_log.index, columns=['Prediction'])
        forecast = np.exp(forecast) # Adjust for non-log values

        # Graph forecasts vs actual
        plt.plot(train.groupby(['date']) ['4. close'].mean(), label='Train')
        plt.plot(test.groupby(['date']) ['4. close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # check performance of model using RMSE as metric

        rms = np.sqrt(mean_squared_error(test_log,np.log(forecast)))

    else:
        model = ExponentialSmoothing(np.array(train['4. close']), trend='mul', seasonal=None, damped=True)
        fit = model.fit()

        # predict values on validation set
        forecast = fit.forecast(len(test))  # n_periods creates errors
        forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])

        # Graph forecasts vs actual
        plt.plot(train.groupby(['date']) ['4. close'].mean(), label='Train')
        plt.plot(test.groupby(['date']) ['4. close'].mean(), label='Test')
        plt.plot(forecast, color='red', label='Prediction')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        # check performance of model using RMSE as metric

        rms = np.sqrt(mean_squared_error(test,forecast))

    result = rating(forecast)
    result.append(rms)
    result.append(ticker)
    return tuple(result)


def goldenDeathCross(data_frame, ticker):

    # split dataset into testing and training sets - This wasn't working, now has been fixed
    limit = int(0.9*data_frame.shape[0])
    test = data_frame[limit:] # last 20% of values
    train = data_frame[:limit] # first 80% of values
    # Plot into graph
    #plt.plot(train.groupby(['date']) ['4. close'].mean(), color='blue', label='Train')
    #plt.plot(test.groupby(['date']) ['4. close'].mean(), color='red', label='Test')
    #plt.legend(loc='best')
    #plt.title('Training and Test Data')
    #plt.show()

    is_stationary = test_stationarity(train['4. close'])

    #-- SUMMARY OF LINES 118 - 194 -------------
    # If the stock prices are not stationary, we remove the trends and seasonality

    # taking ln of training and testing closing values
    train_log = np.log(train['4. close'])
    test_log = np.log(test['4. close'])

    # plotting moving average
    moving_avg_200 = train_log.rolling(200).mean()
    moving_avg_50 = train_log.rolling(50).mean()
    plt.plot(train_log, label='Original')
    plt.plot(moving_avg_200, color='red', label='200 Day Moving Average')
    plt.plot(moving_avg_50, color='green', label='50 Day Moving Average')
    plt.title('Moving Average')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


    # removing trend to make time series stationary
    train_log_moving_avg_diff = train_log - moving_avg_200
    train_log_moving_avg_diff.dropna(inplace=True)  # took average of first 24 values, so rolling mean is not defined for first 23 - drop values
    
    # stabilise mean of time series - required for stationary time series
    train_log_diff = train_log - train_log.shift(1)
    test_stationarity(train_log_diff.dropna())

    #print(test)

    # checking values of moving average curves near split of train and test sets
    #moving_avg_200_value = (np.log((data_frame[limit - 198:limit + 2])['4. close'])).rolling(200).mean()
    #moving_avg_50_value = (np.log((data_frame[limit - 48:limit + 2])['4. close'])).rolling(50).mean()
    
    moving_avg_200.dropna(inplace=True)

    moving_avg_50.dropna(inplace=True)
    #moving_avg_50.drop(moving_avg_50[moving_avg_50['date'] < moving_avg_200['date']].index, axis=1, inplace=True)
    #print(moving_avg_50.index)

    #print(data_frame[:limit + 1].index[-1])
    #boundary = (data_frame[:limit + 1].index[-1]).strftime("%Y-%m-%d")
    #print(boundary)
    #print(type(boundary))

    #print(moving_avg_200.index)
    #print(moving_avg_50.index)
    #print(moving_avg_200[0])
    #moving_avg_200_filtered = moving_avg_200[moving_avg_200.index >= boundary]
    #print(moving_avg_200_filtered)
    #print(moving_avg_200.index)
    #print(moving_avg_50_value)

    if (moving_avg_50[-1] > moving_avg_200[-1]):
        return "BUY"
    else:
        return "SELL"

def modelSelection(data_frame, ticker):
    # inserts an entry into list of warning filters
    warnings.filterwarnings('ignore')
    
    # style of graph being plotted - font style and related preferences (can be changed)
    plt.style.use('dark_background')
    
    # changing default figsize parameter
    rcParams['figure.figsize'] = 10, 6

    #print("Please enter your API key: ")
    #api_key = input()
    
    #ts = TimeSeries(key=api_key, output_format='pandas')
    
    
    #data_frame, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    # pd.read_csv(data)    # modified from example as path was not being picked up
    
    # gets Date column data
    # con = data_frame['date']
    # converts date column to a Python DateTime object
    data_frame['date'] = pd.to_datetime(data_frame.index)
    
    # sets data_frame index to Date column
    data_frame.set_index('date', inplace=True)
    
    # get Year, Month and Day columns
    data_frame['year'] = data_frame.index.year
    data_frame['month'] = data_frame.index.month
    data_frame['day'] = data_frame.index.day
    data_frame = data_frame.sort_index(axis=0 ,ascending=True) # Reverse DataFrame
    indexNames = data_frame[data_frame['year'] < datetime.now().year - 5].index
    data_frame.drop(indexNames , inplace=True)
    # Display a random sample of 5 rows - year, month and day columsn are added
    #print(data_frame.sample(5, random_state=0))
    
    # groups data by date and close - x axis is date, y axis is mean closing price
    temp = data_frame.groupby(['date']) ['4. close'].mean()
    
    # plots line graph of mean closing price vs date - WORKING NOW
    temp.plot(figsize=(15, 5), kind='line', title=f'{ticker} Closing Prices(Monthwise)', fontsize=14)
    plt.show()
    
    # plots bar graph of mean closing price vs month - WORKING NOW
    # new_temp = data_frame.groupby('month')['Close'].mean().plot.bar()
    # plt.title('Average Closing Price By Month')
    # plt.xlabel('Month')
    # plt.ylabel('Stock Price')
    # plt.show()
    
    close = generateCloseModel(data_frame, ticker)
    holt = HoltExp(data_frame, ticker)
    goldenCross = goldenDeathCross(data_frame, ticker)
    return(close, holt, goldenCross)
    #return sorted([close], key=lambda x: x[2])[0]
    #return goldenCross

async def get_data(ticker, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data_frame, _ = await ts.get_daily(symbol=ticker, outputsize='full')
    await ts.close()
    return data_frame




# --- NEXT TASKS ---#
# TODO: Develop other models based on volume, etc
# TODO: Async
# TODO: Present status using InitialState
# TODO: S&P 500 metrics
