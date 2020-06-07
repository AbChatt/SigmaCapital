# Changelog
Changed plot style to dark_background
Fixed closing prices graph - now working
Fixed mean closing price vs month - now working
Fixed train and test sets
Renamed CSV

V2 (Lots of changes)

Had some mistakes in the training and test sets - now completely working
Created a graph to show training vs. test set, so TODO has been removed
Added title to moving average graph
Added graph labels to moving average and several others
Added prediction code back, added prediction line to graph
Fixed axis for prediction model, so now it fits in with the overall graph

V 0.7

Added Root Mean Squared code (blueprint, will be modified later with more complex logic)
Added BUY/SELL rating functionality 
General code cleanup/commenting
Added next several TODOs

V 0.8

Added conditional stationary checks
Commented-out an incessant number of graphs and prints
Cleaned some code
Reorganized some code

V 0.9

Changed prediction line color on graph
Added ticker data to graphs
Encapsulated model generation into one function
Cleaned up code

V 1.0

Created modelSelection to deal with multiple hypothetical models
Created main.py
Added model slope to function output

V 1.0.1

Created a specific rating function for the close price model
Attempt at implementation of Holt’s exponential damping algorithm - HAS BUGS

V 1.1

Implemented Holt’s exponential damping algorithm
Added Holt’s model to the model evaluation function

V 1.2

Model.py
Implemented model selection logic
Added ticker detail to rating

Main.py
Implemented Top 5 / Bottom 5 logic

V 1.3

Model.py

Officially migrated to Alpha Vantage Stocks API
Expanded historical data to 5 years
Major performance optimizations
Changed training and test sets
Added todos

Main.py

Added functionality to the Top 5/ Bottom 5 Logic

V 1.4

Model.py
Added asynchronous data retrieval
Added Golden Cross / Death Cross metric calculation

Main.py
Updated to support calling API asynchronously
Encapsulated API key

V 1.5

Model.py
Added more complex decision logic for Golden/Death cross
Added a third decision option - HOLD. 

Main.py

Added Dashboard support


# SigmaCapital
 A revolutionary new approach to stock investing through data driven recommendations. SigmaCapital seeks to empower you to make informed financial decisions using the power of data analytics.

# Table of Contents
* [Introduction](https://github.com/AbChatt/SigmaCapital#introduction)
* [Technologies](https://github.com/AbChatt/SigmaCapital#technologies)
* [Requirements](https://github.com/AbChatt/SigmaCapital#requirements)
* [Status](https://github.com/AbChatt/SigmaCapital#status)
* [Acknowledgements](https://github.com/AbChatt/SigmaCapital#acknowledgements)

## Introduction
Investing into the stock market has always beeen a daunting task. For first time buyers, there are many pitfalls to avoid. Which stocks should I invest in? What current market trends do I need to be aware of? When should I sell my shares? These are common concerns that can arise when dipping your toes into the stock market. SigmaCapital was conceived as a response to these challenges and seeks to simplify the convoluted world of finance and stock market investing. We use advanced data analytics in order to generate real time predictions about the near future value of a stock, giving you the insights you need in order to make informed decisions on the stock market and maximise your returns. 

## Technologies
The app runs on a Python backend, which samples 5 year historical data from the Alpha Vantage API. This data is fed into a set of models that generate predictions, which are then combined with information from other economic indicators in order to produce a final recommendation of "Buy", "Sell" or "Hold". This recommendation is then streamed to a Initial State Front End.

## Requirements
Python 3.7 is the main requirement. As such, you will also need to pip install the latest versions of alpha-vantage, ISStreamer and pmdarima (which fetches statsmodels, which is also required).

## Status
As of 7th June 2020, the project status is still active. For future improvements, we want to add more sophisticated models and indicators to our repertoire so that we can further optimise the accuracy of predictions. We also want to add support for indices such as the S&P500 and FTSE100 and generate sector specific predictions. We are also exploring plans to integrate our product into banking accounts, so we can generate personalised predictions for our customers. Our vision is to create a comprehensive platform that encompasses stocks, bonds, ETFs, foreign exchange and cryptocurrencies. This is just the beginning.

## Acknowledgements
We used the [Alpha Vantage API](https://github.com/RomelTorres/alpha_vantage) in order to get the latest, up-to-date stock data. Furthermore, parts of the project were adapted from [here](https://towardsdatascience.com/performing-a-time-series-analysis-on-the-aapl-stock-index-3655da9612ff).