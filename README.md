# SigmaCapital
 A revolutionary new approach to stock investing through data driven recommendations. SigmaCapital seeks to empower you to make informed financial decisions using the power of data analytics.

# Table of Contents
* [Introduction](https://github.com/AbChatt/Stocks-Forecasting-Bot#introduction)
* [Technologies](https://github.com/AbChatt/Stocks-Forecasting-Bot#technologies)
* [Requirements](https://github.com/AbChatt/Stocks-Forecasting-Bot#requirements)
* [Acknowledgements](https://github.com/AbChatt/Stocks-Forecasting-Bot#acknowledgements)

## Introduction
Investing into the stock market has always beeen a daunting task. For first time buyers, there are many pitfalls to avoid. Which stocks should I invest in? What current market trends do I need to be aware of? When should I sell my shares? These are commmon concerns that can arise when dipping your toes into the stock market. SigmaCapital was conceived as a response to these challenges and seeks to simplify the convoluted world of finance and stock market investing. We use advanced data analytics in order to generate real time predictions about the near future value of a stock, giving you the insights you need in order to make informed decisions on the stock market and maximise your returns. 

## Technologies
The app runs on a Python backend, which samples 5 year historical data from the Alpha Vantage API. This data is fed into a set of models that generate predictions, which are then combined with information from other economic indicators in order to produce a final recommendation of "Buy", "Sell" or "Hold". This recommendation is then streamed to a Initial State Front End.

## Requirements
Python 3.7 is the main requirement. As such, you will also need to pip install the latest versions of alpha-vantage, ISStreamer and pmdarima (which fetches statsmodels, which is also required).

## Acknowledgements
We used the [Alpha Vantage API](https://github.com/RomelTorres/alpha_vantage) in order to get the latest, up-to-date stock data. Furthermore, parts of the project were adapted from [here](https://towardsdatascience.com/performing-a-time-series-analysis-on-the-aapl-stock-index-3655da9612ff).