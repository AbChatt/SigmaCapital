# SigmaCapital
 A revolutionary new approach to stock investing through data driven recommendations. SigmaCapital seeks to empower you to make informed financial decisions using the power of data analytics. Winner of "Most Viable Startup" at Hack The NorthEast 2020.

# Table of Contents
* [Introduction](https://github.com/AbChatt/SigmaCapital#introduction)
* [Technologies](https://github.com/AbChatt/SigmaCapital#technologies)
* [Requirements](https://github.com/AbChatt/SigmaCapital#requirements)
* [Status](https://github.com/AbChatt/SigmaCapital#status)
* [Acknowledgements](https://github.com/AbChatt/SigmaCapital#acknowledgements)

## Introduction
Investing into the stock market has always beeen a daunting task. For first time buyers, there are many pitfalls to avoid. Which stocks should I invest in? What current market trends do I need to be aware of? When should I sell my shares? These are common concerns that can arise when dipping your toes into the stock market. SigmaCapital was conceived as a response to these challenges and seeks to simplify the convoluted world of finance and stock market investing. We use advanced data analytics in order to generate real time predictions about the near future value of a stock, giving you the insights you need in order to make informed decisions on the stock market and maximise your returns. 

We've made a video about it. Check it out [here](https://youtu.be/NdpAy7NElLA)!

## Technologies
The app runs on a Python backend, which samples 5 year historical data from the Alpha Vantage API. This data is fed into a set of models that generate predictions, which are then combined with information from other economic indicators in order to produce a final recommendation of "Buy", "Sell" or "Hold". This recommendation is then streamed to a Initial State Front End.

## Requirements
Python 3.7 is the main requirement. As such, you will also need to pip install the latest versions of alpha-vantage, ISStreamer and pmdarima (which fetches statsmodels, which is also required).

## Status
As of 7th June 2020, the project status is still active. For future improvements, we want to add more sophisticated models and indicators to our repertoire so that we can further optimise the accuracy of predictions. We also want to add support for indices such as the S&P500 and FTSE100 and generate sector specific predictions. We are also exploring plans to integrate our product into banking accounts, so we can generate personalised predictions for our customers. Our vision is to create a comprehensive platform that encompasses stocks, bonds, ETFs, foreign exchange and cryptocurrencies. This is just the beginning.

## Acknowledgements
We used the [Alpha Vantage API](https://github.com/RomelTorres/alpha_vantage) in order to get the latest, up-to-date stock data. Furthermore, parts of the project were adapted from [here](https://towardsdatascience.com/performing-a-time-series-analysis-on-the-aapl-stock-index-3655da9612ff).