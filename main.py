from model import *
import asyncio
from ISStreamer.Streamer import Streamer

# print(modelSelection('AAPL_Jun_2019_2020.csv', "AAPL"))
# print(modelSelection('AAPL-2.csv', "AAPL"))

# list of companies
# tickers = ['AAPL', 'DIS', 'XOM']
# tickers = ['AAPL', 'DIS', 'XOM']
tickers = input("Please enter the tickers of the stocks you want, seperated by a space: ").split()
print(tickers)
def highlights(ratings, rating):
	buy_and_sell = sorted(ratings, key=lambda x: x[0])
	buy = sorted(list(filter(lambda x: x[0] == "BUY", buy_and_sell)), key=lambda x: x[1])
	sell = sorted(list(filter(lambda x: x[0] == "SELL", buy_and_sell)), key=lambda x: x[1])
	our_highlights = (list(reversed(buy[-5:])), list(sell[0:5]))
	if rating == "BUY":
		return [x[3] for x in our_highlights[0]]
	else:
		return [x[3] for x in our_highlights[1]]

# get API key
print("Please enter your API key: ")
api_key = input()

# asynchronously get input
loop = asyncio.get_event_loop()
tasks = [get_data(ticker, api_key) for ticker in tickers]
group1 = asyncio.gather(*tasks)
results = loop.run_until_complete(group1)


# Configure Streaming Bucket

ACCESS_KEY = "ist_ZmIGaAlTd9_qv0JcanHg3XzKA_Y7Y284"
BUCKET_KEY = "3DQY5TQB3CUB"
BUCKET_NAME = "Hack The Northeast"
streamer = Streamer(bucket_name=BUCKET_NAME, bucket_key=BUCKET_KEY, access_key=ACCESS_KEY)

print("And the results are: ")
for i in range(len(results)):
	# print(modelSelection(results[i][0], tickers[i]))
	now = modelSelection(results[i], tickers[i])
	streamer.log(tickers[i], now[0])
	print(now)
	
streamer.flush()


# print(highlights([("SELL", -1, 1, "GOOGL"), ("SELL", -0.5, 1, "RDSA"), ("BUY", 100, 1, "APPL"), ("BUY", 80, 1, "BP"), ("SELL", -0.3, 1, "XXON"), ("BUY", 70, 1, "MSFT"), ("BUY", 60, 1, "BRK.A"), ("BUY", 50, 1, "JPM"), ("SELL", -0.4, 1, "CMCSA"), ("SELL", -0.2, 1, "PM")], "SELL"))