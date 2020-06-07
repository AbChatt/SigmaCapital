from model import *

# print(modelSelection('AAPL_Jun_2019_2020.csv', "AAPL"))
# print(modelSelection('AAPL-2.csv', "AAPL"))

def highlights(ratings, rating):
	buy_and_sell = sorted(ratings, key=lambda x: x[0])
	buy = sorted(list(filter(lambda x: x[0] == "BUY", buy_and_sell)), key=lambda x: x[1])
	sell = sorted(list(filter(lambda x: x[0] == "SELL", buy_and_sell)), key=lambda x: x[1])
	our_highlights = (list(reversed(buy[-5:])), list(sell[0:5]))
	if rating == "BUY":
		return [x[3] for x in our_highlights[0]]
	else:
		return [x[3] for x in our_highlights[1]]


print(modelSelection("DIS"))
# print(highlights([("SELL", -1, 1, "GOOGL"), ("SELL", -0.5, 1, "RDSA"), ("BUY", 100, 1, "APPL"), ("BUY", 80, 1, "BP"), ("SELL", -0.3, 1, "XXON"), ("BUY", 70, 1, "MSFT"), ("BUY", 60, 1, "BRK.A"), ("BUY", 50, 1, "JPM"), ("SELL", -0.4, 1, "CMCSA"), ("SELL", -0.2, 1, "PM")], "SELL"))