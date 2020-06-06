from model import *

# print(modelSelection('AAPL_Jun_2019_2020.csv', "AAPL"))

def highlights(ratings):
	buy_and_sell = sorted(ratings, key=lambda x: x[0])
	buy = sorted(list(filter(lambda x: x[0] == "BUY", buy_and_sell)), key=lambda x: x[1])
	sell = sorted(list(filter(lambda x: x[0] == "SELL", buy_and_sell)), key=lambda x: x[1])
	return (list(reversed(buy[-5:])), list(sell[0:5]))

# print(highlights([("SELL", -1, "GOOGL"), ("SELL", -0.5, "RDSA"), ("BUY", 100, "APPL"), ("BUY", 80, "BP"), ("SELL", -0.3, "XXON"), ("BUY", 70, "MSFT"), ("BUY", 60, "BRK.A"), ("BUY", 50, "JPM"), ("SELL", -0.4, "CMCSA"), ("SELL", -0.2, "PM")]))