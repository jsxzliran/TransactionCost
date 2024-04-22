import numpy as np
import math
import yfinance as yf
from datetime import datetime
import pandas as pd

class EqualWeight:
    def __init__(self, ticker,  tickers, gamma, manage_fee, start_date, end_date):
        self.ticker = ticker
        self.tickers = tickers
        self.gamma = gamma
        self.managefee = manage_fee
        self.start = start_date
        self.end = end_date
        self.n = len(self.tickers)
    
    def portfolioReturn(self):
        data = yf.Ticker(self.ticker).history(period='1d', start=self.start, end=self.end)['Adj Close']
        data = data.sort_index(axis=1, level=1, key=lambda x: [self.tickers.index(t) for t in x])
        portfolio_return = data.pct_change()[1:]
        return portfolio_return
    
    def assetsReturn(self):
        data = yf.download(self.tickers, start=self.start, end=self.end, progress=False)["Adj Close"]
        data = data.sort_index(axis=1, level=1, key=lambda x: [self.tickers.index(t) for t in x])
        assets_return = data.pct_change()[1:]
        return assets_return
        
    def assetsReturn_month(self):
        data = yf.download(self.tickers, start=self.start, end=self.end, progress=False, interval="1mo")["Adj Close"]
        data = data.sort_index(axis=1, level=1, key=lambda x: [self.tickers.index(t) for t in x])
        assets_return = data.pct_change()[1:]
        return assets_return
    
    def covariance(self):
        return np.array(self.assetsReturn().cov()*252)
    
    def strategy(self):
        return 1/self.n*np.ones([self.n,1])
    
    def mu(self):
        return self.gamma*np.matmul(self.covariance(),self.strategy())
    
        
        
    




