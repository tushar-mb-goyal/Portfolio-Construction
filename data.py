import pandas_datareader.data as web
import numpy as np
import yfinance as yf
import pandas as pd
import bs4 as bs
import pickle
import requests


def data_pandas_reader_tenor(ticker_list,tenor,interval,field='Close',**kwargs):
    prices=pd.DataFrame()
    for asset in ticker_list:
        df=yf.download(asset,interval=interval,period=tenor)
        prices[asset]=df[field]
    return prices

def data_pandas_reader_dates(ticker_list,start,end,interval,field='Close',**kwargs):
    #Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    prices=pd.DataFrame()
    for asset in ticker_list:
        df=yf.download(asset,start,end,interval=interval)
        prices[asset]=df[field]
    return prices