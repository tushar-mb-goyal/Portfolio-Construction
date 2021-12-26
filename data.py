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

def ticker_info(ticker_list,field=['longName','shortName','industry','sector'],**kwargs):
#    Valid Fields :  ['zip', 'sector', 'fullTimeEmployees', 'longBusinessSummary', 'city', 
#    'phone', 'state', 'country', 'companyOfficers', 'website', 'maxAge', 'address1', 
#    'industry', 'ebitdaMargins', 'profitMargins', 'grossMargins', 'operatingCashflow', 
#    'revenueGrowth', 'operatingMargins', 'ebitda', 'targetLowPrice', 'recommendationKey', 
#    'grossProfits', 'freeCashflow', 'targetMedianPrice', 'currentPrice', 'earningsGrowth', 
#    'currentRatio', 'returnOnAssets', 'numberOfAnalystOpinions', 'targetMeanPrice', 'debtToEquity', 
#    'returnOnEquity', 'targetHighPrice', 'totalCash', 'totalDebt', 'totalRevenue', 'totalCashPerShare', 
#    'financialCurrency', 'revenuePerShare', 'quickRatio', 'recommendationMean', 'exchange', 'shortName', 
#    'longName', 'exchangeTimezoneName', 'exchangeTimezoneShortName', 'isEsgPopulated', 'gmtOffSetMilliseconds', 
#    'quoteType', 'symbol', 'messageBoardId', 'market', 'annualHoldingsTurnover', 'enterpriseToRevenue', 
#    'beta3Year', 'enterpriseToEbitda', '52WeekChange', 'morningStarRiskRating', 'forwardEps', 
#    'revenueQuarterlyGrowth', 'sharesOutstanding', 'fundInceptionDate', 'annualReportExpenseRatio', 
#    'totalAssets', 'bookValue', 'sharesShort', 'sharesPercentSharesOut', 'fundFamily', 'lastFiscalYearEnd', 
#    'heldPercentInstitutions', 'netIncomeToCommon', 'trailingEps', 'lastDividendValue', 'SandP52WeekChange',
#    'priceToBook', 'heldPercentInsiders', 'nextFiscalYearEnd', 'yield', 'mostRecentQuarter', 'shortRatio',
#    'sharesShortPreviousMonthDate', 'floatShares', 'beta', 'enterpriseValue', 'priceHint', 
#    'threeYearAverageReturn', 'lastSplitDate', 'lastSplitFactor', 'legalType', 'lastDividendDate',
#    'morningStarOverallRating', 'earningsQuarterlyGrowth', 'priceToSalesTrailing12Months',
#    'dateShortInterest', 'pegRatio', 'ytdReturn', 'forwardPE', 'lastCapGain', 'shortPercentOfFloat', 
#    'sharesShortPriorMonth', 'impliedSharesOutstanding', 'category', 'fiveYearAverageReturn', 'previousClose', 
#    'regularMarketOpen', 'twoHundredDayAverage', 'trailingAnnualDividendYield', 'payoutRatio', 'volume24Hr',
#    'regularMarketDayHigh', 'navPrice', 'averageDailyVolume10Day', 'regularMarketPreviousClose', 'fiftyDayAverage',
#    'trailingAnnualDividendRate', 'open', 'toCurrency', 'averageVolume10days', 'expireDate', 'algorithm', 
#    'dividendRate', 'exDividendDate', 'circulatingSupply', 'startDate', 'regularMarketDayLow', 'currency',
#    'trailingPE', 'regularMarketVolume', 'lastMarket', 'maxSupply', 'openInterest', 'marketCap',
#    'volumeAllCurrencies', 'strikePrice', 'averageVolume', 'dayLow', 'ask', 'askSize', 'volume', 
#    'fiftyTwoWeekHigh', 'fromCurrency', 'fiveYearAvgDividendYield', 'fiftyTwoWeekLow', 'bid',
#    'tradeable', 'dividendYield', 'bidSize', 'dayHigh', 'regularMarketPrice', 'logo_url']
    information=pd.DataFrame(columns=field,index=ticker_list)
    for asset in ticker_list:
        ticker_info = yf.Ticker(asset)
        for attrib in field:
            try:
                information[attrib].loc[asset]=ticker_info.info[attrib]
            except: 
                continue
    return information