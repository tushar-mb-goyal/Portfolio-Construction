import datetime as dt
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
import math 
import numpy as np
import yfinance as yf
import pandas as pd
from scipy import optimize as scopti
import bs4 as bs
import pickle
import requests

def portfolio_risk( weights, assets_covar):
    weights=weights.reshape(-1,1)
    return np.dot(weights.T,np.dot(assets_covar,weights))**0.5 

def risk_contribution(weights, assets_covar):
    weights=weights.reshape(-1,1)
    portfolio_vol=portfolio_risk(weights, assets_covar)
    marginal_risk_contribution=(np.dot(assets_covar,weights)/portfolio_vol).reshape(-1,1)
    risk_contribution= np.multiply(weights,marginal_risk_contribution)
    relative_risk_contribution=risk_contribution/portfolio_vol
    return relative_risk_contribution

def risk_budget_objective_error(weights, args):
    assets_covar = args[0]
    assets_risk_budget = args[1]
    weights=weights.reshape(-1,1)
    portfolio_vol = portfolio_risk(weights, assets_covar)
    assets_risk_contribution=risk_contribution(weights, assets_covar)
    assets_risk_target = np.multiply(portfolio_vol, assets_risk_budget)
    error = float(sum(np.square(assets_risk_contribution - assets_risk_target)))
    return error

def risk_budgeted_backtest(risky_portfolio, safe_portfolio, risk_budget): 
    portfolio= risky_portfolio.merge(safe_portfolio,how='left',left_index=True, right_index=True)
    n=portfolio.shape[1]
    init_guess=np.repeat(1/n,n).reshape(-1,1)
    
    assets_risk_budget=risk_budget
    
    assets_covar=portfolio.cov()
    bounds = ((0.0, 1.0),) * n
    
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    long_only = {'type': 'ineq', 'fun': lambda weights: weights}  
    weights=scopti.minimize(fun=risk_budget_objective_error,
                                   x0=init_guess,
                                   args=[assets_covar, assets_risk_budget],
                                   constraints=(weights_sum_to_1,long_only),
                                   bounds=bounds,
                                   options={'disp': False})   
    return weights.x