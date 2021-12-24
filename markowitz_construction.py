#os.chdir('C:\\Tushar\\Imperial MFE\\Github\\')
import math 
import numpy as np
import yfinance as yf
import pandas as pd
from scipy import optimize as scopti
import statsmodels.api as sm
import scipy.stats
import risk_metrics

def stock_returns(prices):
    ret=(prices.iloc[1:]/prices.iloc[:-1].values)-1
    return ret

def log_returns(prices):
    ret=np.log(prices.iloc[1:]/prices.iloc[:-1].values)
    return ret

def std_returns(returns):
    ret=(returns-returns.mean())/returns.std()
    return ret

def cumulative_returns(returns):
    ret= (1+returns).cumprod()
    return ret
 
def annualised_returns(returns, period_freq):
    return((1+returns).product())**(period_freq/returns.shape[0])-1
    
def annualised_vol(returns, period_freq):
    return returns.std()*(period_freq**0.5)   

def portfolio_returns(weights,returns):
    return np.dot(returns, weights)

def portfolio_vol(weights, covmat):
    return np.dot(weights.T,np.dot(covmat,weights))**0.5 


"""Markowitz Portfolio  
(uses sample expected returns for the analysis which is not a good estimation parameter)"""


def optimal_weights_for_max_sharpe_ratio(risk_free_rate, returns, period_freq, short=False):
    """
    Returns the max_sharpe_ratio that can be achieved given a set of expected returns and a covariance matrix
    """
    er=annualised_returns(returns, period_freq)
    cov=returns.cov()
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    if (short==True):
        bounds = ((-1.0, 1.0),) * n
    else:
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, returns, period_freq,risk_free_rate):
        r=portfolio_returns(weights,returns)
        vol=portfolio_vol(weights, returns.cov())
        annualised_excess_ret=annualised_returns(r-risk_free_rate, period_freq)
        return -1*annualised_excess_ret/vol
        
    weights = scopti.minimize(neg_sharpe_ratio,
                       init_guess,
                       args=(returns, period_freq,risk_free_rate,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return weights.x

"""Defining the Global Minimum Variance (GMV) Portfolio: This portfolio has the minumum variance and 
    is independednt of the expected returns-- hence it is free from the estimation biases in the estimation 
    of expected returns which was a big issue in Markowitz process 
    
    Another way of bypassing expected returns in portfolio calculations is to assume all weights equal.
    This portfolio is called Equally Weighted (EW) prtfolio or Naive Diversification Portfolio
    """

def optimal_weights_for_gmv(returns, period_freq, short=False):
    """
    Returns the max_sharpe_ratio that can be achieved given a set of expected returns and a covariance matrix
    """
    er=annualised_returns(returns, period_freq)
    cov=returns.cov()
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    if (short==True):
        bounds = ((-1.0, 1.0),) * n
    else:
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
            
    weights = scopti.minimize(portfolio_vol,
                       init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return weights.x

"""Plotting Efficient Frontiers and Important Portfolios """

def plot_two_asset_effiecient_frontier(n_points,returns, period_freq):
    annualised_ret=annualised_returns(returns, period_freq)
    if(annualised_ret.shape[0]!=2):
        raise ValueError('plot_two_asset_effiecient_frontier can only plot for 2 assets')
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    ret=[portfolio_returns(w,returns) for w in weights]
    vol=[portfolio_vol(w,returns.cov()) for w in weights]
    ef=pd.DataFrame({'Portfolio Returns': ret,'Portfolio Risk': vol})
    return ef.plot.line(x='Portfolio Risk',y='Portfolio Returns',style='.-')

def minimize_vol(target_return, returns, period_freq, short=False):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    er=annualised_returns(returns, period_freq)
    cov=returns.cov()
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    if (short==True):
        bounds = ((-1.0, 1.0),) * n
    else:
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_returns(weights,returns)
    }
    weights = scopti.minimize(portfolio_vol, init_guess,
                       args=(cov,),
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, returns, period_freq):
    """
    """
    er=annualised_returns(returns, period_freq)
    cov=returns.cov()
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, returns, period_freq) for target_return in target_rs]
    return weights

def plot_multi_asset_effiecient_frontier(n_points, returns, period_freq, risk_free_rate=0,
                                         show_capital_mkt_line=False, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    er=annualised_returns(returns, period_freq)
    cov=returns.cov()
    weights = optimal_weights(n_points, returns, period_freq)
    rets = [portfolio_returns(w,returns, period_freq) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Portfolio Returns": rets, 
        "Portfolio Volatility": vols
    })
    ax=ef.plot.line(x="Portfolio Volatility", y="Portfolio Returns", style='.-',figsize=(12,6))
    if(show_ew):
        n=er.shape[0]
        ew_wts=np.repeat(1/n,n)
        ew_rets=portfolio_returns(ew_wts,returns)
        ew_vol=portfolio_vol(ew_wts, cov)
        ## Add EW
        ax.plot([ew_vol],[ew_rets],color='green',marker="o",markersize=10, label='Equally Weighted (EW) Portfolio')
        
    if(show_gmv):
        n=er.shape[0]
        gmv_wts=optimal_weights_for_gmv(returns, period_freq)
        gmv_rets=portfolio_returns(gmv_wts,returns)
        gmv_vol=portfolio_vol(gmv_wts, cov)
        ## Add EW
        ax.plot([gmv_vol],[gmv_rets],color='midnightblue',marker="o",markersize=10,
                label='Global Minimum Variance (GMV) Portfolio')
        
    if(show_capital_mkt_line):
        ax.set_xlim(left=0)
        msr_wts=optimal_weights_for_max_sharpe_ratio(risk_free_rate, returns, period_freq)
        msr_rets=portfolio_returns(msr_wts,returns)
        msr_vol=portfolio_vol(msr_wts, cov)
        ## Add CML 
        cml_x=[0,msr_vol]
        cml_y=[risk_free_rate,msr_rets]
        ax.plot(cml_x,cml_y,color='red',marker="o",markersize=10, label='Maximum Sharpe Ratio (MSR)/ Markowitz Portfolio')
    plt.legend()
    return ax