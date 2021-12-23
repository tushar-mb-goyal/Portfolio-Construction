import math 
import numpy as np
import yfinance as yf
import pandas as pd
from scipy import optimize as scopti
import statsmodels.api as sm
import scipy.stats

def cumulative_returns(returns):
    ret= (1+returns).cumprod()
    return ret

def annualised_returns(returns, period_freq):
    return((1+returns).product())**(period_freq/returns.shape[0])-1
    
def annualised_vol(returns, period_freq):
    return returns.std()*(period_freq**0.5)   

def portfolio_returns(weights,returns):
    return np.dot(weights.T,returns)

def portfolio_vol(weights, covmat):
    return np.dot(weights.T,np.dot(covmat,weights))**0.5 
    
def sharpe_ratio(returns, period_freq,risk_free_rate=0.03):
    rf_per_period=(1+risk_free_rate)**(1/period_freq)-1
    excess_ret=returns-rf_per_period
    annualised_excess_ret=annualised_returns(excess_ret, period_freq)
    return annualised_excess_ret/annualised_vol(returns, period_freq)

def drawdown(returns):
    wealth_idx=cumulative_returns(returns)
    peaks_idx=wealth_idx.cummax()
    drawdown=(wealth_idx-peaks_idx)/peaks_idx
    return drawdown

def skewness(returns):
    '''It can be done directly by calling scipy.stats.skew()'''
    demeaned_ret=returns-returns.mean()
    vol=returns.std()
    exp=(demeaned_ret**3).mean()
    return exp/vol**3

def excess_kurtosis(returns):
    '''It can be done directly by calling scipy.stats.kurtosis() and calculates excess kurtosis i.e kurtosis wrt 3'''
    demeaned_ret=returns-returns.mean()
    vol=returns.std()
    exp=(demeaned_ret**4).mean()
    return (exp/vol**4)-3

def jarque_bera_normality(returns,level=0.01):
    '''Applies Jarque Bera Normality Test to test the normality at 1% level by default
        Returns True if the null hypothesis of normality is rejected else False;
        Refer to  https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test for  more details'''
    test_statistic, p_value=scipy.stats.jarque_bera(returns)
    return p_value>level

def semi_deviation(returns):
    '''Semi Deviation is calculation of volatility of returns by only considering
        those ones which are less than the mean returns unlike our standard deviation'''
    is_negative=returns<0
    return returns[is_negative].std(ddof=0)

def var_historic(returns, level=5):
    '''returns the historic var (also called H-VaR) from the sample returns with a confidence interval of 100-level% '''
    '''Since this VaR Method uses historical sample data so it has very high Sample risk and very low Model Risk'''
    '''-ve because VaR is always quoted as positive'''
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns,level) 
    else:
        raise TypeError('Expected returns to be a series or dataframe')
        
def var_model(returns, model='Gaussian', level=5):
    ''' here we firstly assume a std model for the distribution of returns and 
        then based on that we calculate the z_score for a particular confidence interval'''
    '''Since this VaR Method fits an assumed model on sample data so it has very high Model risk and very low Sample Risk'''
    '''-ve because VaR is always quoted as positive'''
    if model=='Gaussian':
        z_score=scipy.stats.norm.ppf(0.01*level)
        return -1*(returns.mean()+(z_score*returns.std(ddof=0)))
    else:
        raise TypeError('define z_score for non gaussian models in the function')  
        
def var_modified(returns, level=5):
    ''' we know that the returns are not normally distributed as they have some skewness and excess kurtosis-
        so we use Cornish- Fischer formula to account for the adjustments and 
        calculate a modified z score for a particular confidence interval 
        using the standard gaussian z score and skewness and kurtosis'''
    
    '''Since this VaR Method fits makes an adjustment to account for irregualrites assumed in our model 
        on sample data so it provides a tradeoff between Model risk and Sample Risk'''
    '''-ve because VaR is always quoted as positive'''
    
    z_score=scipy.stats.norm.ppf(0.01*level)
    s=skewness(returns)
    k= excess_kurtosis(returns)
    z_modified=(z_score
                +(z_score**2-1)*s/6 
                +(z_score**3-3*z_score)*k/24
                -(2*z_score**3-5*z_score)*(s**2)/36
               )
    return -1*(returns.mean()+(z_modified*returns.std(ddof=0)))

def cvar_historic(returns, level=5):
    '''CVAR or Expected Shortfall takes mean of all the losses below the Var at a certain Condfidence Level'''
    '''returns the historic conditional var/ Expected ShortFall from the sample returns with a confidence interval 
        of 100-level% '''
    '''Since this CVaR Method uses historical sample data so it has very high Sample risk and very low Model Risk'''
    '''-ve because CVaR is always quoted as positive'''
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(cvar_historic, level=level)
    elif isinstance(returns, pd.Series):
        is_beyond=returns<=-1*var_historic(returns, level=level)
        return -returns[is_beyond].mean() 
    else:
        raise TypeError('Expected returns to be a series or dataframe')

def cvar_model(returns, model='Gaussian', level=5):
    '''CVAR or Expected Shortfall takes mean of all the losses below the Var at a certain Condfidence Level'''
    ''' here we firstly assume a std model for the distribution of returns and 
        then based on that we calculate the ES for a particular confidence interval'''
    '''Since this CVaR Method fits an assumed model on sample data so it has very high Model risk and very low Sample Risk'''
    '''-ve because CVaR is always quoted as positive'''
    is_beyond=returns<=-1*var_model(returns,model=model, level=level)
    return -returns[is_beyond].mean()

def cvar_modified(returns, level=5):
    ''' we use Cornish- Fischer formula to account for the adjustments and 
        calculate ES for a particular confidence interval 
        using the standard gaussian z score and skewness and kurtosis'''
    
    '''Since this CVaR Method fits makes an adjustment to account for irregualrites assumed in our model 
        on sample data so it provides a tradeoff between Model risk and Sample Risk'''
    '''-ve because CVaR is always quoted as positive'''
    
    is_beyond=returns<=-1*var_modified(returns, level=level)
    return -returns[is_beyond].mean()

######### Portfolio Summary ###########
def summary_stats(r, risk_free_rate=0.03,period_freq=12):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualised_returns, period_freq=period_freq)
    ann_vol = r.aggregate(annualised_vol, period_freq=period_freq)
    ann_sr = r.aggregate(sharpe_ratio, risk_free_rate=risk_free_rate, period_freq=period_freq)
    dd = r.aggregate(lambda r: drawdown(r).min())
    skew = r.aggregate(skewness)
    excess_kurt = r.aggregate(excess_kurtosis)
    mod_var5=r.aggregate(var_model,model='Gaussian')
    cf_var5 = r.aggregate(var_modified)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Excess Kurtosis": excess_kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Gaussian VaR (5%)": mod_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })