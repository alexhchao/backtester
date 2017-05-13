import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from backtester.strategy.helper import *


class Strategy(object):
    """
    Strategy object - abstract strategy class
    Initialize with securityData Object
    
    """
    __metaclass__ = ABCMeta
    def __init__(self, securityDataObj):
        self.capital = None
        self.weights = None
        self.security_data = securityDataObj

    @abstractmethod
    def run_strategy(self):
        raise NotImplementedError



class AssetAllocationStrategy(Strategy):

    def __init__(self, securityDataObj):
        Strategy.__init__(self, securityDataObj)

    # CALC NEW WEIGHTS EVERY REBALANCE PERIOD
    def run_strategy(self, target_weights=None, freq='BM', method='min_var', lookback=252,
                                     capital=1000):
        """
        Run asset allocation strategy based on rebalance frequency
        * DYNAMIC WEIGHTS AT EACH REBALANCE PERIOD
        target weights -- list
        freq -- rebalance frequency
           * BM = Monthly
           * BQ = Quarterly
           * BA = Annually
           * none 
        method -- max_sharpe or min_variance
        capital -- capital to start with 

        """
        print('Running {} strategy with freq = {}'.format(method,freq))

        if target_weights is None: # default to equal weight
            num_tickers = self.security_data.returns.shape[1]
            target_weights = [1.00/num_tickers for x in range(num_tickers)]
        returns = self.security_data.returns

        weights = returns.copy()
        weights.ix[:, :] = np.NaN  # initialize

        weights.iloc[0, :] = target_weights

        if freq == 'none':
            rebalance_days = []
        else:
            rebalance_days = weights.resample(freq, how='last').index

        for i in range(1, len(weights)):
            if weights.index[i] in (rebalance_days):
                # RECALC WEIGHTS, BASED ON SOME LOOKBACK
                # =====================================
                print('Rebalancing portfolio on {}'.format(weights.index[i]))
                min_index = max(i - lookback, 0)
                if method == 'min_variance':
                    sigma = returns.iloc[min_index:i - 1, :].cov()
                    new_weights = get_min_var_weights(sigma)
                #elif method == 'max_sharpe':
                #    new_weights = calc_mean_var_weights(returns.iloc[min_index:i - 1, :])
                elif method == 'momentum':  # USE MOMENTUM
                    new_weights = get_momentum_ranks(self.security_data.price, i, lookback)
                else: # static weights
                    new_weights = target_weights
                # ==================================
                weights.iloc[i, :] = new_weights  # rebalance portfolio
            else:
                weights.iloc[i, :] = weights.iloc[i - 1, :] * (
                    1 + returns.iloc[i, :])  # calculate weights each day based on returns
                weights.iloc[i, :] = weights.iloc[i, :] / weights.iloc[i,
                                                          :].sum()  # we need to divide by sum to normalize to 1

        port_returns = (weights.shift(1) * returns).sum(axis=1)
        self.capital = (1 + port_returns).cumprod()

        self.weights = weights


class Portfolio(object):
    """
    Portfolio object - given weights and price, generates equity curve
    """
    def __init__(self, tickers):
        self.tickers = tickers
        self.


class SecurityData(object):
    """
    object to store stock level data - price, returns, tickers
    Daily freq
    """
    def __init__(self):
        self.tickers = None
        self.returns = None

        self.price = None
        self.start = None
        self.end = None

    def load_data(self, tickers, start, end = datetime.now()):
        """
        Loads data from yahoo finance
        :return: 
        """
        self.tickers = tickers
        self.start = start
        self.end = end

        print('Loading data from yahoo finance...')
        self.price = get_data_from_yahoo(self.tickers, start, end,'Adj Close').dropna()
        self.returns = self.price.pct_change()


# ====

tickers = ['SPY','TLT','VWO','GLD']
start = datetime(2006,1,1)
data = SecurityData()
data.load_data(tickers,start)

strat = AssetAllocationStrategy(data)
strat.run_strategy(freq='BM')

strat.capital
strat.weights

get_stats(strat.capital)

get_annual_returns(strat.capital)