# Strategy, Performance, SecurityData, FamaFrenchData
# classes used to test asset allocation strategy
#  Author: Alex H Chao
# ================================

import numpy as np
import pandas as pd
import argparse

from abc import ABCMeta, abstractmethod
from pandas_datareader.famafrench import get_available_datasets
from collections import OrderedDict

import pandas_datareader.data as web

from strategy.helper import *


class Strategy(object):
    """
    Strategy object - abstract strategy class
    Initialize with securityData Object
    
    """
    __metaclass__ = ABCMeta
    def __init__(self, securityDataObj):
        self.capital = None
        self.weights = None
        if not isinstance(securityDataObj,SecurityData):
            raise TypeError('Please pass in a SecurityData object.')
        self.security_data = securityDataObj

    @abstractmethod
    def run_strategy(self):
        raise NotImplementedError


class AssetAllocationStrategy(Strategy):
    """
    Dynamic Asset allocation strategy based on a rebalance frequency
    """
    def __init__(self, securityDataObj):
        Strategy.__init__(self, securityDataObj)

    # CALC NEW WEIGHTS EVERY REBALANCE PERIOD
    def run_strategy(self, target_weights=None, freq='BM', method='min_var', lookback=252,
                                     capital=1000):
        """
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

"""
NOT IN USE
class Portfolio(object):
 
    def __init__(self, tickers):
        self.tickers = tickers
"""

class SecurityData(object):
    """
    SecurityData object is used to store stock data - historical price, returns, tickers
    Daily freq, for now
    """
    def __init__(self, tickers, start, end = datetime.now()):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.returns = None
        self.price = None

    def load_data(self):
        """
        Loads data from yahoo finance
        :return: 
        """
        print('Loading data from yahoo finance...')
        self.price = get_data_from_yahoo(self.tickers, self.start, self.end,'Adj Close').dropna()
        self.returns = self.price.pct_change()

class Performance(object):
    """
    Performance object is used to performance metrics and stats for a given equity curve
    """
    def __init__(self, equity_curve):
        self.equity_curve = equity_curve
        self.stats = OrderedDict()
        self.returns = self.equity_curve.pct_change()
        self.ff_regression = None
        self.annual_returns = None

    def calc_stats(self):
        self.stats = get_stats(self.equity_curve)
        self.annual_returns = get_annual_returns(self.equity_curve)

    def run_fama_french_regression(self, ff_obj):
        if not isinstance(ff_obj,FamaFrenchData):
            raise TypeError('please pass in a FamaFrenchData object..')
        print('Running fama french regression...')
        self.ff_regression = pd.ols(x=ff_obj.data.set_index('date'), y=self.returns*100)
        self.stats['annual_alpha'] = self.ff_regression._beta_raw[-1] * 252 / 100 # daily
        self.stats['beta'] = self.ff_regression._beta_raw[0]
        self.stats['alpha_t_stat'] = self.ff_regression.t_stat[-1]

    def print_stats(self):
        print('------------------------ Stats --------------------------')
        print_dict(self.stats)
        print('------------------- Annual Returns ----------------------')
        print(self.annual_returns)
        print(self.ff_regression)


class FamaFrenchData(object):
    """
    FamaFrenchObject is used to store fama french factor data
    """
    def __init__(self,type='4_factor', start = datetime(2006,1,1), end = datetime.now()):
        self.data = None
        self.type = type
        self.start = start
        self.end = end

    def load_data(self):
        print('Loading fama french factor data..')
        if self.type == '4_factor': # 3 factor plus momentum
            ff_3 = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
            ff_mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench")[1]
            ff = pd.merge(ff_3, ff_mom, left_index=True, right_index=True).reset_index()
            ff.columns = ['date','mkt-rf','smb','hml','rf','umd']
        elif self.type == '3_factor':
            ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0].reset_index()
            ff.columns = ['date', 'mkt-rf', 'smb', 'hml', 'rf']
        elif self.type == '5_factor':
            ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench").reset_index()
            ff.columns = ['date', 'mkt-rf', 'smb', 'hml', 'rmw','cma','rf']
        else:
            raise ValueError('Type not valid, please enter 3_factor, 4_factor, or 5_factor.')
        ff.date = pd.to_datetime(ff.date, format='%Y%m%d')
        #ff.data.iloc[:, 1:] = ff.data.iloc[:, 1:] / 100 # scale the numbers to percentages
        self.data = ff[(ff.date >= self.start) & (ff.date <= self.end)]

# ====

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers',nargs='*', default = ['SPY','TLT'], help='list of tickers')
    parser.add_argument('--start', default='20060101', help='Starting date, e.g. 20060101')
    parser.add_argument('--end', default='20161231', help='Ending date, e.g. 20060101')
    parser.add_argument('--method', default='static', help='method of rebalance - min_variance, momentum, static')
    parser.add_argument('--freq', default='BM', help='frequency of rebalance - BA, BM..')
    parser.add_argument('--ff_model', default='4_factor', help='3_factor or 4_factor ff model')

    #import pdb; pdb.set_trace()

    args = parser.parse_args()

    tickers = args.tickers
    start = args.start
    end = args.end
    method = args.method
    freq = args.freq
    type = args.ff_model

    data = SecurityData(tickers,start)
    data.load_data()

    strat = AssetAllocationStrategy(data)
    strat.run_strategy(freq=freq)

    ff = FamaFrenchData(type=type,start=start)
    ff.load_data()

    perf = Performance(strat.capital)
    perf.calc_stats()
    perf.run_fama_french_regression(ff)
    perf.print_stats()


"""

tickers = ['SPXL','TMF']
tickers = ['DFLVX']
start = datetime(2006,1,1)
data = SecurityData(tickers,start)
data.load_data()

strat = AssetAllocationStrategy(data)
strat.run_strategy(freq='BM')

ff = FamaFrenchData(type='4_factor',start=start)
ff.load_data()
ff.data

perf = Performance(strat.capital)
perf.calc_stats()
perf.run_fama_french_regression(ff)
print(perf.ff_regression)
perf.print_stats()

print(perf)

"""