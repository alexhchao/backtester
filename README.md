# backtester

Used for asset allocation back testing - backtest a list of tickers through time, specifying weights,
rebalance methods and frequency.

## Usage

'''
Example:

python backtester\strategy.py --tickers ITOT IEMG GLD TLT

'''

## Strategy

The Strategy object is an abstract base class. The main one currently in use is AssetAllocationStrategy

## Performance

The Performance object calculates and stores all performance related information and backtest stats for a given
equity curve

## SecurityData

The SecurityData object is used to store price and returns information (from yahoo finance)

## FamaFrenchData

The FamaFrenchData object is used to store Fama French factor data (from the Fama French website)

