# backtester

Used for asset allocation back testing - backtest a list of tickers through time, specifying weights,
rebalance methods and frequency.

## Usage

```
Example:

python backtester\strategy.py --tickers ITOT IEMG GLD TLT

Arguments:

tickers - list of tickers, seperated by spaces
start - Starting date, e.g. '20060101'
end' - Ending date, e.g. '20060101'
method - Method of rebalance - min_variance, momentum, static
freq' - frequency of rebalance - BA, BM..
ff_model - 3_factor or 4_factor

```

## Strategy

The Strategy object is an abstract base class. The main one currently in use is AssetAllocationStrategy

## Performance

The Performance object calculates and stores all performance related information and backtest stats for a given
equity curve

## SecurityData

The SecurityData object is used to store price and returns information (from yahoo finance)

## FamaFrenchData

The FamaFrenchData object is used to store Fama French factor data (from the Fama French website)

## To do

- Implement for monthly (in addition to daily)
- Add additional mean-variance methods

