from quantopian.pipeline.data.builtin import USEquityPricing



tickers = ['ITOT','MUB','IEFA','EEM','TLT','HDV','USMV']
start = datetime(2006,1,1)
end = datetime(2017,12,31)
price = get_data_from_yahoo(tickers,'1/1/2003')

price = price.dropna()
returns = price.pct_change()

weights, capital = asset_alloc_strategy(price, returns, [.10,.10,.20,.40,0.0,.10,.10])

get_stats(capital.portfolio)

mdd,st,end = max_drawdown_absolute(capital.portfolio.pct_change())

weights, capital = dynamic_asset_alloc_strategy(price, returns, [.10,.10,.20,.40,0.0,.10,.10],
                             freq='BM', method='min_var', lookback=252,
                             capital=1000)
get_stats(capital)
capital.plot()


