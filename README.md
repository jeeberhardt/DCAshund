# DCAshund
Long on strategy, short on complexity, your new loyal and compact DCA companion.

## Disclaimer
This tool is provided for educational and informational purposes only and should not be construed as investment advice. Investing involves risk, including the possible loss. Past performance is not indicative of future results. Always do your own research and consider your financial situation carefully before making investment decisions.

## Installation
I highly recommand you to install Mamba (https://github.com/conda-forge/miniforge#mambaforge) if you want a clean python environnment. To install everything properly with `mamba`, you just have to do this:

```bash
mamba env create -f environment.yaml -n dcashund
mamba activate dcashund
```

Finally, we can install the `DCAshund` package
```bash
$ git clone https://github.com/jeeberhardt/dcashund
$ cd dcashund
$ pip install .
```

## Quick tutorial

Here's a step-by-step guide on how to use DCAshund for simulating DCA strategies:

1. Import necessary modules: We'll need numpy, yfinance for retrieving stock data, and several modules from `DCAshund``:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import yfinance as yf
import pandas as pd

from dcashund import DCAshund, merge_stock_data, plot_simulation

```

2. Retrieve and save stock data: We'll use `yfinance`` to retrieve historical data for certain stocks and save the data as CSV files. Replace the stock symbols with those of the stocks you're interested in:

```python
isins = [('FR0010315770', 'Lyxor MSCI World UCITS ETF - Dist'),
         ('LU0533033667', 'Lyxor MSCI World Information Technology ETF - Dist')]

for fund in isins:
    ticker = yf.Ticker(fund[0])
    history = ticker.history(period="max")

    history['Date'] = pd.to_datetime(history.index).date
    history.set_index('Date', inplace=True)

    if not history.empty:
        history.to_csv(f'{fund[0]}.csv')

```

3. Merge stock data: If you have data from multiple stocks, you can use merge_stock_data to combine them into one dataframe:

```python
merged_data = merge_stock_data(['FR0010687749.csv', 'FR0010315770.csv'])

```

Simulate DCA strategies: Now we're ready to simulate some DCA strategies. We'll initialize a DCAshund object, simulate a strategy, and plot the results:

- The `weights` parameter represents the proportion of each investment in the portfolio. The order of the weights corresponds to the order of the asset data files provided in `merge_stock_data`.
- The `dca` parameter represents the amount of money invested in each period for the Dollar Cost Averaging strategy. 
- The `entry_fee` parameter represents the transaction fees associated with buying assets, expressed as a proportion of the transaction amount.

```python
# We are going to use the Close prices
dh = DCAshund('Close')

# Here we invest 500 euros (85 % in FR0010315770 and 15 % in LU0533033667) 
# every month starting on the fourth day, considering an entry fee of 1 %.
results = dh.simulate(merged_data, '2019-03-04', '2023-07-14', weights=[0.85, 0.15], dca=500, entry_fee=1./100)

# Plot the simulation result
plot_simulation(results)

```

That's it! You've now learned how to use `DCAshund` to simulate DCA strategies. Remember, investing involves risk. Always do your own research before making investment decisions.
