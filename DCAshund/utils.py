#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCAshund - utils
#

from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt


def merge_stock_data(file_paths, fill_missing=True):
    dataframes = []
    start_dates = []
    end_dates = []
    
    for file_path in file_paths:
        # Load the data
        data = pd.read_csv(file_path)
        # Keep only the 'Date', 'Close', 'High', 'Low', and 'Open' columns
        data = data[['Date', 'Close', 'High', 'Low', 'Open']]
        # Set the 'Date' column as the index
        data.set_index('Date', inplace=True)
        # Get the name of the file without the extension
        file_name = Path(file_path).stem
        # Add a level to the columns with the file name
        data.columns = pd.MultiIndex.from_product([[file_name], data.columns])
        # Append the dataframe to the list
        dataframes.append(data)
        
        # Store the start and end dates
        start_dates.append(data.index.min())
        end_dates.append(data.index.max())
    
    # Merge all the dataframes on the 'Date' index
    merged_data = pd.concat(dataframes, axis=1)
    
    # Find the maximum start date and minimum end date
    max_start_date = max(start_dates)
    min_end_date = min(end_dates)
    
    # Crop the merged dataframe to this date range
    merged_data = merged_data.loc[max_start_date:min_end_date]
    
    # If fill_missing option is enabled, fill missing data using forward fill method
    if fill_missing:
        if merged_data.isna().any().any():
            print("Warning: Missing data detected. Forward filling missing data...")
        merged_data.fillna(method='ffill', inplace=True)
    
    return merged_data


def plot_simulation(simulation_result: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(15,5))

    # Plot value and cumulative_investment
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value and Cumulative Investment', color='tab:blue')
    ax1.plot(pd.to_datetime(simulation_result.index), simulation_result['value'], color='tab:blue', label='Value')
    ax1.plot(pd.to_datetime(simulation_result.index), simulation_result['cumulative_investment'], color='tab:orange', label='Cumulative Investment')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Performance (%)', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(pd.to_datetime(simulation_result.index), simulation_result['perf'], color='tab:red', label='Performance')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    # Set x-axis major ticks to yearly interval
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
