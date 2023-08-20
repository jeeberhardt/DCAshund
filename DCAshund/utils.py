#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCAshund - utils
#

import re
import requests
from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt


def get_market_data_from_boursorama(symbol, duration='4Y', start_date=None):
    """
    Get market data from boursorama.com
    
    Parameters
    ----------
    symbol : str
        Symbol of the stock (e.g. 'AAPL' for Apple).
    start_date : str
        Start date of the data (default is None). Format is 'DD/MM/YYYY'.
    duration : str
        Duration of the data (default is '4Y').

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the data. Columns are ['Open', 'High', 'Low', 'Close']. Index is the date.
    
    """
    base_url = "https://www.boursorama.com/_formulaire-periode/"
    current_page = 1
    number_pages = 1
    duration_left = 0
    data = []
    if start_date is None:
        start_date = ''

    assert len(duration) >= 2, "Duration must be 2 characters. The first one is a number and the second one is the time unit (Y or M)."
    assert duration[-1] in ['Y', 'M'], "The second character of duration must be Y or M."

    if duration[-1] == 'Y':
        if int(duration[:-1]) > 4:
            duration_left = int(duration[:-1]) - 4
            duration = '4Y'

    while current_page <= number_pages:
        tmp = []
        parameters_url = f"page-{current_page}?symbol={symbol}&historic_search[duration]={duration}&historic_search[startDate]={start_date}"

        response = requests.get(f"{base_url}{parameters_url}")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get number of pages
        try:
            table = soup.find_all("span", {"class": "c-pagination__content"})[-1]
            number_pages = int(table.text)
        except:
            number_pages = 1

        # Get data
        regex = re.compile('c-table__cell *')
        table = soup.find_all("td", {"class" : regex})

        for i, t in enumerate(table):
            if i % 6 == 0:
                if tmp:
                    data.append(tmp)
                tmp = [t.text.strip()]
            else:
                tmp.append(t.text.strip())

        data.append(tmp)
        current_page += 1

    try:
        df = pd.DataFrame(data, columns=['Date', 'Close', 'Variation', 'High', 'Low', 'Open'])
    except:
        # No data, return empty dataframe
        df = pd.DataFrame([], columns=['Open', 'High', 'Low', 'Close'])
        return df

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.set_index('Date')
    df = df[['Open', 'High', 'Low', 'Close']]

    if duration_left:
        # Little trick to get data from the previous years > 4 years
        start_date = df.index[-1] - pd.DateOffset(years=duration_left)
        df2 = get_data_from_boursorama(symbol, f"{duration_left}Y", start_date.strftime("%d/%m/%Y"))
        df = pd.concat([df2, df])
        df = df[~df.index.duplicated(keep='first')]

    df.sort_index(inplace=True)

    return df


def merge_stock_data(file_paths, fill_missing=True):
    """
    Merge stock data from multiple files into a single dataframe.

    Parameters
    ----------
    file_paths : list
        A list of file paths to the stock data files.
    fill_missing : bool, optional
        Whether to fill missing data using forward fill method. The default is True.

    Returns
    -------
    merged_data : pandas.DataFrame
        The merged stock data.
    
    """
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


def plot_simulation(simulation_result, fig_filename=None):
    """
    Plot the simulation result.

    Parameters
    ----------
    simulation_result : pandas.DataFrame
        The simulation result returned by DCAshund.simulate().
    fig_filename : str, optional
        The filename to save the figure to. If not specified, the figure will just be displayed instead.

    """
    fig, ax1 = plt.subplots(figsize=(15, 5))

    # Plot value and cumulative_investment
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value and Cumulative Investment', color='tab:blue')
    ax1.plot(pd.to_datetime(simulation_result.index), simulation_result['value'], color='tab:blue', label='Value')
    ax1.plot(pd.to_datetime(simulation_result.index), simulation_result['cumulative_investment'], color='tab:orange', label='Cumulative Investment')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Performance (%)', color='tab:red')  # we already handled the x-label with ax1
    ax2.plot(pd.to_datetime(simulation_result.index), simulation_result['perf'], color='tab:red', label='Performance')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Set x-axis major ticks to yearly interval
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    handles, labels = [], []
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    plt.legend(handles, labels)

    if fig_filename:
        plt.savefig(fig_filename, bbox_inches='tight', dpi=300)

    plt.show()
