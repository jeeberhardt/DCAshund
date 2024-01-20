#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCAshund
#

import calendar

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset


class DCAshund:
    def __init__(self, price_type, n_samples=1):
        if price_type not in ['Open', 'Close', 'Low', 'High']:
            raise ValueError("price_type must be either 'Open', 'Close', 'Low' or 'High'")
        self.price_type = price_type
        self.n_samples = n_samples

    def _generate_monthly_dates(self, start_date, end_date):
        """
        Generate all dates within a period that are the specified day of the month.
        
        Parameters
        ----------
        start_date : str
            The start date of the period in 'YYYY-MM-DD' format.
        end_date : str
            The end date of the period in 'YYYY-MM-DD' format.

        Returns
        -------
        dates : pandas.DatetimeIndex
            A DatetimeIndex containing all dates within the specified period.

        """
        # Parse the start and end dates
        start_date_parsed = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)

        # Create an empty list to store the dates
        dates = []

        # Generate all dates within the specified period
        current_date = start_date_parsed
        while current_date <= end_date_parsed:
            # Append the current date to the list
            dates.append(current_date)

            # Calculate the next date
            year = current_date.year
            month = current_date.month % 12 + 1
            day = start_date_parsed.day

            # Adjust the year if necessary
            if month == 1 and current_date.month == 12:
                year += 1

            # Adjust the day if necessary
            day = min(day, calendar.monthrange(year, month)[1])

            # Update the current date
            current_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")

        # Convert dates to 'YYYY-MM-DD' format
        dates = [date + pd.offsets.BDay(1) if date.weekday() > 4 else date for date in dates]
        dates = [date.strftime('%Y-%m-%d') for date in dates]

        # Ensure that dates is of dtype datetime64[ns]
        dates = pd.to_datetime(dates)

        return dates

    def simulate(self, data, start_date, end_date, weights=None, dca=100, entry_fee=0):
        """
        Simulate the performance of a portfolio over a period of time.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe containing the historical prices of the portfolio.
        start_date : str
            The start date of the period in 'YYYY-MM-DD' format.
        end_date : str
            The end date of the period in 'YYYY-MM-DD' format.
        weights : list, optional
            A list of weights for each asset in the portfolio.
            If N  is the number of assets in the portfolio,
            the list must contain N weights that sum to 1.
            If no weights are provided, the function assumes an equal weighting for all assets.
        dca : float, optional
            The amount of money to invest each month. The default is 100.
        entry_fee : float, optional
            The entry fee to pay when buying assets. The default is 0.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe containing the performance of the portfolio.

        """
        assert entry_fee >= 0, f'entry fee must be positive or equal to zero'
        assert dca > 0, f'dca value must be superior than zero'
        
        data = data.xs(self.price_type, level=1, axis=1)

        # Ensure that data.index is of dtype datetime64[ns]
        data.index = pd.to_datetime(data.index)
        
        # Define weights after handling missing data
        if weights is None:
            weights = np.ones(len(data.columns)) / len(data.columns)
            
        msg_error = f'weights must have the same length as the number of columns in history'
        assert len(weights) == len(data.columns), msg_error
        assert np.sum(weights) == 1, f'weights must sum to 1'
        
        # Convert start_date and end_date to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate monthly dates
        drange = self._generate_monthly_dates(start_date, end_date)

        # Select the data within the specified period
        df = data[data.index.isin(drange)].copy()

        # Subtracts the entry fee from the DCA
        dca_after_fee = dca * (1 - entry_fee)

        # Compute the weighted pct change between each day
        df['weighted_price'] = (df * weights).sum(axis=1)

        # Compute the cumulative investment
        df['cumulative_investment'] = dca_after_fee
        df['cumulative_investment'] = df['cumulative_investment'].cumsum()

        # Compute the number of shares bought
        df['shares_bought'] = dca_after_fee / df['weighted_price']
    
        # Compute the value of the portfolio at each day
        df['value'] = df['shares_bought'].cumsum() * df['weighted_price']
    
        # Compute the performance
        df['perf'] = ((df['value'] / df['cumulative_investment']) * 100.) - 100.
        
        df.drop(columns=['shares_bought'], inplace=True)
        df = df.round(decimals=2)

        return df
    
    def simulate_multi_entry_points(self, data, start_date, end_date, frequency='1D', weights=None, dca=100, entry_fee=0):
        """
        Simulate the performance of a portfolio over a period of time starting 
        at different entry points following a given frequency.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe containing the historical prices of the portfolio.
        start_date : str
            The start date of the period in 'YYYY-MM-DD' format.
        end_date : str
            The end date of the period in 'YYYY-MM-DD' format.
        freqency : str
            The frequency of the entry points. The default is '1M'.
            Allowed: 'XM', 'XW', 'XD' for X months, weeks or days, respectly.
            Replace X by a number.
        weights : list, optional
            A list of weights for each asset in the portfolio.
            If N  is the number of assets in the portfolio,
            the list must contain N weights that sum to 1.
            If no weights are provided, the function assumes an equal weighting for all assets.
        dca : float, optional
            The amount of money to invest each month. The default is 100.
        entry_fee : float, optional
            The entry fee to pay when buying assets. The default is 0.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe containing the performance of the portfolio.

        """
        index = []
        data_results = []

        if not ('D' in frequency or 'M' in frequency or 'W' in frequency):
            raise ValueError("freqency must be either 'XM', 'XW' or 'XD' for X months, weeks or days, respectly.")

        if 'D' in frequency:
            offset = int(frequency.replace('D', ''))
            date_offset = DateOffset(days=offset)
        elif 'W' in frequency:
            offset = int(frequency.replace('W', ''))
            date_offset = DateOffset(weeks=offset)
        elif 'M' in frequency:
            offset = int(frequency.replace('M', ''))
            date_offset = DateOffset(months=offset)

        for date in pd.date_range(start_date, end_date, freq=date_offset):
            results = self.simulate(data, date, end_date, weights=weights, dca=dca, entry_fee=entry_fee)
            index.append(results.index[0])
            data_results.append((results["cumulative_investment"][-1], results["perf"][-1], results['value'][-1]))

        df = pd.DataFrame(data=data_results, index=index, columns=['cumulative_investment', 'perf', 'value'])
        df.index = pd.to_datetime(df.index)

        # If we have dataset with only business days data, we will have duplicates
        # since we are using DateOffset to generate the dates.
        df = df[~df.index.duplicated(keep='first')]
        # It happens that dates are not sorted
        df.sort_index(inplace=True)
        
        return df
