"""
Data fetcher module for downloading stock price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockDataFetcher:
    """Fetch and preprocess stock data from Yahoo Finance."""
    
    def __init__(self, ticker, start_date, end_date, interval='1d'):
        """
        Initialize the data fetcher.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval (default: '1d')
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = None
        
    def fetch_data(self):
        """
        Download stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Raw stock data with OHLCV columns
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        try:
            # Download data using yfinance
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            self.data = data
            print(f"Successfully fetched {len(data)} rows of data")
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise
    
    def handle_missing_values(self):
        """
        Handle missing values and non-trading days.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        print("\nHandling missing values...")
        
        # Check for missing values
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Forward fill for missing values (use previous day's data)
        self.data.fillna(method='ffill', inplace=True)
        
        # Backward fill for any remaining NaN at the start
        self.data.fillna(method='bfill', inplace=True)
        
        # Drop any remaining rows with NaN (if any)
        self.data.dropna(inplace=True)
        
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        
        return self.data
    
    def get_data_summary(self):
        """
        Print summary statistics of the fetched data.
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        print("\n" + "="*60)
        print(f"DATA SUMMARY FOR {self.ticker}")
        print("="*60)
        print(f"\nShape: {self.data.shape}")
        print(f"\nDate Range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"\nColumns: {list(self.data.columns)}")
        print("\nStatistical Summary:")
        print(self.data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        print("="*60 + "\n")
    
    def save_data(self, filepath):
        """
        Save the fetched data to CSV.
        
        Args:
            filepath (str): Path to save the CSV file
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        self.data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Convenience function to fetch and clean stock data.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval
    
    Returns:
        pd.DataFrame: Cleaned stock data
    """
    fetcher = StockDataFetcher(ticker, start_date, end_date, interval)
    data = fetcher.fetch_data()
    data = fetcher.handle_missing_values()
    fetcher.get_data_summary()
    
    return data


if __name__ == "__main__":
    # Test the data fetcher
    from .config import Config
    
    data = fetch_stock_data(
        ticker=Config.DEFAULT_TICKER,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE,
        interval=Config.DATA_INTERVAL
    )
    
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
