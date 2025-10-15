"""
Feature engineering module for creating technical indicators and preparing data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle


class FeatureEngineer:
    """Create technical indicators and prepare features for modeling."""
    
    def __init__(self, data):
        """
        Initialize the feature engineer.
        
        Args:
            data (pd.DataFrame): Raw stock data with OHLCV columns
        """
        self.data = data.copy()
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
    def add_moving_averages(self, short_window=7, long_window=21):
        """
        Add moving average features.
        
        Args:
            short_window (int): Short-term MA window
            long_window (int): Long-term MA window
        """
        print(f"Adding moving averages (MA{short_window}, MA{long_window})...")
        
        self.data[f'MA_{short_window}'] = self.data['Close'].rolling(window=short_window).mean()
        self.data[f'MA_{long_window}'] = self.data['Close'].rolling(window=long_window).mean()
        
        return self.data
    
    def add_rsi(self, period=14):
        """
        Add Relative Strength Index (RSI).
        
        Args:
            period (int): RSI period
        """
        print(f"Adding RSI (period={period})...")
        
        # Calculate price changes
        delta = self.data['Close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        return self.data
    
    def add_macd(self, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Args:
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
        """
        print(f"Adding MACD (fast={fast}, slow={slow}, signal={signal})...")
        
        # Calculate EMAs
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        self.data['MACD'] = ema_fast - ema_slow
        
        # Signal line
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        
        # MACD histogram
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        return self.data
    
    def add_volatility(self, window=20):
        """
        Add volatility feature (rolling standard deviation).
        
        Args:
            window (int): Rolling window for volatility calculation
        """
        print(f"Adding volatility (window={window})...")
        
        self.data['Volatility'] = self.data['Close'].rolling(window=window).std()
        
        return self.data
    
    def add_price_change(self):
        """Add price change and percentage change features."""
        print("Adding price change features...")
        
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['Price_Change_Pct'] = self.data['Close'].pct_change() * 100
        
        return self.data
    
    def create_all_features(self, ma_short=7, ma_long=21, rsi_period=14,
                           macd_fast=12, macd_slow=26, macd_signal=9):
        """
        Create all technical indicators.
        
        Args:
            ma_short (int): Short MA window
            ma_long (int): Long MA window
            rsi_period (int): RSI period
            macd_fast (int): MACD fast period
            macd_slow (int): MACD slow period
            macd_signal (int): MACD signal period
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Add all features
        self.add_moving_averages(ma_short, ma_long)
        self.add_rsi(rsi_period)
        self.add_macd(macd_fast, macd_slow, macd_signal)
        self.add_volatility()
        self.add_price_change()
        
        # Drop rows with NaN values (from rolling calculations)
        rows_before = len(self.data)
        self.data.dropna(inplace=True)
        rows_after = len(self.data)
        
        print(f"\nRows dropped due to NaN: {rows_before - rows_after}")
        print(f"Final dataset size: {rows_after} rows")
        
        # Define feature columns (excluding Date and target)
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            f'MA_{ma_short}', f'MA_{ma_long}',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Volatility', 'Price_Change', 'Price_Change_Pct'
        ]
        
        print(f"\nTotal features created: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns}")
        print("="*60 + "\n")
        
        return self.data
    
    def scale_features(self, fit=True):
        """
        Scale features using MinMaxScaler.
        
        Args:
            fit (bool): Whether to fit the scaler (True for training, False for inference)
        
        Returns:
            np.ndarray: Scaled features
        """
        print("Scaling features...")
        
        if fit:
            scaled_data = self.scaler.fit_transform(self.data[self.feature_columns])
        else:
            scaled_data = self.scaler.transform(self.data[self.feature_columns])
        
        return scaled_data
    
    def save_scaler(self, filepath):
        """
        Save the fitted scaler.
        
        Args:
            filepath (str): Path to save the scaler
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath):
        """
        Load a fitted scaler.
        
        Args:
            filepath (str): Path to the saved scaler
        """
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")


def create_sequences(data, lookback, forecast_horizon):
    """
    Create supervised learning sequences using sliding window.
    
    Args:
        data (np.ndarray): Scaled feature data
        lookback (int): Number of time steps to look back
        forecast_horizon (int): Number of time steps to forecast
    
    Returns:
        tuple: (X, y) where X is input sequences and y is target sequences
    """
    print(f"\nCreating sequences with lookback={lookback}, forecast_horizon={forecast_horizon}...")
    
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast_horizon + 1):
        # Input: lookback window of all features
        X.append(data[i - lookback:i, :])
        
        # Target: next forecast_horizon closing prices (index 3 is 'Close')
        y.append(data[i:i + forecast_horizon, 3])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape} (samples, lookback, features)")
    print(f"y shape: {y.shape} (samples, forecast_horizon)")
    
    return X, y


def prepare_data_for_training(data, lookback, forecast_horizon, 
                              train_ratio=0.7, val_ratio=0.2):
    """
    Complete data preparation pipeline.
    
    Args:
        data (pd.DataFrame): Raw stock data
        lookback (int): Lookback window
        forecast_horizon (int): Forecast horizon
        train_ratio (float): Training data ratio
        val_ratio (float): Validation data ratio
    
    Returns:
        dict: Dictionary containing train/val/test splits and metadata
    """
    # Create features
    engineer = FeatureEngineer(data)
    engineered_data = engineer.create_all_features()
    
    # Scale features
    scaled_data = engineer.scale_features(fit=True)
    
    # Create sequences
    X, y = create_sequences(scaled_data, lookback, forecast_horizon)
    
    # Split data
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print("\n" + "="*60)
    print("DATA SPLIT SUMMARY")
    print("="*60)
    print(f"Total sequences: {n_samples}")
    print(f"Training: {len(X_train)} ({train_ratio*100:.0f}%)")
    print(f"Validation: {len(X_val)} ({val_ratio*100:.0f}%)")
    print(f"Test: {len(X_test)} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print("="*60 + "\n")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': engineer.scaler,
        'feature_columns': engineer.feature_columns,
        'engineered_data': engineered_data
    }


if __name__ == "__main__":
    # Test feature engineering
    from .data_fetcher import fetch_stock_data
    from .config import Config
    
    # Fetch data
    data = fetch_stock_data(
        ticker=Config.DEFAULT_TICKER,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE
    )
    
    # Prepare data
    prepared_data = prepare_data_for_training(
        data,
        lookback=Config.LOOKBACK_WINDOW,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    
    print("\nFeature engineering completed successfully!")
