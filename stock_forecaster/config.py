"""
Configuration file for stock price forecasting model.
Contains all hyperparameters and settings.
"""

import torch

class Config:
    """Configuration class for the stock forecasting pipeline."""
    
    # Data Configuration
    DEFAULT_TICKER = "AAPL"
    START_DATE = "2000-01-01"  # 25 years of data
    END_DATE = "2025-01-01"
    DATA_INTERVAL = "1d"
    
    # Feature Engineering
    LOOKBACK_WINDOW = 60  # Number of days to look back
    FORECAST_HORIZON = 30  # Number of days to forecast (7-30)
    
    # Data Pipeline
    USE_DATA_PIPELINE = True  # Enable organized data storage
    DATA_BASE_PATH = "data"
    
    # Technical Indicators
    MA_SHORT = 7
    MA_LONG = 21
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Train/Val/Test Split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Model Architecture
    INPUT_FEATURES = 10  # Will be updated based on feature engineering
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    BIDIRECTIONAL = True
    
    # Training Configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15  # Early stopping patience
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    MODEL_SAVE_PATH = "models/stock_forecaster.pth"
    SCALER_SAVE_PATH = "models/scaler.pkl"
    OUTPUT_PATH = "data/output/"
    PLOTS_PATH = "plots/"
    
    # Random Seed for Reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def update_forecast_horizon(cls, days):
        """Update forecast horizon dynamically."""
        if 7 <= days <= 30:
            cls.FORECAST_HORIZON = days
        else:
            raise ValueError("Forecast horizon must be between 7 and 30 days")
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary."""
        return {
            'ticker': cls.DEFAULT_TICKER,
            'lookback_window': cls.LOOKBACK_WINDOW,
            'forecast_horizon': cls.FORECAST_HORIZON,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'hidden_size': cls.HIDDEN_SIZE,
            'num_layers': cls.NUM_LAYERS,
            'dropout': cls.DROPOUT,
            'device': str(cls.DEVICE)
        }
