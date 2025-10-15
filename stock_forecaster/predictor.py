"""
Predictor module for making multi-step stock price forecasts.
"""

import torch
import numpy as np
import pandas as pd
from .model_trainer import LSTMForecaster


class StockPredictor:
    """Make predictions using trained LSTM model."""
    
    def __init__(self, model, scaler, device):
        """
        Initialize the predictor.
        
        Args:
            model (nn.Module): Trained model
            scaler: Fitted MinMaxScaler
            device (torch.device): Device to run predictions on
        """
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (np.ndarray): Input sequences
        
        Returns:
            np.ndarray: Predictions
        """
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def predict_batch(self, X):
        """
        Make batch predictions.
        
        Args:
            X (np.ndarray): Batch of input sequences
        
        Returns:
            np.ndarray: Batch predictions
        """
        return self.predict(X)
    
    def inverse_transform_predictions(self, predictions):
        """
        Convert scaled predictions back to original price scale.
        
        Args:
            predictions (np.ndarray): Scaled predictions
        
        Returns:
            np.ndarray: Predictions in original scale
        """
        # Create a dummy array with the same shape as scaler expects
        n_features = self.scaler.n_features_in_
        
        # Predictions are for 'Close' price (index 3 in features)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Create dummy array
        dummy = np.zeros((predictions.shape[0], n_features))
        
        # Place predictions in the 'Close' column (index 3)
        dummy[:, 3] = predictions[:, 0] if predictions.shape[1] == 1 else predictions[:, -1]
        
        # Inverse transform
        inverse = self.scaler.inverse_transform(dummy)
        
        # Extract the 'Close' prices
        return inverse[:, 3]
    
    def forecast_future(self, last_sequence, forecast_steps):
        """
        Forecast multiple steps into the future using iterative prediction.
        
        Args:
            last_sequence (np.ndarray): Last known sequence (lookback_window, features)
            forecast_steps (int): Number of steps to forecast
        
        Returns:
            np.ndarray: Forecasted values
        """
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_steps):
            # Predict next step
            pred = self.predict(current_sequence.reshape(1, *current_sequence.shape))
            
            # Take the first prediction (next immediate step)
            next_value = pred[0, 0]
            forecasts.append(next_value)
            
            # Update sequence: remove oldest, add newest
            # Create new row with predicted close price
            new_row = current_sequence[-1].copy()
            new_row[3] = next_value  # Update 'Close' price
            
            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(forecasts)


def load_model_for_prediction(model_path, scaler_path, input_size, hidden_size, 
                              num_layers, forecast_horizon, dropout, 
                              bidirectional, device):
    """
    Load a trained model and scaler for prediction.
    
    Args:
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
        input_size (int): Number of input features
        hidden_size (int): Hidden layer size
        num_layers (int): Number of LSTM layers
        forecast_horizon (int): Forecast horizon
        dropout (float): Dropout rate
        bidirectional (bool): Bidirectional LSTM
        device (torch.device): Device to load model on
    
    Returns:
        StockPredictor: Initialized predictor
    """
    import pickle
    
    # Load model
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_horizon=forecast_horizon,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")
    
    return StockPredictor(model, scaler, device)


def predict_on_test_set(predictor, X_test, y_test):
    """
    Make predictions on test set.
    
    Args:
        predictor (StockPredictor): Initialized predictor
        X_test (np.ndarray): Test input sequences
        y_test (np.ndarray): Test target values
    
    Returns:
        dict: Dictionary with predictions and actuals
    """
    print("\n" + "="*60)
    print("MAKING PREDICTIONS ON TEST SET")
    print("="*60)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    print(f"Test samples: {len(X_test)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Actuals shape: {y_test.shape}")
    print("="*60 + "\n")
    
    return {
        'predictions': predictions,
        'actuals': y_test,
        'X_test': X_test
    }


def generate_forecast_report(predictor, predictions, actuals, 
                            forecast_horizon, ticker, save_path=None):
    """
    Generate a detailed forecast report.
    
    Args:
        predictor (StockPredictor): Predictor instance
        predictions (np.ndarray): Model predictions
        actuals (np.ndarray): Actual values
        forecast_horizon (int): Forecast horizon
        ticker (str): Stock ticker
        save_path (str): Path to save report
    
    Returns:
        pd.DataFrame: Forecast report
    """
    # Take last prediction and actual for detailed comparison
    last_pred = predictions[-1]
    last_actual = actuals[-1]
    
    # Create DataFrame
    days = list(range(1, forecast_horizon + 1))
    
    report_data = {
        'Day': days,
        'Predicted_Close': last_pred,
        'Actual_Close': last_actual,
        'Absolute_Error': np.abs(last_pred - last_actual),
        'Percentage_Error': np.abs((last_pred - last_actual) / last_actual) * 100
    }
    
    report_df = pd.DataFrame(report_data)
    
    # Add summary statistics
    print("\n" + "="*60)
    print(f"FORECAST REPORT FOR {ticker}")
    print("="*60)
    print(f"\nForecast Horizon: {forecast_horizon} days")
    print(f"\nSummary Statistics:")
    print(f"  Mean Absolute Error: {report_df['Absolute_Error'].mean():.4f}")
    print(f"  Mean Percentage Error: {report_df['Percentage_Error'].mean():.2f}%")
    print(f"  Max Absolute Error: {report_df['Absolute_Error'].max():.4f}")
    print(f"  Min Absolute Error: {report_df['Absolute_Error'].min():.4f}")
    print("\n" + "="*60 + "\n")
    
    # Save report
    if save_path:
        report_df.to_csv(save_path, index=False)
        print(f"Forecast report saved to {save_path}")
    
    return report_df


if __name__ == "__main__":
    # Test predictor
    from .data_fetcher import fetch_stock_data
    from .feature_engineering import prepare_data_for_training
    from .config import Config
    import os
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Model not found. Please train the model first.")
    else:
        # Fetch and prepare data
        data = fetch_stock_data(
            ticker=Config.DEFAULT_TICKER,
            start_date=Config.START_DATE,
            end_date=Config.END_DATE
        )
        
        prepared_data = prepare_data_for_training(
            data,
            lookback=Config.LOOKBACK_WINDOW,
            forecast_horizon=Config.FORECAST_HORIZON,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO
        )
        
        # Load model
        input_size = prepared_data['X_train'].shape[2]
        predictor = load_model_for_prediction(
            model_path=Config.MODEL_SAVE_PATH,
            scaler_path=Config.SCALER_SAVE_PATH,
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            forecast_horizon=Config.FORECAST_HORIZON,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL,
            device=Config.DEVICE
        )
        
        # Make predictions
        results = predict_on_test_set(
            predictor,
            prepared_data['X_test'],
            prepared_data['y_test']
        )
        
        # Generate report
        report = generate_forecast_report(
            predictor,
            results['predictions'],
            results['actuals'],
            Config.FORECAST_HORIZON,
            Config.DEFAULT_TICKER,
            save_path=Config.OUTPUT_PATH + 'forecast_report.csv'
        )
        
        print("\nPrediction completed successfully!")
