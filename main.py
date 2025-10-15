"""
Main orchestration script for stock price forecasting pipeline.
Handles the complete workflow from data fetching to model evaluation.
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add stock_forecaster to path
sys.path.append(os.path.dirname(__file__))

from stock_forecaster.config import Config
from stock_forecaster.data_fetcher import fetch_stock_data
from stock_forecaster.feature_engineering import prepare_data_for_training
from stock_forecaster.model_trainer import LSTMForecaster, ModelTrainer, create_data_loaders
from stock_forecaster.predictor import load_model_for_prediction, predict_on_test_set, generate_forecast_report
from stock_forecaster.evaluator import ModelEvaluator


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_model(ticker, start_date, end_date, forecast_horizon=30):
    """
    Train a new stock forecasting model.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        forecast_horizon (int): Number of days to forecast
    """
    print("\n" + "="*70)
    print(" "*15 + "STOCK PRICE FORECASTING PIPELINE")
    print("="*70)
    print(f"\nTicker: {ticker}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Forecast Horizon: {forecast_horizon} days")
    print(f"Device: {Config.DEVICE}")
    print("="*70 + "\n")
    
    # Set random seeds
    set_random_seeds(Config.RANDOM_SEED)
    
    # Update config
    Config.DEFAULT_TICKER = ticker
    Config.START_DATE = start_date
    Config.END_DATE = end_date
    Config.update_forecast_horizon(forecast_horizon)
    
    # Step 1: Fetch data
    print("\n[STEP 1/5] Fetching stock data...")
    data = fetch_stock_data(ticker, start_date, end_date, Config.DATA_INTERVAL)
    
    # Step 2: Feature engineering and data preparation
    print("\n[STEP 2/5] Feature engineering and data preparation...")
    prepared_data = prepare_data_for_training(
        data,
        lookback=Config.LOOKBACK_WINDOW,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    
    # Save scaler
    os.makedirs(os.path.dirname(Config.SCALER_SAVE_PATH), exist_ok=True)
    import pickle
    with open(Config.SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(prepared_data['scaler'], f)
    print(f"Scaler saved to {Config.SCALER_SAVE_PATH}")
    
    # Step 3: Create data loaders
    print("\n[STEP 3/5] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        prepared_data['X_train'],
        prepared_data['y_train'],
        prepared_data['X_val'],
        prepared_data['y_val'],
        Config.BATCH_SIZE
    )
    
    # Step 4: Initialize and train model
    print("\n[STEP 4/5] Initializing and training model...")
    input_size = prepared_data['X_train'].shape[2]
    
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        forecast_horizon=Config.FORECAST_HORIZON,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL
    )
    
    trainer = ModelTrainer(
        model=model,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE
    )
    
    # Save model
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    trainer.save_model(Config.MODEL_SAVE_PATH)
    
    # Step 5: Evaluate model
    print("\n[STEP 5/5] Evaluating model on test set...")
    
    # Make predictions on test set
    X_test_tensor = torch.FloatTensor(prepared_data['X_test']).to(Config.DEVICE)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()
    
    # Evaluate
    evaluator = ModelEvaluator(
        predictions=test_predictions,
        actuals=prepared_data['y_test'],
        ticker=ticker
    )
    
    metrics = evaluator.calculate_all_metrics()
    
    # Generate all plots
    evaluator.generate_all_plots(
        train_losses=history['train_losses'],
        val_losses=history['val_losses'],
        forecast_horizon=Config.FORECAST_HORIZON,
        plots_dir=Config.PLOTS_PATH
    )
    
    # Save metrics
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    evaluator.save_metrics_to_file(
        os.path.join(Config.OUTPUT_PATH, 'evaluation_metrics.txt')
    )
    
    # Generate forecast report with actual dollar values
    generate_forecast_report(
        predictor=None,
        predictions=test_predictions,
        actuals=prepared_data['y_test'],
        forecast_horizon=Config.FORECAST_HORIZON,
        ticker=ticker,
        scaler=prepared_data['scaler'],
        save_path=os.path.join(Config.OUTPUT_PATH, 'forecast_report.csv')
    )
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED!")
    print("="*70)
    print(f"\nModel saved to: {Config.MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {Config.SCALER_SAVE_PATH}")
    print(f"Plots saved to: {Config.PLOTS_PATH}")
    print(f"Results saved to: {Config.OUTPUT_PATH}")
    print("\n" + "="*70 + "\n")
    
    return model, prepared_data, metrics


def predict_with_trained_model(ticker, start_date, end_date, forecast_horizon=30):
    """
    Make predictions using a trained model.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        forecast_horizon (int): Number of days to forecast
    """
    print("\n" + "="*70)
    print(" "*20 + "MAKING PREDICTIONS")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"\nError: Model not found at {Config.MODEL_SAVE_PATH}")
        print("Please train a model first using --mode train")
        return
    
    # Update config
    Config.DEFAULT_TICKER = ticker
    Config.update_forecast_horizon(forecast_horizon)
    
    # Fetch and prepare data
    print("\nFetching and preparing data...")
    data = fetch_stock_data(ticker, start_date, end_date, Config.DATA_INTERVAL)
    
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
        ticker,
        save_path=os.path.join(Config.OUTPUT_PATH, f'{ticker}_forecast_report.csv')
    )
    
    print("\n" + "="*70)
    print(" "*18 + "PREDICTION COMPLETED!")
    print("="*70)
    print(f"\nForecast report saved to: {Config.OUTPUT_PATH}")
    print("="*70 + "\n")


def main():
    """Main entry point for the stock forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description='Stock Price Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model for AAPL with 30-day forecast
  python main.py --mode train --ticker AAPL --forecast-days 30
  
  # Make predictions using trained model
  python main.py --mode predict --ticker AAPL --forecast-days 30
  
  # Train with custom date range
  python main.py --mode train --ticker MSFT --start 2019-01-01 --end 2024-12-31 --forecast-days 14
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        default='train',
        help='Mode: train a new model or predict with existing model'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default=Config.DEFAULT_TICKER,
        help=f'Stock ticker symbol (default: {Config.DEFAULT_TICKER})'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=Config.START_DATE,
        help=f'Start date in YYYY-MM-DD format (default: {Config.START_DATE})'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=Config.END_DATE,
        help=f'End date in YYYY-MM-DD format (default: {Config.END_DATE})'
    )
    
    parser.add_argument(
        '--forecast-days',
        type=int,
        default=Config.FORECAST_HORIZON,
        help=f'Number of days to forecast (7-30) (default: {Config.FORECAST_HORIZON})'
    )
    
    args = parser.parse_args()
    
    # Validate forecast days
    if not (7 <= args.forecast_days <= 30):
        print("Error: forecast-days must be between 7 and 30")
        return
    
    # Run appropriate mode
    if args.mode == 'train':
        train_model(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            forecast_horizon=args.forecast_days
        )
    else:
        predict_with_trained_model(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            forecast_horizon=args.forecast_days
        )


if __name__ == "__main__":
    main()
