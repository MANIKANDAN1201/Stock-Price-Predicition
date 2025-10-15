"""
Batch training script for multiple stock tickers.
Trains models for popular stocks and saves them for the Streamlit app.
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
from stock_forecaster.evaluator import ModelEvaluator
import pickle


# Popular stock tickers to train
POPULAR_TICKERS = [
    # Tech Giants
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Alphabet (Google)
    'META',   # Meta (Facebook)
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
    
    # Other Major Companies
    'JPM',    # JPMorgan Chase
    'V',      # Visa
    'WMT',    # Walmart
    'JNJ',    # Johnson & Johnson
    'PG',     # Procter & Gamble
    'DIS',    # Disney
    'NFLX',   # Netflix
    'INTC',   # Intel
]


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_single_ticker(ticker, start_date, end_date, forecast_horizon=30, epochs=100):
    """
    Train model for a single ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        forecast_horizon (int): Number of days to forecast
        epochs (int): Number of training epochs
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print(f" TRAINING MODEL FOR {ticker} ".center(80, "="))
    print("="*80 + "\n")
    
    try:
        # Set random seeds
        set_random_seeds(Config.RANDOM_SEED)
        
        # Step 1: Fetch data
        print(f"[1/5] Fetching data for {ticker}...")
        data = fetch_stock_data(ticker, start_date, end_date, Config.DATA_INTERVAL)
        
        if len(data) < 200:
            print(f"⚠️  Insufficient data for {ticker} (only {len(data)} rows). Skipping...")
            return False
        
        # Step 2: Feature engineering
        print(f"[2/5] Engineering features...")
        prepared_data = prepare_data_for_training(
            data,
            lookback=Config.LOOKBACK_WINDOW,
            forecast_horizon=forecast_horizon,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO
        )
        
        # Save scaler with ticker-specific name
        scaler_path = f"models/{ticker}_scaler.pkl"
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(prepared_data['scaler'], f)
        print(f"Scaler saved to {scaler_path}")
        
        # Step 3: Create data loaders
        print(f"[3/5] Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            prepared_data['X_train'],
            prepared_data['y_train'],
            prepared_data['X_val'],
            prepared_data['y_val'],
            Config.BATCH_SIZE
        )
        
        # Step 4: Train model
        print(f"[4/5] Training model...")
        input_size = prepared_data['X_train'].shape[2]
        
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            forecast_horizon=forecast_horizon,
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
            epochs=epochs,
            patience=Config.PATIENCE
        )
        
        # Save model with ticker-specific name
        model_path = f"models/{ticker}_forecaster.pth"
        trainer.save_model(model_path)
        
        # Step 5: Quick evaluation
        print(f"[5/5] Evaluating model...")
        X_test_tensor = torch.FloatTensor(prepared_data['X_test']).to(Config.DEVICE)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor).cpu().numpy()
        
        evaluator = ModelEvaluator(
            predictions=test_predictions,
            actuals=prepared_data['y_test'],
            ticker=ticker
        )
        
        metrics = evaluator.calculate_all_metrics()
        
        # Save metrics
        output_dir = f"data/output/{ticker}"
        os.makedirs(output_dir, exist_ok=True)
        evaluator.save_metrics_to_file(f"{output_dir}/metrics.txt")
        
        print("\n" + "="*80)
        print(f" ✅ {ticker} MODEL TRAINED SUCCESSFULLY ".center(80, "="))
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Scaler: {scaler_path}")
        print(f"RMSE: {metrics['RMSE']:.6f} | MAE: {metrics['MAE']:.6f} | MAPE: {metrics['MAPE']:.2f}%")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error training {ticker}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main batch training function."""
    parser = argparse.ArgumentParser(
        description='Batch train stock forecasting models for multiple tickers'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help='List of tickers to train (default: popular stocks)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2018-01-01',
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2025-01-01',
        help='End date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--forecast-days',
        type=int,
        default=30,
        help='Number of days to forecast (7-30)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode (fewer epochs for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine tickers to train
    tickers = args.tickers if args.tickers else POPULAR_TICKERS
    
    # Adjust epochs for quick mode
    epochs = 20 if args.quick else args.epochs
    
    print("\n" + "="*80)
    print(" BATCH TRAINING - STOCK PRICE FORECASTING ".center(80, "="))
    print("="*80)
    print(f"\nTickers to train: {', '.join(tickers)}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Forecast horizon: {args.forecast_days} days")
    print(f"Epochs per model: {epochs}")
    print(f"Device: {Config.DEVICE}")
    if args.quick:
        print("⚡ Quick mode enabled (reduced epochs)")
    print("\n" + "="*80 + "\n")
    
    # Train each ticker
    results = {}
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*80}")
        print(f" Progress: {i}/{len(tickers)} ".center(80, "="))
        print(f"{'='*80}\n")
        
        success = train_single_ticker(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            forecast_horizon=args.forecast_days,
            epochs=epochs
        )
        
        results[ticker] = success
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print(" BATCH TRAINING COMPLETE ".center(80, "="))
    print("="*80)
    print(f"\nTotal tickers: {len(tickers)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    print("\n📊 Detailed Results:")
    for ticker, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {ticker:8s} - {status}")
    
    print("\n" + "="*80)
    print("\n🚀 You can now use the Streamlit app:")
    print("   streamlit run app.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
