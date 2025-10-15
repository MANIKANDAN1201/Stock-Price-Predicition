"""
Enhanced batch training script with proper ML pipeline and data organization.
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add stock_forecaster to path
sys.path.append(os.path.dirname(__file__))

from stock_forecaster.config import Config
from stock_forecaster.data_fetcher import fetch_stock_data
from stock_forecaster.feature_engineering import prepare_data_for_training
from stock_forecaster.model_trainer import LSTMForecaster, ModelTrainer, create_data_loaders
from stock_forecaster.evaluator import ModelEvaluator
from stock_forecaster.data_pipeline import DataPipeline, create_data_report
import pickle


# Popular stock tickers
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA',
    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'DIS', 'NFLX', 'INTC',
]


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_predictions(predictions, actuals, ticker, current_price):
    """
    Validate predictions for sanity checks.
    
    Args:
        predictions (np.ndarray): Model predictions (scaled)
        actuals (np.ndarray): Actual values (scaled)
        ticker (str): Stock ticker
        current_price (float): Current actual stock price
    
    Returns:
        dict: Validation results
    """
    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    # Check for anomalies
    warnings = []
    
    if mape > 15:
        warnings.append(f"⚠️  High MAPE: {mape:.2f}% (predictions may be unreliable)")
    
    if mae > 0.2:
        warnings.append(f"⚠️  High MAE: {mae:.4f} (large prediction errors)")
    
    # Check prediction range
    pred_min, pred_max = predictions.min(), predictions.max()
    if pred_max - pred_min > 0.5:
        warnings.append(f"⚠️  Large prediction range: {pred_min:.3f} to {pred_max:.3f}")
    
    validation = {
        'mae': mae,
        'mape': mape,
        'warnings': warnings,
        'passed': len(warnings) == 0
    }
    
    return validation


def train_single_ticker_v2(ticker, start_date, end_date, forecast_horizon=30, epochs=100):
    """
    Enhanced training with proper data pipeline.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        forecast_horizon (int): Number of days to forecast
        epochs (int): Number of training epochs
    
    Returns:
        dict: Training results with validation info
    """
    print("\n" + "="*80)
    print(f" TRAINING MODEL FOR {ticker} ".center(80, "="))
    print("="*80 + "\n")
    
    try:
        # Initialize data pipeline
        pipeline = DataPipeline(ticker, Config.DATA_BASE_PATH)
        
        # Set random seeds
        set_random_seeds(Config.RANDOM_SEED)
        
        # Step 1: Fetch data
        print(f"[1/7] Fetching data for {ticker}...")
        print(f"      Date range: {start_date} to {end_date}")
        data = fetch_stock_data(ticker, start_date, end_date, Config.DATA_INTERVAL)
        
        if len(data) < 500:
            print(f"⚠️  Insufficient data for {ticker} (only {len(data)} rows). Need at least 500.")
            print(f"   Try using a longer date range or check if ticker is valid.")
            return None
        
        # Save raw data
        print(f"\n[2/7] Saving raw data...")
        pipeline.save_raw_data(data, start_date, end_date)
        
        # Get current price for validation
        current_price = float(data['Close'].iloc[-1])
        print(f"      Current {ticker} price: ${current_price:.2f}")
        
        # Step 3: Feature engineering
        print(f"\n[3/7] Engineering features and preprocessing...")
        prepared_data = prepare_data_for_training(
            data,
            lookback=Config.LOOKBACK_WINDOW,
            forecast_horizon=forecast_horizon,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO
        )
        
        # Save processed data
        print(f"\n[4/7] Saving processed data...")
        pipeline.save_processed_data(
            prepared_data['engineered_data'],
            prepared_data['feature_columns'],
            prepared_data['scaler']
        )
        
        # Save train/val/test splits
        pipeline.save_training_results(
            prepared_data['X_train'],
            prepared_data['y_train'],
            prepared_data['X_val'],
            prepared_data['y_val'],
            prepared_data['X_test'],
            prepared_data['y_test']
        )
        
        # Save scaler to models directory (for app compatibility)
        scaler_path = f"models/{ticker}_scaler.pkl"
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(prepared_data['scaler'], f)
        
        # Step 5: Create data loaders
        print(f"\n[5/7] Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            prepared_data['X_train'],
            prepared_data['y_train'],
            prepared_data['X_val'],
            prepared_data['y_val'],
            Config.BATCH_SIZE
        )
        
        # Step 6: Train model
        print(f"\n[6/7] Training model...")
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
        
        # Save model
        model_path = f"models/{ticker}_forecaster.pth"
        trainer.save_model(model_path)
        
        # Step 7: Evaluate and validate
        print(f"\n[7/7] Evaluating model and validating predictions...")
        X_test_tensor = torch.FloatTensor(prepared_data['X_test']).to(Config.DEVICE)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor).cpu().numpy()
        
        # Validate predictions
        validation = validate_predictions(
            test_predictions,
            prepared_data['y_test'],
            ticker,
            current_price
        )
        
        # Calculate metrics
        evaluator = ModelEvaluator(
            predictions=test_predictions,
            actuals=prepared_data['y_test'],
            ticker=ticker
        )
        
        metrics = evaluator.calculate_all_metrics()
        
        # Save predictions
        pipeline.save_predictions(test_predictions, prepared_data['y_test'])
        
        # Save metrics
        evaluator.save_metrics_to_file(
            os.path.join(pipeline.output_dir, 'metrics.txt')
        )
        
        # Create comprehensive report
        report_path = os.path.join(pipeline.output_dir, 'data_report.txt')
        create_data_report(ticker, data, metrics, report_path)
        
        # Display results
        print("\n" + "="*80)
        print(f" ✅ {ticker} MODEL TRAINED SUCCESSFULLY ".center(80, "="))
        print("="*80)
        print(f"\n📊 DATA SUMMARY:")
        print(f"   Total data points: {len(data)}")
        print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"   Current price: ${current_price:.2f}")
        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   RMSE: {metrics['RMSE']:.6f}")
        print(f"   MAE: {metrics['MAE']:.6f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        print(f"\n✓ VALIDATION:")
        if validation['passed']:
            print(f"   Status: PASSED ✓")
        else:
            print(f"   Status: WARNINGS ⚠️")
            for warning in validation['warnings']:
                print(f"   {warning}")
        print(f"\n📁 FILES SAVED:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Raw data: {pipeline.raw_dir}")
        print(f"   Processed data: {pipeline.processed_dir}")
        print(f"   Results: {pipeline.output_dir}")
        print("="*80 + "\n")
        
        return {
            'success': True,
            'ticker': ticker,
            'metrics': metrics,
            'validation': validation,
            'data_points': len(data),
            'current_price': current_price
        }
        
    except Exception as e:
        print(f"\n❌ Error training {ticker}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }


def main():
    """Main batch training function with enhanced pipeline."""
    parser = argparse.ArgumentParser(
        description='Enhanced batch training with proper ML pipeline'
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
        default='2000-01-01',
        help='Start date in YYYY-MM-DD format (default: 2000-01-01 for 25 years)'
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
    
    # Determine tickers
    tickers = args.tickers if args.tickers else POPULAR_TICKERS
    epochs = 20 if args.quick else args.epochs
    
    print("\n" + "="*80)
    print(" ENHANCED BATCH TRAINING - ML PIPELINE ".center(80, "="))
    print("="*80)
    print(f"\n📊 CONFIGURATION:")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Date range: {args.start} to {args.end}")
    print(f"   Forecast horizon: {args.forecast_days} days")
    print(f"   Epochs: {epochs}")
    print(f"   Device: {Config.DEVICE}")
    if args.quick:
        print(f"   Mode: ⚡ Quick (testing)")
    print(f"\n📁 DATA ORGANIZATION:")
    print(f"   Raw data: data/raw/<ticker>/")
    print(f"   Processed data: data/processed/<ticker>/")
    print(f"   Results: data/output/<ticker>/")
    print("\n" + "="*80 + "\n")
    
    # Train each ticker
    results = []
    successful = 0
    failed = 0
    warnings_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*80}")
        print(f" Progress: {i}/{len(tickers)} ".center(80, "="))
        print(f"{'='*80}\n")
        
        result = train_single_ticker_v2(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            forecast_horizon=args.forecast_days,
            epochs=epochs
        )
        
        if result:
            results.append(result)
            if result['success']:
                successful += 1
                if not result.get('validation', {}).get('passed', True):
                    warnings_count += 1
            else:
                failed += 1
    
    # Final summary
    print("\n" + "="*80)
    print(" BATCH TRAINING COMPLETE ".center(80, "="))
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"   Total tickers: {len(tickers)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ⚠️  With warnings: {warnings_count}")
    print(f"   ❌ Failed: {failed}")
    
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        if result['success']:
            ticker = result['ticker']
            metrics = result['metrics']
            validation = result['validation']
            
            status = "✅" if validation['passed'] else "⚠️ "
            print(f"  {ticker:8s} - {status} | MAPE: {metrics['MAPE']:5.2f}% | "
                  f"Data: {result['data_points']:,} points | "
                  f"Price: ${result['current_price']:.2f}")
        else:
            print(f"  {result['ticker']:8s} - ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("\n🚀 NEXT STEPS:")
    print("   1. Review warnings for any models with ⚠️  status")
    print("   2. Check data/output/<ticker>/ for detailed reports")
    print("   3. Launch Streamlit app: streamlit run app.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
