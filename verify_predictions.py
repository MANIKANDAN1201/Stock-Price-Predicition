"""
Script to verify model predictions and check for accuracy issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(__file__))

from stock_forecaster.config import Config


def get_current_price(ticker):
    """Get current stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        return float(current_price)
    except:
        # Fallback: get from recent data
        data = yf.download(ticker, period='1d', progress=False)
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None


def check_prediction_sanity(ticker, model_path=None):
    """
    Check if model predictions are reasonable.
    
    Args:
        ticker (str): Stock ticker
        model_path (str): Path to model file
    """
    print("\n" + "="*80)
    print(f" PREDICTION VERIFICATION FOR {ticker} ".center(80, "="))
    print("="*80 + "\n")
    
    # Get current actual price
    print("📊 Fetching current market data...")
    current_price = get_current_price(ticker)
    
    if current_price is None:
        print(f"❌ Could not fetch current price for {ticker}")
        return
    
    print(f"✓ Current {ticker} price: ${current_price:.2f}")
    
    # Check if model exists
    model_file = model_path or f"models/{ticker}_forecaster.pth"
    scaler_file = f"models/{ticker}_scaler.pkl"
    
    if not os.path.exists(model_file):
        print(f"\n⚠️  Model not found: {model_file}")
        print(f"   Train the model first: python batch_train_v2.py --tickers {ticker}")
        return
    
    print(f"✓ Model found: {model_file}")
    
    # Load and check recent predictions
    output_dir = f"data/output/{ticker}"
    if os.path.exists(output_dir):
        pred_files = [f for f in os.listdir(output_dir) if f.startswith(f"{ticker}_predictions_")]
        
        if pred_files:
            latest_pred = sorted(pred_files)[-1]
            pred_path = os.path.join(output_dir, latest_pred)
            
            print(f"✓ Latest predictions: {latest_pred}")
            
            # Load predictions
            pred_df = pd.read_csv(pred_path)
            
            print(f"\n📈 PREDICTION STATISTICS:")
            print(f"   Total predictions: {len(pred_df)}")
            print(f"   Mean prediction: {pred_df['Prediction'].mean():.4f} (scaled)")
            print(f"   Mean actual: {pred_df['Actual'].mean():.4f} (scaled)")
            print(f"   Mean error: {pred_df['Error'].mean():.4f}")
            print(f"   Mean absolute error: {pred_df['Absolute_Error'].mean():.4f}")
            print(f"   Mean percentage error: {pred_df['Percentage_Error'].mean():.2f}%")
            
            # Check for anomalies
            print(f"\n🔍 SANITY CHECKS:")
            
            mape = pred_df['Percentage_Error'].mean()
            if mape > 15:
                print(f"   ⚠️  HIGH ERROR: MAPE = {mape:.2f}% (should be <10%)")
            else:
                print(f"   ✓ Error acceptable: MAPE = {mape:.2f}%")
            
            # Check prediction range
            pred_range = pred_df['Prediction'].max() - pred_df['Prediction'].min()
            if pred_range > 0.5:
                print(f"   ⚠️  Large prediction range: {pred_range:.3f} (scaled)")
            else:
                print(f"   ✓ Prediction range reasonable: {pred_range:.3f}")
    
    # Check data quality
    raw_dir = f"data/raw/{ticker}"
    if os.path.exists(raw_dir):
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if csv_files:
            latest_raw = sorted(csv_files)[-1]
            raw_path = os.path.join(raw_dir, latest_raw)
            
            raw_df = pd.read_csv(raw_path)
            
            # Convert Date to datetime
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            
            print(f"\n📁 DATA QUALITY:")
            print(f"   Total data points: {len(raw_df):,}")
            print(f"   Date range: {str(raw_df['Date'].min())[:10]} to {str(raw_df['Date'].max())[:10]}")
            
            # Calculate years of data
            start = raw_df['Date'].min()
            end = raw_df['Date'].max()
            years = (end - start).days / 365.25
            
            print(f"   Years of data: {years:.1f} years")
            
            if years < 5:
                print(f"   ⚠️  Limited data: {years:.1f} years (recommend 10+ years)")
            else:
                print(f"   ✓ Sufficient data: {years:.1f} years")
            
            # Check recent price
            latest_price = float(raw_df['Close'].iloc[-1])
            print(f"\n💰 PRICE COMPARISON:")
            print(f"   Latest in data: ${latest_price:.2f}")
            print(f"   Current market: ${current_price:.2f}")
            
            price_diff = abs(current_price - latest_price)
            price_diff_pct = (price_diff / current_price) * 100
            
            if price_diff_pct > 5:
                print(f"   ⚠️  Data may be outdated: {price_diff_pct:.1f}% difference")
                print(f"   → Retrain with latest data")
            else:
                print(f"   ✓ Data is recent: {price_diff_pct:.1f}% difference")
    
    # Load metrics
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    if os.path.exists(metrics_file):
        print(f"\n📊 MODEL METRICS:")
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line and '=' not in line:
                    print(f"   {line.strip()}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if years < 10:
        print(f"   1. Retrain with more data:")
        print(f"      python batch_train_v2.py --tickers {ticker} --start 2000-01-01")
    
    if mape > 10:
        print(f"   2. Improve model accuracy:")
        print(f"      python batch_train_v2.py --tickers {ticker} --epochs 150")
    
    if price_diff_pct > 5:
        print(f"   3. Update with latest data:")
        print(f"      python batch_train_v2.py --tickers {ticker}")
    
    if not any([years < 10, mape > 10, price_diff_pct > 5]):
        print(f"   ✓ Model looks good! Ready to use in app.")
    
    print("\n" + "="*80 + "\n")


def compare_all_models():
    """Compare all trained models."""
    print("\n" + "="*80)
    print(" MODEL COMPARISON ".center(80, "="))
    print("="*80 + "\n")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_forecaster.pth')]
    
    if not model_files:
        print("No trained models found.")
        return
    
    tickers = [f.replace('_forecaster.pth', '') for f in model_files]
    
    print(f"Found {len(tickers)} trained models:\n")
    
    results = []
    
    for ticker in tickers:
        current_price = get_current_price(ticker)
        
        # Load metrics
        metrics_file = f"data/output/{ticker}/metrics.txt"
        mape = None
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                for line in f:
                    if 'MAPE:' in line:
                        try:
                            mape = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
        
        # Check data
        raw_dir = f"data/raw/{ticker}"
        data_points = 0
        years = 0
        
        if os.path.exists(raw_dir):
            csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
            if csv_files:
                raw_df = pd.read_csv(os.path.join(raw_dir, csv_files[-1]))
                data_points = len(raw_df)
                start = pd.to_datetime(raw_df['Date'].min())
                end = pd.to_datetime(raw_df['Date'].max())
                years = (end - start).days / 365.25
        
        results.append({
            'ticker': ticker,
            'price': current_price,
            'mape': mape,
            'data_points': data_points,
            'years': years
        })
    
    # Display table
    print(f"{'Ticker':<8} {'Price':<10} {'MAPE':<10} {'Data Points':<12} {'Years':<8} {'Status'}")
    print("-" * 80)
    
    for r in results:
        ticker = r['ticker']
        price = f"${r['price']:.2f}" if r['price'] else "N/A"
        mape = f"{r['mape']:.2f}%" if r['mape'] else "N/A"
        data_points = f"{r['data_points']:,}" if r['data_points'] else "N/A"
        years_str = f"{r['years']:.1f}y" if r['years'] else "N/A"
        
        # Determine status
        status = "✓"
        if r['mape'] and r['mape'] > 10:
            status = "⚠️  High error"
        elif r['years'] and r['years'] < 10:
            status = "⚠️  Limited data"
        
        print(f"{ticker:<8} {price:<10} {mape:<10} {data_points:<12} {years_str:<8} {status}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify model predictions')
    parser.add_argument('--ticker', type=str, help='Ticker to verify')
    parser.add_argument('--all', action='store_true', help='Compare all models')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models()
    elif args.ticker:
        check_prediction_sanity(args.ticker)
    else:
        print("Usage:")
        print("  python verify_predictions.py --ticker AAPL")
        print("  python verify_predictions.py --all")
