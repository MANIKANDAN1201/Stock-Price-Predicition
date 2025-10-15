# 🚀 Streamlit Web App Guide

## Overview

The Stock Price Forecaster now includes a beautiful web interface built with Streamlit that allows you to:
- ✅ Select any trained stock ticker
- ✅ Generate real-time forecasts (7-30 days)
- ✅ View interactive visualizations
- ✅ Download forecast results as CSV
- ✅ See detailed metrics and model information

## 📦 Installation

Install the additional dependency:

```bash
pip install streamlit>=1.28.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Step 1: Train Models for Popular Stocks

Train models for multiple popular stocks at once:

```bash
# Train all popular stocks (15 tickers)
python batch_train.py

# Train specific stocks
python batch_train.py --tickers AAPL MSFT GOOGL META TSLA

# Quick training mode (fewer epochs for testing)
python batch_train.py --tickers AAPL MSFT --quick

# Custom date range
python batch_train.py --tickers AAPL --start 2020-01-01 --end 2024-12-31
```

**Popular stocks included:**
- **Tech Giants:** AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA
- **Finance:** JPM, V
- **Consumer:** WMT, PG, DIS
- **Healthcare:** JNJ
- **Entertainment:** NFLX
- **Semiconductors:** INTC

### Step 2: Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎨 Using the Web Interface

### Main Features

1. **Ticker Selection**
   - Enter any stock ticker in the sidebar (e.g., AAPL, MSFT, GOOGL)
   - The app shows which models are available

2. **Forecast Horizon**
   - Use the slider to select 7-30 days forecast
   - Default is 30 days

3. **Generate Forecast**
   - Click "🔮 Generate Forecast" button
   - The app will:
     - Load the trained model
     - Fetch latest stock data
     - Generate predictions
     - Display visualizations

4. **View Results**
   - **Key Metrics:** Current price, 7-day, 14-day, and final forecasts
   - **Interactive Chart:** Historical prices + forecasted prices
   - **Detailed Table:** Day-by-day predictions with percentage changes
   - **Download CSV:** Export forecast data

5. **Model Information**
   - Expand "ℹ️ Model Information" to see:
     - Model architecture details
     - Training loss
     - Device used (CPU/GPU)

## 📊 Understanding the Output

### Forecast Values (0-1 Range)

The predictions are in **normalized (scaled) values** between 0 and 1. The app automatically converts them back to actual dollar prices for display.

**Why normalized?**
- Better training stability
- Faster convergence
- Works across different price ranges

**Conversion happens automatically:**
```python
# Scaled prediction (0-1) → Actual price ($)
scaled_price = 0.75420976
actual_price = $856.11  # After inverse transformation
```

### Metrics Explained

- **Current Price:** Last known closing price
- **7/14/30-Day Forecast:** Predicted price at that day
- **Change %:** Percentage change from current price
- **Green/Red arrows:** Indicate price increase/decrease

## 🔧 Batch Training Options

### Basic Usage

```bash
# Train default popular stocks
python batch_train.py
```

### Advanced Options

```bash
python batch_train.py \
  --tickers AAPL MSFT GOOGL \     # Specific tickers
  --start 2019-01-01 \            # Start date
  --end 2024-12-31 \              # End date
  --forecast-days 14 \            # Forecast horizon
  --epochs 50 \                   # Training epochs
  --quick                         # Quick mode (20 epochs)
```

### Training Output

For each ticker, you'll see:
```
==================================================
 TRAINING MODEL FOR AAPL 
==================================================

[1/5] Fetching data for AAPL...
[2/5] Engineering features...
[3/5] Creating data loaders...
[4/5] Training model...
[5/5] Evaluating model...

✅ AAPL MODEL TRAINED SUCCESSFULLY
Model: models/AAPL_forecaster.pth
Scaler: models/AAPL_scaler.pkl
RMSE: 0.023456 | MAE: 0.018234 | MAPE: 2.34%
```

## 📁 File Structure After Training

```
StockPrice/
├── models/
│   ├── AAPL_forecaster.pth      # Trained model
│   ├── AAPL_scaler.pkl          # Data scaler
│   ├── MSFT_forecaster.pth
│   ├── MSFT_scaler.pkl
│   └── ...
│
├── data/output/
│   ├── AAPL/
│   │   └── metrics.txt          # Evaluation metrics
│   ├── MSFT/
│   │   └── metrics.txt
│   └── ...
│
└── app.py                        # Streamlit app
```

## 🎯 Example Workflow

### Complete End-to-End Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models for tech stocks
python batch_train.py --tickers AAPL MSFT GOOGL META NVDA

# 3. Launch web app
streamlit run app.py

# 4. In the browser:
#    - Select ticker: AAPL
#    - Set forecast: 30 days
#    - Click "Generate Forecast"
#    - Download results as CSV
```

## 🔍 Troubleshooting

### "No trained model found for {ticker}"

**Solution:** Train the model first
```bash
python batch_train.py --tickers {TICKER}
```

### "Failed to fetch or prepare data"

**Possible causes:**
- Internet connection issue
- Invalid ticker symbol
- Market closed (try again during market hours)

**Solution:** Check ticker symbol and internet connection

### App is slow

**Solutions:**
- Use GPU if available (automatic)
- Reduce forecast horizon (7-14 days instead of 30)
- Train with fewer epochs in quick mode

### Model predictions seem off

**Solutions:**
- Retrain with more recent data
- Increase training epochs
- Use longer historical data period

## 📈 Tips for Best Results

1. **Training Data**
   - Use at least 3-5 years of historical data
   - More data = better model generalization

2. **Forecast Horizon**
   - Shorter horizons (7-14 days) are more accurate
   - Longer horizons (21-30 days) have higher uncertainty

3. **Model Updates**
   - Retrain models monthly for best accuracy
   - Market conditions change over time

4. **Multiple Stocks**
   - Train models for stocks you frequently track
   - Each stock has unique patterns

## 🚀 Advanced Features

### Custom Model Training

Edit `stock_forecaster/config.py` to customize:

```python
LOOKBACK_WINDOW = 60      # Days to look back
HIDDEN_SIZE = 128         # Model capacity
NUM_LAYERS = 3            # Model depth
EPOCHS = 100              # Training duration
```

### API Integration (Future)

The predictor can be wrapped in a REST API:
```python
# Coming soon: REST API endpoint
POST /predict
{
  "ticker": "AAPL",
  "forecast_days": 30
}
```

## ⚠️ Important Notes

1. **Predictions are not financial advice**
   - Use for educational purposes only
   - Always consult financial professionals

2. **Market volatility**
   - Unexpected events can affect accuracy
   - Models learn from historical patterns

3. **Data quality**
   - Predictions depend on data quality
   - Yahoo Finance data is free but may have gaps

## 📞 Support

For issues or questions:
1. Check this guide
2. Review error messages
3. Verify model files exist
4. Ensure internet connection

## 🎉 Enjoy!

You now have a complete AI-powered stock forecasting system with a beautiful web interface!

```bash
streamlit run app.py
```

Happy forecasting! 📈
