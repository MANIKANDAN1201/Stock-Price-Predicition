# 📚 Usage Examples

## 🌐 Web Interface Examples

### Example 1: Quick Start with Popular Stocks

```bash
# 1. Install Streamlit
pip install streamlit

# 2. Train models for tech giants
python batch_train.py --tickers AAPL MSFT GOOGL META NVDA

# 3. Launch web app
streamlit run app.py
```

**What you'll see:**
- Beautiful web interface at `http://localhost:8501`
- Sidebar with ticker input and forecast slider
- List of available trained models
- Interactive charts and metrics

### Example 2: Using the Web App

1. **Enter ticker:** Type `AAPL` in the sidebar
2. **Set forecast:** Move slider to `30 days`
3. **Click:** "🔮 Generate Forecast" button
4. **View results:**
   - Current price: $175.43
   - 7-day forecast: $178.21 (+1.58%)
   - 14-day forecast: $180.45 (+2.86%)
   - 30-day forecast: $182.67 (+4.13%)
5. **Download:** Click "📥 Download Forecast as CSV"

## 📊 Batch Training Examples

### Example 3: Train All Popular Stocks

```bash
# Train 15 popular stocks with default settings
python batch_train.py
```

**Stocks trained:**
- Tech: AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, INTC, NFLX
- Finance: JPM, V
- Consumer: WMT, PG, DIS
- Healthcare: JNJ

**Output:**
```
==================================================
 BATCH TRAINING - STOCK PRICE FORECASTING
==================================================

Tickers to train: AAPL, MSFT, GOOGL, META, ...
Date range: 2018-01-01 to 2025-01-01
Forecast horizon: 30 days
Epochs per model: 100
Device: cuda

Progress: 1/15
==================================================
 TRAINING MODEL FOR AAPL
==================================================
...
✅ AAPL MODEL TRAINED SUCCESSFULLY
RMSE: 0.023456 | MAE: 0.018234 | MAPE: 2.34%
```

### Example 4: Train Specific Stocks

```bash
# Train only tech stocks
python batch_train.py --tickers AAPL MSFT GOOGL META NVDA TSLA

# Train with custom settings
python batch_train.py \
  --tickers AAPL MSFT \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --forecast-days 14 \
  --epochs 50
```

### Example 5: Quick Testing Mode

```bash
# Fast training for testing (20 epochs instead of 100)
python batch_train.py --tickers AAPL --quick
```

**Use cases:**
- Testing the pipeline
- Quick model validation
- Development/debugging

## 💻 Command Line Examples

### Example 6: Single Stock Training

```bash
# Train Apple with 30-day forecast
python main.py --mode train --ticker AAPL --forecast-days 30

# Train Tesla with 7-day forecast
python main.py --mode train --ticker TSLA --forecast-days 7

# Train Microsoft with custom date range
python main.py \
  --mode train \
  --ticker MSFT \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --forecast-days 14
```

### Example 7: Making Predictions

```bash
# Predict AAPL for next 30 days
python main.py --mode predict --ticker AAPL --forecast-days 30

# Predict MSFT for next 14 days
python main.py --mode predict --ticker MSFT --forecast-days 14
```

## 🔧 Advanced Examples

### Example 8: Custom Configuration

Edit `stock_forecaster/config.py`:

```python
# Increase model capacity
HIDDEN_SIZE = 256  # Default: 128
NUM_LAYERS = 4     # Default: 3

# Longer lookback
LOOKBACK_WINDOW = 90  # Default: 60

# More training
EPOCHS = 150  # Default: 100
```

Then train:
```bash
python main.py --mode train --ticker AAPL --forecast-days 30
```

### Example 9: Different Forecast Horizons

```bash
# Short-term (1 week)
python batch_train.py --tickers AAPL --forecast-days 7

# Medium-term (2 weeks)
python batch_train.py --tickers AAPL --forecast-days 14

# Long-term (1 month)
python batch_train.py --tickers AAPL --forecast-days 30
```

### Example 10: Historical Data Ranges

```bash
# Maximum historical data (if available)
python batch_train.py \
  --tickers AAPL \
  --start 2010-01-01 \
  --end 2024-12-31

# Recent data only (last 2 years)
python batch_train.py \
  --tickers AAPL \
  --start 2023-01-01 \
  --end 2024-12-31

# Specific period (e.g., post-COVID)
python batch_train.py \
  --tickers AAPL \
  --start 2020-03-01 \
  --end 2024-12-31
```

## 📈 Real-World Scenarios

### Scenario 1: Portfolio Tracking

```bash
# Train models for your portfolio
python batch_train.py --tickers AAPL MSFT GOOGL AMZN TSLA

# Launch app and check each stock
streamlit run app.py
```

### Scenario 2: Sector Analysis

```bash
# Tech sector
python batch_train.py --tickers AAPL MSFT GOOGL META NVDA INTC

# Finance sector
python batch_train.py --tickers JPM BAC GS MS C WFC

# Healthcare sector
python batch_train.py --tickers JNJ PFE UNH ABBV MRK
```

### Scenario 3: Weekly Updates

```bash
# Retrain models weekly with latest data
python batch_train.py --tickers AAPL MSFT GOOGL --epochs 50

# Quick check in web app
streamlit run app.py
```

### Scenario 4: Comparison Study

```bash
# Train same stocks with different horizons
python batch_train.py --tickers AAPL --forecast-days 7
python batch_train.py --tickers AAPL --forecast-days 14
python batch_train.py --tickers AAPL --forecast-days 30

# Compare results in output files
```

## 🎯 Workflow Examples

### Complete Workflow 1: New User

```bash
# Day 1: Setup
pip install -r requirements.txt

# Day 1: Train first model
python batch_train.py --tickers AAPL --quick

# Day 1: Test web app
streamlit run app.py

# Day 2: Train more stocks
python batch_train.py --tickers MSFT GOOGL META

# Day 3: Regular usage
streamlit run app.py
```

### Complete Workflow 2: Daily Trader

```bash
# Morning: Check forecasts
streamlit run app.py

# Weekly: Update models
python batch_train.py --tickers AAPL MSFT GOOGL --epochs 50

# Monthly: Full retrain
python batch_train.py
```

### Complete Workflow 3: Research

```bash
# Train with different parameters
python batch_train.py --tickers AAPL --start 2015-01-01 --epochs 150
python batch_train.py --tickers AAPL --start 2020-01-01 --epochs 150

# Compare metrics in data/output/AAPL/metrics.txt

# Analyze in web app
streamlit run app.py
```

## 📊 Output Examples

### Web App Output

**Metrics Display:**
```
Current Price: $175.43
7-Day Forecast: $178.21 (+1.58%)
14-Day Forecast: $180.45 (+2.86%)
30-Day Forecast: $182.67 (+4.13%)
```

**CSV Download:**
```csv
Date,Day,Predicted_Price,Change_from_Last,Change_Percent
2025-01-14,1,176.23,+0.80,+0.46%
2025-01-15,2,176.89,+1.46,+0.83%
2025-01-16,3,177.45,+2.02,+1.15%
...
```

### Command Line Output

**Training:**
```
[STEP 1/5] Fetching stock data...
Successfully fetched 1567 rows of data

[STEP 2/5] Feature engineering...
Total features created: 14

[STEP 3/5] Creating data loaders...
Training batches: 36

[STEP 4/5] Training model...
Epoch [  5/100] | Train Loss: 0.003456 | Val Loss: 0.003789
Epoch [ 10/100] | Train Loss: 0.002345 | Val Loss: 0.002567
...

[STEP 5/5] Evaluating model...
RMSE: 0.023456
MAE: 0.018234
MAPE: 2.34%
Directional Accuracy: 62.45%
```

## 🚀 Pro Tips

1. **Start small:** Train 1-2 stocks first to test
2. **Use quick mode:** For testing and development
3. **GPU acceleration:** Automatically used if available
4. **Regular updates:** Retrain monthly for best results
5. **Multiple horizons:** Train different forecast periods
6. **Batch processing:** Train multiple stocks overnight
7. **Web interface:** Easiest way to use the system
8. **CSV exports:** Save forecasts for analysis

## ⚠️ Common Mistakes

❌ **Don't:**
- Train without enough data (< 2 years)
- Use very old data only
- Expect 100% accuracy
- Make financial decisions solely on predictions

✅ **Do:**
- Use 3-5 years of historical data
- Retrain models regularly
- Use predictions as one of many indicators
- Understand model limitations

## 📞 Need Help?

Check these files:
- `STREAMLIT_GUIDE.md` - Web app details
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide

Happy forecasting! 📈
