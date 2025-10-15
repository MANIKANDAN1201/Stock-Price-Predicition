# 🔬 ML Pipeline Guide - Proper Data Organization

## 📁 New Directory Structure

```
StockPrice/
├── data/
│   ├── raw/                    # Raw fetched data
│   │   ├── AAPL/
│   │   │   ├── AAPL_raw_2000-01-01_to_2025-01-01_20250113.csv
│   │   │   └── AAPL_metadata_20250113.json
│   │   ├── MSFT/
│   │   └── ...
│   │
│   ├── processed/              # Preprocessed data with features
│   │   ├── AAPL/
│   │   │   ├── AAPL_processed_20250113.csv
│   │   │   ├── AAPL_scaler_20250113.pkl
│   │   │   ├── AAPL_features_20250113.json
│   │   │   ├── AAPL_splits_20250113.npz
│   │   │   └── AAPL_split_info_20250113.json
│   │   ├── MSFT/
│   │   └── ...
│   │
│   └── output/                 # Model results and predictions
│       ├── AAPL/
│       │   ├── metrics.txt
│       │   ├── data_report.txt
│       │   └── AAPL_predictions_20250113.csv
│       ├── MSFT/
│       └── ...
│
├── models/                     # Trained models
│   ├── AAPL_forecaster.pth
│   ├── AAPL_scaler.pkl
│   └── ...
│
└── plots/                      # Visualizations
```

## 🎯 Key Improvements

### 1. **25 Years of Historical Data**
```python
START_DATE = "2000-01-01"  # Changed from 2018-01-01
END_DATE = "2025-01-01"
```

**Benefits:**
- More training data = better model generalization
- Captures multiple market cycles (bull, bear, crashes, recoveries)
- Better understanding of long-term patterns

### 2. **Organized Data Storage**

**Raw Data:**
- Original fetched data from Yahoo Finance
- Includes metadata (date range, rows, columns)
- Timestamped for version control

**Processed Data:**
- Engineered features (MA, RSI, MACD, etc.)
- Scaled/normalized data
- Train/val/test splits
- Feature statistics

**Output:**
- Model predictions
- Performance metrics
- Comprehensive reports

### 3. **Prediction Validation**

The new pipeline includes automatic validation:

```python
def validate_predictions(predictions, actuals, ticker, current_price):
    """Check for prediction anomalies."""
    
    # Calculate error metrics
    mape = mean_absolute_percentage_error(predictions, actuals)
    
    # Sanity checks
    if mape > 15%:
        WARNING: "High error rate"
    
    if prediction_range > 50%:
        WARNING: "Unrealistic predictions"
```

### 4. **Data Quality Checks**

**Minimum Data Requirements:**
- At least 500 trading days
- No excessive missing values
- Valid price ranges

**Automatic Handling:**
- Missing values filled
- Outliers detected
- Data normalized properly

## 🚀 Usage

### Enhanced Training Script

```bash
# Train with 25 years of data
python batch_train_v2.py --tickers AAPL

# Train multiple stocks
python batch_train_v2.py --tickers AAPL MSFT GOOGL

# Quick test mode
python batch_train_v2.py --tickers AAPL --quick

# Custom date range
python batch_train_v2.py --tickers AAPL --start 2010-01-01 --end 2024-12-31
```

### What Gets Saved

**For each ticker (e.g., AAPL):**

1. **Raw Data** (`data/raw/AAPL/`)
   - `AAPL_raw_2000-01-01_to_2025-01-01_timestamp.csv`
   - `AAPL_metadata_timestamp.json`

2. **Processed Data** (`data/processed/AAPL/`)
   - `AAPL_processed_timestamp.csv` - With all features
   - `AAPL_scaler_timestamp.pkl` - Fitted scaler
   - `AAPL_features_timestamp.json` - Feature statistics
   - `AAPL_splits_timestamp.npz` - Train/val/test arrays
   - `AAPL_split_info_timestamp.json` - Split metadata

3. **Results** (`data/output/AAPL/`)
   - `metrics.txt` - RMSE, MAE, MAPE, accuracy
   - `data_report.txt` - Comprehensive analysis
   - `AAPL_predictions_timestamp.csv` - All predictions

4. **Models** (`models/`)
   - `AAPL_forecaster.pth` - Trained model
   - `AAPL_scaler.pkl` - For app compatibility

## 📊 Understanding the Output

### Data Report Example

```
================================================================================
                          DATA REPORT FOR AAPL
================================================================================

DATA INFORMATION
--------------------------------------------------------------------------------
Total rows: 6,289
Date range: 2000-01-03 to 2024-12-31
Trading days: 6,289

PRICE STATISTICS
--------------------------------------------------------------------------------
Current price: $245.32
Highest price: $237.23
Lowest price: $0.79
Average price: $89.45
Price volatility (std): $67.32

MODEL PERFORMANCE
--------------------------------------------------------------------------------
RMSE: 0.023456
MAE: 0.018234
MAPE: 2.34%
Directional Accuracy: 62.45%

================================================================================
```

### Validation Warnings

The system will warn you if:

1. **High MAPE (>15%)**
   - Predictions have large percentage errors
   - May need more training or better features

2. **High MAE (>0.2 in scaled space)**
   - Large absolute errors
   - Check data quality

3. **Large Prediction Range**
   - Predictions vary too much
   - Model may be unstable

## 🔍 Why Predictions Were Wrong

### Previous Issue: Predicting $201 when current is $245

**Causes:**
1. ❌ Only 5 years of data (2018-2025)
2. ❌ No validation checks
3. ❌ Scaling issues not detected
4. ❌ Model overfitting to recent data

**Solutions Applied:**
1. ✅ 25 years of data (2000-2025)
2. ✅ Automatic validation with warnings
3. ✅ Better data organization and tracking
4. ✅ Comprehensive error metrics
5. ✅ Data quality checks

## 📈 Expected Performance

With 25 years of data and proper pipeline:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| MAPE | <5% | 5-10% | >10% |
| MAE (scaled) | <0.05 | 0.05-0.15 | >0.15 |
| Directional Accuracy | >60% | 50-60% | <50% |

## 🛠️ Troubleshooting

### Issue: Still Getting Bad Predictions

**Check:**
1. Data quality in `data/raw/<ticker>/`
2. Feature statistics in `data/processed/<ticker>/`
3. Validation warnings in training output
4. Metrics in `data/output/<ticker>/metrics.txt`

**Solutions:**
```bash
# Retrain with more epochs
python batch_train_v2.py --tickers AAPL --epochs 150

# Check data report
cat data/output/AAPL/data_report.txt

# Verify current price
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['currentPrice'])"
```

### Issue: Insufficient Data

**Error:** "Insufficient data for AAPL (only 234 rows)"

**Solutions:**
1. Use longer date range: `--start 1990-01-01`
2. Check if ticker is valid
3. Verify internet connection

### Issue: High MAPE Warning

**Warning:** "⚠️ High MAPE: 18.45%"

**Solutions:**
1. Train longer: `--epochs 200`
2. Adjust model size in `config.py`
3. Add more features
4. Check for data quality issues

## 📚 File Formats

### Raw Data CSV
```csv
Date,Open,High,Low,Close,Volume
2000-01-03,3.74,4.00,3.65,3.99,133949200
2000-01-04,3.86,3.99,3.66,3.71,128094400
...
```

### Processed Data CSV
```csv
Date,Open,High,Low,Close,Volume,MA_7,MA_21,RSI,MACD,...
2000-02-15,3.45,3.67,3.42,3.65,95847200,3.58,3.62,52.3,0.02,...
...
```

### Predictions CSV
```csv
Prediction,Actual,Error,Absolute_Error,Percentage_Error
0.7542,0.8561,0.1019,0.1019,11.90
0.7612,0.8684,0.1072,0.1072,12.35
...
```

### Metadata JSON
```json
{
  "ticker": "AAPL",
  "start_date": "2000-01-01",
  "end_date": "2025-01-01",
  "rows": 6289,
  "columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
  "date_range": {
    "min": "2000-01-03",
    "max": "2024-12-31"
  },
  "saved_at": "20250113_180000"
}
```

## 🎯 Best Practices

### 1. Data Collection
- ✅ Use maximum available history (25+ years)
- ✅ Verify data quality before training
- ✅ Save raw data for reproducibility

### 2. Preprocessing
- ✅ Document all transformations
- ✅ Save scalers and feature info
- ✅ Track data splits

### 3. Training
- ✅ Monitor validation metrics
- ✅ Check for warnings
- ✅ Save comprehensive reports

### 4. Validation
- ✅ Review prediction sanity
- ✅ Compare with current prices
- ✅ Check error distributions

### 5. Deployment
- ✅ Use validated models only
- ✅ Monitor predictions in production
- ✅ Retrain regularly (monthly)

## 🔄 Retraining Workflow

```bash
# Monthly retraining
python batch_train_v2.py --tickers AAPL MSFT GOOGL

# Check results
cat data/output/AAPL/data_report.txt

# If good, deploy
streamlit run app.py

# If warnings, investigate
python -c "import pandas as pd; df = pd.read_csv('data/output/AAPL/AAPL_predictions_*.csv'); print(df.describe())"
```

## 📞 Quick Commands

```bash
# Train with proper pipeline
python batch_train_v2.py --tickers AAPL

# Check data organization
tree data/

# View metrics
cat data/output/AAPL/metrics.txt

# View full report
cat data/output/AAPL/data_report.txt

# Check raw data
head data/raw/AAPL/*.csv

# Check processed features
head data/processed/AAPL/*_processed_*.csv

# Launch app
streamlit run app.py
```

## ✅ Summary

**Old Pipeline:**
- ❌ 5 years data
- ❌ No organization
- ❌ No validation
- ❌ Poor predictions

**New Pipeline:**
- ✅ 25 years data
- ✅ Organized structure
- ✅ Automatic validation
- ✅ Better predictions
- ✅ Full traceability
- ✅ Comprehensive reports

**Result:** More accurate, reliable, and maintainable ML system! 🚀
