# 🚀 Quick Start Guide

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Train Your First Model (AAPL - 30 days)

```bash
python main.py --mode train --ticker AAPL --forecast-days 30
```

This will:
- Download 5 years of AAPL data from Yahoo Finance
- Create 14 technical indicators
- Train a bidirectional LSTM model
- Generate evaluation metrics and plots
- Save the model to `models/stock_forecaster.pth`

**Expected runtime**: 5-15 minutes (depending on your hardware)

### 2. Make Predictions with Trained Model

```bash
python main.py --mode predict --ticker AAPL --forecast-days 30
```

## More Examples

### Short-term Forecast (7 days)
```bash
python main.py --mode train --ticker TSLA --forecast-days 7
```

### Medium-term Forecast (14 days)
```bash
python main.py --mode train --ticker MSFT --forecast-days 14
```

### Custom Date Range
```bash
python main.py --mode train --ticker GOOGL --start 2020-01-01 --end 2024-12-31 --forecast-days 21
```

## What Gets Generated

After training, check these folders:

- **`models/`** - Trained model weights and scaler
- **`plots/`** - 4 visualization plots
- **`data/output/`** - Metrics and forecast reports

## Tips

1. **First time?** Start with AAPL and 30-day forecast
2. **GPU available?** The model will automatically use CUDA
3. **Want better results?** Increase training data (longer date range)
4. **Experiment!** Try different stocks and forecast horizons

## Troubleshooting

- **"No module named 'torch'"** → Run `pip install -r requirements.txt`
- **"Model not found"** → Train a model first with `--mode train`
- **Out of memory** → Reduce batch size in `stock_forecaster/config.py`

Enjoy forecasting! 📈
