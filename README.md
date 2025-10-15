# 📈 Stock Price Forecasting Model

A comprehensive machine learning pipeline for forecasting stock prices 7-30 days into the future using LSTM neural networks and technical indicators.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **🌐 Web Interface**: Beautiful Streamlit app for easy forecasting (NEW!)
- **📊 Batch Training**: Train models for multiple stocks at once (NEW!)
- **Real-time Data Fetching**: Automatically downloads historical stock data from Yahoo Finance
- **Advanced Feature Engineering**: Creates 14+ technical indicators including MA, RSI, MACD, and volatility metrics
- **Deep Learning Architecture**: Bidirectional LSTM with dropout and attention mechanisms
- **Multi-step Forecasting**: Predict 7-30 days ahead with configurable horizon
- **Comprehensive Evaluation**: RMSE, MAE, MAPE, and directional accuracy metrics
- **Rich Visualizations**: Training curves, prediction plots, error distributions, and scatter plots
- **Modular Design**: Easy to extend and customize for different stocks and timeframes

## 📁 Project Structure

```
StockPrice/
│
├── stock_forecaster/          # Main package
│   ├── __init__.py
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_fetcher.py        # Yahoo Finance data downloader
│   ├── feature_engineering.py # Technical indicators and preprocessing
│   ├── model_trainer.py       # LSTM model and training logic
│   ├── predictor.py           # Prediction and forecasting
│   └── evaluator.py           # Metrics and visualization
│
├── models/                    # Saved model weights
│   ├── stock_forecaster.pth
│   └── scaler.pkl
│
├── data/                      # Data storage
│   └── output/                # Forecast reports and metrics
│
├── plots/                     # Generated visualizations
│   ├── training_history.png
│   ├── predictions_vs_actual.png
│   ├── error_distribution.png
│   └── actual_vs_predicted_scatter.png
│
├── main.py                    # Main orchestration script
├── batch_train.py             # Batch training for multiple stocks (NEW!)
├── app.py                     # Streamlit web application (NEW!)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd StockPrice

# Install dependencies
pip install -r requirements.txt
```

### 2. Option A: Use Web Interface (Recommended!)

```bash
# Train models for popular stocks
python batch_train.py --tickers AAPL MSFT GOOGL META TSLA

# Launch the web app
streamlit run app.py
```

Then open your browser and:
- Select a stock ticker
- Choose forecast horizon (7-30 days)
- Click "Generate Forecast"
- View results and download CSV

### 2. Option B: Command Line Interface

Train a model for Apple (AAPL) with 30-day forecast:

```bash
python main.py --mode train --ticker AAPL --forecast-days 30
```

Train for Microsoft with custom date range:

```bash
python main.py --mode train --ticker MSFT --start 2019-01-01 --end 2024-12-31 --forecast-days 14
```

Make predictions:

```bash
python main.py --mode predict --ticker AAPL --forecast-days 30
```

### 3. Batch Training Multiple Stocks

```bash
# Train all popular stocks (15 tickers)
python batch_train.py

# Train specific stocks
python batch_train.py --tickers AAPL MSFT GOOGL

# Quick mode (fewer epochs for testing)
python batch_train.py --tickers AAPL --quick
```

## 📊 Technical Indicators

The model uses the following features:

| Feature | Description |
|---------|-------------|
| **OHLCV** | Open, High, Low, Close, Volume |
| **MA_7** | 7-day Moving Average |
| **MA_21** | 21-day Moving Average |
| **RSI** | Relative Strength Index (14-day) |
| **MACD** | Moving Average Convergence Divergence |
| **MACD_Signal** | MACD Signal Line |
| **MACD_Hist** | MACD Histogram |
| **Volatility** | 20-day rolling standard deviation |
| **Price_Change** | Daily price change |
| **Price_Change_Pct** | Daily percentage change |

## 🧠 Model Architecture

### LSTM Forecaster

- **Input Layer**: Multi-variate time series (14 features)
- **LSTM Layers**: 3 bidirectional layers with 128 hidden units
- **Dropout**: 20% dropout for regularization
- **Output Layer**: Fully connected layers for multi-step forecasting
- **Total Parameters**: ~500K trainable parameters

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Lookback Window | 60 days |
| Forecast Horizon | 7-30 days (configurable) |
| Batch Size | 32 |
| Learning Rate | 0.001 (with scheduling) |
| Optimizer | Adam with weight decay |
| Loss Function | MSE (Mean Squared Error) |
| Early Stopping | 15 epochs patience |

## 📈 Evaluation Metrics

The model is evaluated using:

1. **RMSE (Root Mean Squared Error)**: Overall prediction accuracy
2. **MAE (Mean Absolute Error)**: Average absolute deviation
3. **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
4. **Directional Accuracy**: % of correct up/down trend predictions

## 🎨 Visualizations

The pipeline automatically generates:

1. **Training History**: Loss curves for training and validation
2. **Predictions vs Actual**: Multi-window forecast comparisons
3. **Error Distribution**: Histogram and box plot of prediction errors
4. **Scatter Plot**: Actual vs predicted prices with R² score

## ⚙️ Configuration

Edit `stock_forecaster/config.py` to customize:

```python
# Data Configuration
DEFAULT_TICKER = "AAPL"
START_DATE = "2018-01-01"
LOOKBACK_WINDOW = 60
FORECAST_HORIZON = 30

# Model Architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.2
BIDIRECTIONAL = True

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15
```

## 📝 Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --mode {train,predict}    Mode: train new model or predict with existing
  --ticker TICKER           Stock ticker symbol (default: AAPL)
  --start START_DATE        Start date YYYY-MM-DD (default: 2018-01-01)
  --end END_DATE            End date YYYY-MM-DD (default: 2025-01-01)
  --forecast-days DAYS      Days to forecast, 7-30 (default: 30)
```

## 🔬 Example Usage

### Training Multiple Stocks

```bash
# Train for different stocks
python main.py --mode train --ticker AAPL --forecast-days 30
python main.py --mode train --ticker GOOGL --forecast-days 14
python main.py --mode train --ticker TSLA --forecast-days 7
```

### Custom Date Ranges

```bash
# Train on recent data only
python main.py --mode train --ticker AAPL --start 2022-01-01 --end 2024-12-31
```

## 📊 Output Files

After training, the following files are generated:

```
models/
  ├── stock_forecaster.pth    # Trained model weights
  └── scaler.pkl              # Fitted MinMaxScaler

data/output/
  ├── evaluation_metrics.txt  # Performance metrics
  └── forecast_report.csv     # Detailed forecast report

plots/
  ├── training_history.png
  ├── predictions_vs_actual.png
  ├── error_distribution.png
  └── actual_vs_predicted_scatter.png
```

## 🛠️ Extending the Pipeline

### Add New Technical Indicators

Edit `stock_forecaster/feature_engineering.py`:

```python
def add_custom_indicator(self):
    """Add your custom indicator."""
    self.data['Custom_Indicator'] = ...
    return self.data
```

### Use Different Model Architectures

Edit `stock_forecaster/model_trainer.py` to implement:
- Transformer models
- GRU networks
- Temporal Fusion Transformers (TFT)

### Add External Data Sources

Extend `data_fetcher.py` to include:
- News sentiment data
- Macroeconomic indicators
- Social media trends

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Reduce `HIDDEN_SIZE` or `NUM_LAYERS`

### Poor Performance
- Increase training data (longer date range)
- Adjust `LOOKBACK_WINDOW`
- Add more technical indicators
- Tune hyperparameters

### Data Fetching Issues
- Check internet connection
- Verify ticker symbol is valid
- Ensure date range is reasonable

## 📚 Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.1+
- **yfinance**: Stock data API
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Preprocessing and metrics
- **matplotlib/seaborn**: Visualization

## 🎓 Model Performance

Typical performance on major stocks (AAPL, MSFT, GOOGL):

- **RMSE**: 2-5% of stock price
- **MAE**: 1-3% of stock price
- **MAPE**: 2-6%
- **Directional Accuracy**: 55-65%

*Note: Performance varies by stock volatility and market conditions*

## ⚠️ Disclaimer

This model is for **educational and research purposes only**. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct thorough research before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement Transformer/TFT models
- [ ] Add hyperparameter tuning with Optuna
- [ ] Integrate sentiment analysis
- [ ] Create REST API for predictions
- [ ] Add multi-ticker batch forecasting
- [ ] Implement ensemble methods

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock data API
- PyTorch team for the excellent deep learning framework
- The open-source community for various libraries used

---

**Built with ❤️ for stock market enthusiasts and ML practitioners**

For questions or issues, please open an issue on the repository.
