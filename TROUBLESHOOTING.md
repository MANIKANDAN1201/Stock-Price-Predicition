# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. TypeError: unsupported format string passed to numpy.ndarray.__format__

**Error Message:**
```
TypeError: unsupported format string passed to numpy.ndarray.__format__
File "D:\StockPrice\app.py", line 254, in main
    f"${last_known_price:.2f}"
```

**Cause:**
- NumPy arrays cannot be directly formatted with f-strings
- Need to convert to Python float first

**Solution:**
✅ **FIXED** - All numpy values are now explicitly converted to float:
```python
# Before (causes error)
f"${last_known_price:.2f}"

# After (works correctly)
f"${float(last_known_price):.2f}"
```

**Files Updated:**
- `app.py` - All price formatting now uses `float()` conversion

---

### 1b. ValueError: setting an array element with a sequence

**Error Message:**
```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape
File "D:\StockPrice\app.py", line 301, in main
    ax.plot([recent_dates[-1], forecast_dates[0]], [recent_prices[-1], predictions[0]])
```

**Cause:**
- Matplotlib expects scalar values for plotting points
- NumPy array indexing can return arrays instead of scalars

**Solution:**
✅ **FIXED** - Convert to float in plotting:
```python
# Before (causes error)
ax.plot([recent_dates[-1], forecast_dates[0]], 
        [recent_prices[-1], predictions[0]], 'g:')

# After (works correctly)
ax.plot([recent_dates[-1], forecast_dates[0]], 
        [float(recent_prices[-1]), float(predictions[0])], 'g:')
```

**Files Updated:**
- `app.py` - Line 301-302 now uses `float()` conversion

---

### 2. ModuleNotFoundError: No module named 'model_trainer'

**Error Message:**
```
ModuleNotFoundError: No module named 'model_trainer'
```

**Cause:**
- Incorrect import statements (absolute instead of relative)

**Solution:**
✅ **FIXED** - All imports now use relative imports:
```python
# Before
from model_trainer import LSTMForecaster

# After
from .model_trainer import LSTMForecaster
```

---

### 3. TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'

**Error Message:**
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

**Cause:**
- PyTorch 2.4+ removed the `verbose` parameter

**Solution:**
✅ **FIXED** - Removed `verbose` parameter and added manual logging:
```python
# Before
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# After
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=5
)
# Manual logging added in training loop
```

---

### 4. Streamlit App Not Loading

**Symptoms:**
- Blank page
- Connection errors
- Port already in use

**Solutions:**

**A. Port Already in Use:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**B. Clear Cache:**
```bash
# In the app, press 'C' to clear cache
# Or restart the app
```

**C. Check Dependencies:**
```bash
pip install --upgrade streamlit
```

---

### 5. Model Not Found Error

**Error Message:**
```
❌ No trained model found for AAPL
```

**Cause:**
- Model hasn't been trained yet
- Model files in wrong location

**Solutions:**

**A. Train the Model:**
```bash
python batch_train.py --tickers AAPL
```

**B. Check Model Files:**
```bash
# Should exist:
models/AAPL_forecaster.pth
models/AAPL_scaler.pkl
```

**C. Verify File Paths:**
```python
import os
print(os.path.exists("models/AAPL_forecaster.pth"))
```

---

### 6. Data Fetching Errors

**Error Message:**
```
Failed to fetch or prepare data
```

**Possible Causes & Solutions:**

**A. Invalid Ticker:**
```bash
# Check ticker symbol is correct
# Use uppercase: AAPL not aapl
```

**B. Internet Connection:**
```bash
# Test connection
ping finance.yahoo.com
```

**C. Market Hours:**
- Yahoo Finance may have delays
- Try again in a few minutes

**D. Delisted Stock:**
- Stock may no longer trade
- Try a different ticker

---

### 7. CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**A. Reduce Batch Size:**
Edit `stock_forecaster/config.py`:
```python
BATCH_SIZE = 16  # Default: 32
```

**B. Use CPU Instead:**
Edit `stock_forecaster/config.py`:
```python
DEVICE = torch.device('cpu')
```

**C. Reduce Model Size:**
```python
HIDDEN_SIZE = 64   # Default: 128
NUM_LAYERS = 2     # Default: 3
```

---

### 8. Poor Prediction Accuracy

**Symptoms:**
- High MAPE (>10%)
- Predictions don't match trends
- Large errors

**Solutions:**

**A. More Training Data:**
```bash
python batch_train.py --tickers AAPL --start 2015-01-01
```

**B. More Training Epochs:**
```bash
python batch_train.py --tickers AAPL --epochs 150
```

**C. Retrain Regularly:**
```bash
# Retrain monthly with latest data
python batch_train.py --tickers AAPL
```

**D. Shorter Forecast Horizon:**
```bash
# 7-day forecasts are more accurate than 30-day
python batch_train.py --tickers AAPL --forecast-days 7
```

---

### 9. Import Errors in Jupyter Notebook

**Error:**
```
ImportError: attempted relative import with no known parent package
```

**Solution:**
Use absolute imports in notebooks:
```python
import sys
sys.path.append('path/to/StockPrice')

from stock_forecaster.config import Config
from stock_forecaster.data_fetcher import fetch_stock_data
```

---

### 10. Pandas FutureWarning

**Warning:**
```
FutureWarning: DataFrame.fillna with 'method' is deprecated
```

**Solution:**
Update pandas or ignore warnings:
```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

Or update the code:
```python
# Old
df.fillna(method='ffill')

# New
df.ffill()
```

---

## Installation Issues

### Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Version Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account...
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### PyTorch Installation

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

**For GPU (CUDA):**
```bash
# Visit pytorch.org for your specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Issues

### Slow Training

**Solutions:**
1. Use GPU if available
2. Reduce epochs in quick mode
3. Reduce batch size
4. Use fewer features

### Slow Web App

**Solutions:**
1. Models are cached after first load
2. Reduce forecast horizon
3. Use faster internet connection
4. Close other browser tabs

---

## Data Issues

### Missing Values

**Handled automatically by:**
- Forward fill for gaps
- Backward fill for start
- Dropping remaining NaN

### Non-Trading Days

**Handled automatically:**
- Weekend gaps filled
- Holiday gaps filled
- Uses previous day's data

---

## Getting Help

### Check These Files First:
1. `TROUBLESHOOTING.md` (this file)
2. `README.md` - Full documentation
3. `STREAMLIT_GUIDE.md` - Web app guide
4. `USAGE_EXAMPLES.md` - Practical examples

### Debug Mode:
```python
# Add to app.py for debugging
import streamlit as st
st.write("Debug info:", variable_name)
```

### Common Commands:
```bash
# Test imports
python -c "from stock_forecaster.config import Config; print('OK')"

# Check model files
ls models/

# View logs
streamlit run app.py --logger.level=debug
```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Format error | All fixed - update app.py |
| Import error | Use relative imports |
| PyTorch verbose | Removed from code |
| Model not found | Run batch_train.py |
| Data fetch fail | Check internet & ticker |
| Out of memory | Reduce batch size |
| Poor accuracy | More data & epochs |
| Slow training | Use GPU or quick mode |

---

## Still Having Issues?

1. **Check Python version:** Requires Python 3.8+
2. **Update packages:** `pip install --upgrade -r requirements.txt`
3. **Fresh install:** Delete `venv` and reinstall
4. **Check files:** Ensure all files are present
5. **Review error:** Read full error message carefully

---

**Last Updated:** After fixing numpy formatting issues
**Status:** All known issues resolved ✅
