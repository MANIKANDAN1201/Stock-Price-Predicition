"""
Streamlit Web Application for Stock Price Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf

# Add stock_forecaster to path
sys.path.append(os.path.dirname(__file__))

from stock_forecaster.config import Config
from stock_forecaster.data_fetcher import fetch_stock_data
from stock_forecaster.feature_engineering import prepare_data_for_training, FeatureEngineer
from stock_forecaster.model_trainer import LSTMForecaster
from stock_forecaster.predictor import StockPredictor
import pickle

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_for_ticker(ticker, forecast_horizon=30):
    """Load trained model for a specific ticker."""
    model_path = f"models/{ticker}_forecaster.pth"
    scaler_path = f"models/{ticker}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None
    
    try:
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Determine input size from scaler
        input_size = scaler.n_features_in_
        
        # Load model
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            forecast_horizon=forecast_horizon,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL
        )
        
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        predictor = StockPredictor(model, scaler, Config.DEVICE)
        
        return predictor, scaler, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def prepare_latest_data(ticker, scaler, lookback=60):
    """Prepare the latest data for prediction."""
    try:
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None, None
        
        data.reset_index(inplace=True)
        
        # Feature engineering
        engineer = FeatureEngineer(data)
        engineered_data = engineer.create_all_features(
            ma_short=Config.MA_SHORT,
            ma_long=Config.MA_LONG,
            rsi_period=Config.RSI_PERIOD,
            macd_fast=Config.MACD_FAST,
            macd_slow=Config.MACD_SLOW,
            macd_signal=Config.MACD_SIGNAL
        )
        
        # Scale features
        engineer.scaler = scaler  # Use the pre-fitted scaler
        scaled_data = engineer.scale_features(fit=False)
        
        # Get last lookback window
        if len(scaled_data) < lookback:
            return None, None
        
        last_sequence = scaled_data[-lookback:]
        
        # Get actual recent prices for context
        recent_prices = engineered_data['Close'].tail(30).values
        
        return last_sequence, recent_prices
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None


def inverse_transform_price(scaler, scaled_price):
    """Convert scaled price back to original scale."""
    n_features = scaler.n_features_in_
    dummy = np.zeros((1, n_features))
    dummy[0, 3] = scaled_price  # Close price is at index 3
    inverse = scaler.inverse_transform(dummy)
    return float(inverse[0, 3])


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Price Forecaster</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Multi-Day Stock Price Prediction")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL, META, TSLA)"
    ).upper()
    
    # Forecast horizon
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=30,
        value=30,
        help="Number of days to forecast into the future"
    )
    
    # Available models
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Available Trained Models")
    
    model_files = [f.replace("_forecaster.pth", "") for f in os.listdir("models") if f.endswith("_forecaster.pth")]
    
    if model_files:
        for model_ticker in sorted(model_files):
            st.sidebar.markdown(f"✅ {model_ticker}")
    else:
        st.sidebar.warning("No trained models found")
    
    # Main content
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Selected Ticker", ticker)
    with col2:
        st.metric("Forecast Horizon", f"{forecast_days} days")
    with col3:
        st.metric("Model Status", "🟢 Ready" if ticker in model_files else "🔴 Not Trained")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        if ticker not in model_files:
            st.error(f"❌ No trained model found for {ticker}. Please train the model first.")
            st.info(f"Run: `python batch_train.py --tickers {ticker}` to train this model.")
            return
        
        with st.spinner(f"Loading model and generating forecast for {ticker}..."):
            # Load model
            predictor, scaler, checkpoint = load_model_for_ticker(ticker, forecast_days)
            
            if predictor is None:
                st.error("Failed to load model. Please check the model files.")
                return
            
            # Prepare latest data
            last_sequence, recent_prices = prepare_latest_data(ticker, scaler, Config.LOOKBACK_WINDOW)
            
            if last_sequence is None:
                st.error("Failed to fetch or prepare data. Please try again.")
                return
            
            # Make prediction
            predictions_scaled = predictor.predict(last_sequence.reshape(1, *last_sequence.shape))
            predictions_scaled = predictions_scaled[0]  # Get first (and only) sample
            
            # Inverse transform predictions
            predictions = np.array([inverse_transform_price(scaler, p) for p in predictions_scaled])
            
            # Get last known price (ensure it's a scalar)
            last_known_price = float(recent_prices[-1])
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=datetime.now() + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Day': range(1, forecast_days + 1),
                'Predicted_Price': predictions,
                'Change_from_Last': predictions - last_known_price,
                'Change_Percent': ((predictions - last_known_price) / last_known_price) * 100
            })
            
            # Display results
            st.success(f"✅ Forecast generated successfully for {ticker}!")
            
            # Key metrics
            st.markdown("### 📊 Key Forecast Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${float(last_known_price):.2f}"
                )
            
            with col2:
                day7_price = float(predictions[6] if len(predictions) >= 7 else predictions[-1])
                day7_change = ((day7_price - last_known_price) / last_known_price) * 100
                st.metric(
                    "7-Day Forecast",
                    f"${day7_price:.2f}",
                    f"{float(day7_change):+.2f}%"
                )
            
            with col3:
                day14_price = float(predictions[13] if len(predictions) >= 14 else predictions[-1])
                day14_change = ((day14_price - last_known_price) / last_known_price) * 100
                st.metric(
                    "14-Day Forecast",
                    f"${day14_price:.2f}",
                    f"{float(day14_change):+.2f}%"
                )
            
            with col4:
                final_price = float(predictions[-1])
                final_change = ((final_price - last_known_price) / last_known_price) * 100
                st.metric(
                    f"{forecast_days}-Day Forecast",
                    f"${final_price:.2f}",
                    f"{float(final_change):+.2f}%"
                )
            
            # Visualization
            st.markdown("### 📈 Price Forecast Visualization")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot recent historical prices
            recent_dates = pd.date_range(
                end=datetime.now(),
                periods=len(recent_prices),
                freq='D'
            )
            ax.plot(recent_dates, recent_prices, 'b-o', label='Historical Prices', linewidth=2, markersize=4)
            
            # Plot forecast
            ax.plot(forecast_dates, predictions, 'r--s', label='Forecasted Prices', linewidth=2, markersize=5)
            
            # Add connection line (ensure scalar values)
            ax.plot([recent_dates[-1], forecast_dates[0]], 
                   [float(recent_prices[-1]), float(predictions[0])], 
                   'g:', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_title(f'{ticker} Stock Price Forecast - Next {forecast_days} Days', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Detailed forecast table
            st.markdown("### 📋 Detailed Forecast Table")
            
            # Format the dataframe for display
            display_df = forecast_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"${float(x):.2f}")
            display_df['Change_from_Last'] = display_df['Change_from_Last'].apply(lambda x: f"${float(x):+.2f}")
            display_df['Change_Percent'] = display_df['Change_Percent'].apply(lambda x: f"{float(x):+.2f}%")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Model info
            with st.expander("ℹ️ Model Information"):
                st.markdown(f"""
                **Model Architecture:** Bidirectional LSTM
                - **Hidden Size:** {Config.HIDDEN_SIZE}
                - **Layers:** {Config.NUM_LAYERS}
                - **Dropout:** {Config.DROPOUT}
                - **Lookback Window:** {Config.LOOKBACK_WINDOW} days
                - **Training Loss:** {checkpoint.get('best_val_loss', 'N/A'):.6f}
                - **Device:** {Config.DEVICE}
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. 
        Stock predictions are inherently uncertain. Always consult financial professionals before making investment decisions.</p>
        <p>Built with ❤️ using PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
