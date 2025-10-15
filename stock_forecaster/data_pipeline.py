"""
Enhanced data pipeline with proper folder structure and data management.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json


class DataPipeline:
    """Manages data storage, preprocessing, and organization."""
    
    def __init__(self, ticker, base_path="data"):
        """
        Initialize data pipeline for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            base_path (str): Base directory for data storage
        """
        self.ticker = ticker
        self.base_path = base_path
        
        # Create directory structure
        self.raw_dir = os.path.join(base_path, "raw", ticker)
        self.processed_dir = os.path.join(base_path, "processed", ticker)
        self.output_dir = os.path.join(base_path, "output", ticker)
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        for directory in [self.raw_dir, self.processed_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_raw_data(self, data, start_date, end_date):
        """
        Save raw fetched data.
        
        Args:
            data (pd.DataFrame): Raw stock data
            start_date (str): Start date of data
            end_date (str): End date of data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.ticker}_raw_{start_date}_to_{end_date}_{timestamp}.csv"
        filepath = os.path.join(self.raw_dir, filename)
        
        data.to_csv(filepath, index=False)
        print(f"✓ Raw data saved: {filepath}")
        
        # Save metadata
        metadata = {
            'ticker': self.ticker,
            'start_date': start_date,
            'end_date': end_date,
            'rows': len(data),
            'columns': list(data.columns),
            'date_range': {
                'min': str(data['Date'].min()),
                'max': str(data['Date'].max())
            },
            'saved_at': timestamp
        }
        
        metadata_file = os.path.join(self.raw_dir, f"{self.ticker}_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def save_processed_data(self, data, feature_columns, scaler):
        """
        Save processed data with features.
        
        Args:
            data (pd.DataFrame): Processed data with features
            feature_columns (list): List of feature column names
            scaler: Fitted scaler object
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed data
        filename = f"{self.ticker}_processed_{timestamp}.csv"
        filepath = os.path.join(self.processed_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"✓ Processed data saved: {filepath}")
        
        # Save scaler
        scaler_file = os.path.join(self.processed_dir, f"{self.ticker}_scaler_{timestamp}.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Scaler saved: {scaler_file}")
        
        # Save feature info
        feature_info = {
            'ticker': self.ticker,
            'feature_columns': feature_columns,
            'num_features': len(feature_columns),
            'data_shape': data.shape,
            'processed_at': timestamp,
            'statistics': {
                col: {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                } for col in feature_columns if col in data.columns
            }
        }
        
        feature_file = os.path.join(self.processed_dir, f"{self.ticker}_features_{timestamp}.json")
        with open(feature_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Feature info saved: {feature_file}")
        
        return filepath, scaler_file
    
    def save_training_results(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Save train/val/test splits.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy arrays
        splits_file = os.path.join(self.processed_dir, f"{self.ticker}_splits_{timestamp}.npz")
        np.savez_compressed(
            splits_file,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )
        print(f"✓ Train/val/test splits saved: {splits_file}")
        
        # Save split info
        split_info = {
            'ticker': self.ticker,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'input_shape': X_train.shape,
            'output_shape': y_train.shape,
            'split_at': timestamp
        }
        
        info_file = os.path.join(self.processed_dir, f"{self.ticker}_split_info_{timestamp}.json")
        with open(info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return splits_file
    
    def save_predictions(self, predictions, actuals, dates=None):
        """
        Save model predictions.
        
        Args:
            predictions (np.ndarray): Model predictions
            actuals (np.ndarray): Actual values
            dates (list): Optional dates for predictions
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'Prediction': predictions.flatten(),
            'Actual': actuals.flatten()
        })
        
        if dates is not None:
            pred_df['Date'] = dates
        
        pred_df['Error'] = pred_df['Actual'] - pred_df['Prediction']
        pred_df['Absolute_Error'] = np.abs(pred_df['Error'])
        pred_df['Percentage_Error'] = (pred_df['Absolute_Error'] / pred_df['Actual']) * 100
        
        # Save predictions
        pred_file = os.path.join(self.output_dir, f"{self.ticker}_predictions_{timestamp}.csv")
        pred_df.to_csv(pred_file, index=False)
        print(f"✓ Predictions saved: {pred_file}")
        
        return pred_file
    
    def get_latest_processed_data(self):
        """Get the most recent processed data file."""
        files = [f for f in os.listdir(self.processed_dir) if f.startswith(f"{self.ticker}_processed_")]
        if not files:
            return None
        
        latest = sorted(files)[-1]
        return os.path.join(self.processed_dir, latest)
    
    def get_latest_scaler(self):
        """Get the most recent scaler file."""
        files = [f for f in os.listdir(self.processed_dir) if f.startswith(f"{self.ticker}_scaler_")]
        if not files:
            return None
        
        latest = sorted(files)[-1]
        return os.path.join(self.processed_dir, latest)
    
    def get_data_summary(self):
        """Get summary of all data for this ticker."""
        summary = {
            'ticker': self.ticker,
            'raw_files': len([f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]),
            'processed_files': len([f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]),
            'output_files': len([f for f in os.listdir(self.output_dir) if f.endswith('.csv')])
        }
        
        return summary


def create_data_report(ticker, data, metrics, save_path):
    """
    Create a comprehensive data report.
    
    Args:
        ticker (str): Stock ticker
        data (pd.DataFrame): Stock data
        metrics (dict): Model metrics
        save_path (str): Path to save report
    """
    report = []
    report.append("="*80)
    report.append(f"DATA REPORT FOR {ticker}".center(80))
    report.append("="*80)
    report.append("")
    
    # Data info
    report.append("DATA INFORMATION")
    report.append("-"*80)
    report.append(f"Total rows: {len(data)}")
    report.append(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    report.append(f"Trading days: {len(data)}")
    report.append("")
    
    # Price statistics
    report.append("PRICE STATISTICS")
    report.append("-"*80)
    report.append(f"Current price: ${float(data['Close'].iloc[-1]):.2f}")
    report.append(f"Highest price: ${float(data['Close'].max()):.2f}")
    report.append(f"Lowest price: ${float(data['Close'].min()):.2f}")
    report.append(f"Average price: ${float(data['Close'].mean()):.2f}")
    report.append(f"Price volatility (std): ${float(data['Close'].std()):.2f}")
    report.append("")
    
    # Model metrics
    if metrics:
        report.append("MODEL PERFORMANCE")
        report.append("-"*80)
        for key, value in metrics.items():
            if 'Accuracy' in key or 'MAPE' in key:
                report.append(f"{key}: {value:.2f}%")
            else:
                report.append(f"{key}: {value:.6f}")
        report.append("")
    
    report.append("="*80)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Data report saved: {save_path}")
    return '\n'.join(report)


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataPipeline("AAPL")
    print(pipeline.get_data_summary())
