"""
Evaluation module for assessing model performance with metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


class ModelEvaluator:
    """Evaluate model performance with various metrics and visualizations."""
    
    def __init__(self, predictions, actuals, ticker):
        """
        Initialize the evaluator.
        
        Args:
            predictions (np.ndarray): Model predictions
            actuals (np.ndarray): Actual values
            ticker (str): Stock ticker symbol
        """
        self.predictions = predictions
        self.actuals = actuals
        self.ticker = ticker
        self.metrics = {}
        
    def calculate_rmse(self):
        """Calculate Root Mean Squared Error."""
        rmse = np.sqrt(mean_squared_error(self.actuals.flatten(), self.predictions.flatten()))
        self.metrics['RMSE'] = rmse
        return rmse
    
    def calculate_mae(self):
        """Calculate Mean Absolute Error."""
        mae = mean_absolute_error(self.actuals.flatten(), self.predictions.flatten())
        self.metrics['MAE'] = mae
        return mae
    
    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error."""
        mape = np.mean(np.abs((self.actuals - self.predictions) / self.actuals)) * 100
        self.metrics['MAPE'] = mape
        return mape
    
    def calculate_directional_accuracy(self):
        """
        Calculate directional accuracy (% of correct up/down predictions).
        
        Returns:
            float: Directional accuracy percentage
        """
        # For multi-step forecasts, check direction of first step
        if self.predictions.ndim > 1 and self.predictions.shape[1] > 1:
            # Compare first predicted step with previous actual
            pred_direction = np.diff(self.predictions[:, 0])
            actual_direction = np.diff(self.actuals[:, 0])
        else:
            pred_direction = np.diff(self.predictions.flatten())
            actual_direction = np.diff(self.actuals.flatten())
        
        # Check if directions match
        correct_direction = np.sign(pred_direction) == np.sign(actual_direction)
        directional_accuracy = np.mean(correct_direction) * 100
        
        self.metrics['Directional_Accuracy'] = directional_accuracy
        return directional_accuracy
    
    def calculate_all_metrics(self):
        """
        Calculate all evaluation metrics.
        
        Returns:
            dict: Dictionary of all metrics
        """
        print("\n" + "="*60)
        print(f"EVALUATION METRICS FOR {self.ticker}")
        print("="*60)
        
        rmse = self.calculate_rmse()
        mae = self.calculate_mae()
        mape = self.calculate_mape()
        dir_acc = self.calculate_directional_accuracy()
        
        print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {dir_acc:.2f}%")
        
        print("="*60 + "\n")
        
        return self.metrics
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses (list): Training losses
            val_losses (list): Validation losses
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title(f'Training and Validation Loss - {self.ticker}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add best validation loss marker
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Val Loss: {best_loss:.6f}')
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def plot_predictions_vs_actual(self, forecast_horizon, save_path=None, num_samples=5):
        """
        Plot predicted vs actual stock prices for multiple forecast windows.
        
        Args:
            forecast_horizon (int): Number of days forecasted
            save_path (str): Path to save the plot
            num_samples (int): Number of forecast windows to display
        """
        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
        
        if num_samples == 1:
            axes = [axes]
        
        # Select evenly spaced samples from test set
        indices = np.linspace(0, len(self.predictions)-1, num_samples, dtype=int)
        
        for idx, ax in enumerate(axes):
            sample_idx = indices[idx]
            
            days = range(1, forecast_horizon + 1)
            pred = self.predictions[sample_idx]
            actual = self.actuals[sample_idx]
            
            ax.plot(days, actual, 'b-o', label='Actual', linewidth=2, markersize=6)
            ax.plot(days, pred, 'r--s', label='Predicted', linewidth=2, markersize=6)
            
            ax.set_xlabel('Days Ahead', fontsize=10)
            ax.set_ylabel('Stock Price', fontsize=10)
            ax.set_title(f'Forecast Window {sample_idx + 1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add error annotation
            mae = np.mean(np.abs(pred - actual))
            ax.text(0.02, 0.98, f'MAE: {mae:.2f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Stock Price Forecasts vs Actuals - {self.ticker}', 
                    fontsize=14, fontweight='bold', y=1.001)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions vs actual plot saved to {save_path}")
        
        plt.close()
    
    def plot_error_distribution(self, save_path=None):
        """
        Plot error distribution.
        
        Args:
            save_path (str): Path to save the plot
        """
        errors = (self.predictions - self.actuals).flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(errors, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('Prediction Error', fontsize=11)
        axes[1].set_title('Error Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Prediction Error Analysis - {self.ticker}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        
        plt.close()
    
    def plot_actual_vs_predicted_scatter(self, save_path=None):
        """
        Create scatter plot of actual vs predicted values.
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 10))
        
        actual_flat = self.actuals.flatten()
        pred_flat = self.predictions.flatten()
        
        plt.scatter(actual_flat, pred_flat, alpha=0.5, s=20, color='blue')
        
        # Perfect prediction line
        min_val = min(actual_flat.min(), pred_flat.min())
        max_val = max(actual_flat.max(), pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price', fontsize=12)
        plt.ylabel('Predicted Price', fontsize=12)
        plt.title(f'Actual vs Predicted Prices - {self.ticker}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        correlation = np.corrcoef(actual_flat, pred_flat)[0, 1]
        r_squared = correlation ** 2
        plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to {save_path}")
        
        plt.close()
    
    def save_metrics_to_file(self, filepath):
        """
        Save metrics to a text file.
        
        Args:
            filepath (str): Path to save metrics
        """
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"EVALUATION METRICS FOR {self.ticker}\n")
            f.write("="*60 + "\n\n")
            
            for metric, value in self.metrics.items():
                if 'Accuracy' in metric or 'MAPE' in metric:
                    f.write(f"{metric}: {value:.2f}%\n")
                else:
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Metrics saved to {filepath}")
    
    def generate_all_plots(self, train_losses, val_losses, forecast_horizon, plots_dir):
        """
        Generate all evaluation plots.
        
        Args:
            train_losses (list): Training losses
            val_losses (list): Validation losses
            forecast_horizon (int): Forecast horizon
            plots_dir (str): Directory to save plots
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION PLOTS")
        print("="*60 + "\n")
        
        os.makedirs(plots_dir, exist_ok=True)
        
        # Training history
        self.plot_training_history(
            train_losses, 
            val_losses,
            save_path=os.path.join(plots_dir, 'training_history.png')
        )
        
        # Predictions vs actual
        self.plot_predictions_vs_actual(
            forecast_horizon,
            save_path=os.path.join(plots_dir, 'predictions_vs_actual.png')
        )
        
        # Error distribution
        self.plot_error_distribution(
            save_path=os.path.join(plots_dir, 'error_distribution.png')
        )
        
        # Scatter plot
        self.plot_actual_vs_predicted_scatter(
            save_path=os.path.join(plots_dir, 'actual_vs_predicted_scatter.png')
        )
        
        print("All plots generated successfully!\n")


if __name__ == "__main__":
    # Test evaluator
    from .data_fetcher import fetch_stock_data
    from .feature_engineering import prepare_data_for_training
    from .predictor import load_model_for_prediction, predict_on_test_set
    from .config import Config
    import os
    import torch
    
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Model not found. Please train the model first.")
    else:
        # Fetch and prepare data
        data = fetch_stock_data(
            ticker=Config.DEFAULT_TICKER,
            start_date=Config.START_DATE,
            end_date=Config.END_DATE
        )
        
        prepared_data = prepare_data_for_training(
            data,
            lookback=Config.LOOKBACK_WINDOW,
            forecast_horizon=Config.FORECAST_HORIZON,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO
        )
        
        # Load model and make predictions
        input_size = prepared_data['X_train'].shape[2]
        predictor = load_model_for_prediction(
            model_path=Config.MODEL_SAVE_PATH,
            scaler_path=Config.SCALER_SAVE_PATH,
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            forecast_horizon=Config.FORECAST_HORIZON,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL,
            device=Config.DEVICE
        )
        
        results = predict_on_test_set(
            predictor,
            prepared_data['X_test'],
            prepared_data['y_test']
        )
        
        # Evaluate
        evaluator = ModelEvaluator(
            predictions=results['predictions'],
            actuals=results['actuals'],
            ticker=Config.DEFAULT_TICKER
        )
        
        metrics = evaluator.calculate_all_metrics()
        
        # Load training history (mock for testing)
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        # Generate plots
        evaluator.generate_all_plots(
            train_losses,
            val_losses,
            Config.FORECAST_HORIZON,
            Config.PLOTS_PATH
        )
        
        # Save metrics
        evaluator.save_metrics_to_file(
            os.path.join(Config.OUTPUT_PATH, 'evaluation_metrics.txt')
        )
        
        print("\nEvaluation completed successfully!")
