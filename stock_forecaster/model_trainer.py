"""
Model training module with LSTM architecture for stock price forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from datetime import datetime


class LSTMForecaster(nn.Module):
    """LSTM-based model for multi-step stock price forecasting."""
    
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon, 
                 dropout=0.2, bidirectional=True):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Hidden layer size
            num_layers (int): Number of LSTM layers
            forecast_horizon (int): Number of steps to forecast
            dropout (float): Dropout rate
            bidirectional (bool): Use bidirectional LSTM
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, forecast_horizon)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features)
        
        Returns:
            torch.Tensor: Output predictions of shape (batch, forecast_horizon)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        out = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class ModelTrainer:
    """Train and validate the LSTM forecasting model."""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train
            device (torch.device): Device to train on
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                epoch_loss += loss.item()
        
        return epoch_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs, patience=15):
        """
        Train the model with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
        
        Returns:
            dict: Training history
        """
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*60 + "\n")
        
        start_time = time.time()
        patience_counter = 0
        prev_lr = self.optimizer.param_groups[0]['lr']
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check if learning rate changed
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"\nLearning rate reduced: {prev_lr:.6f} -> {current_lr:.6f}")
                prev_lr = current_lr
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                improvement = "✓"
            else:
                patience_counter += 1
                improvement = ""
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {epoch_time:.2f}s {improvement}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print("="*60 + "\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Model loaded from {filepath}")


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size):
    """
    Create PyTorch data loaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size (int): Batch size
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"Created data loaders with batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test model training
    from .data_fetcher import fetch_stock_data
    from .feature_engineering import prepare_data_for_training
    from .config import Config
    
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Fetch and prepare data
    print("Fetching data...")
    data = fetch_stock_data(
        ticker=Config.DEFAULT_TICKER,
        start_date=Config.START_DATE,
        end_date=Config.END_DATE
    )
    
    print("\nPreparing data...")
    prepared_data = prepare_data_for_training(
        data,
        lookback=Config.LOOKBACK_WINDOW,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        prepared_data['X_train'],
        prepared_data['y_train'],
        prepared_data['X_val'],
        prepared_data['y_val'],
        Config.BATCH_SIZE
    )
    
    # Initialize model
    input_size = prepared_data['X_train'].shape[2]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        forecast_horizon=Config.FORECAST_HORIZON,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL
    )
    
    # Train model
    trainer = ModelTrainer(
        model=model,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE
    )
    
    # Save model
    trainer.save_model(Config.MODEL_SAVE_PATH)
    
    print("\nModel training completed successfully!")
