import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ChunkSizePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # Smaller, focused architecture
        self.hidden_sizes = [64, 32, 16]
        
        # Input layer with batch normalization
        layers = [
            nn.Linear(input_size, self.hidden_sizes[0]),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)  # Increased dropout for better regularization
        ]
        
        # Hidden layers with residual connections
        for i in range(len(self.hidden_sizes)-1):
            layers.extend([
                nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]),
                nn.BatchNorm1d(self.hidden_sizes[i+1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            ])
        
        # Output layer
        layers.append(nn.Linear(self.hidden_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, input_size, preprocessor=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChunkSizePredictor(input_size).to(self.device)
        self.best_model_state = None
        self.preprocessor = preprocessor
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_metrics': {}}
        
        # Create plots directory
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def train_fold(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train.reshape(-1, 1))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        criterion = nn.HuberLoss(delta=0.1)  # More robust to outliers than MSE
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val.reshape(-1, 1))
                val_losses.append(val_loss.item())
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation with multiple metrics
        metrics = self._calculate_metrics(X_val, y_val)
        
        # Plot predictions vs actual
        self._plot_predictions(X_val, y_val)
        
        # For backward compatibility, return MSE and R² as a tuple
        if self.preprocessor is not None:
            return metrics['mse_orig'], metrics['r2_orig'], metrics
        return metrics['mse'], metrics['r2'], metrics
    
    def _calculate_metrics(self, X_val, y_val):
        """Calculate multiple evaluation metrics."""
        self.model.eval()
        with torch.no_grad():
            val_predictions = self.model(X_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            
            # Calculate metrics on scaled data
            mse = mean_squared_error(y_val_np, val_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_np, val_predictions)
            r2 = r2_score(y_val_np, val_predictions)
            
            # If preprocessor is available, calculate metrics on original scale
            if self.preprocessor is not None:
                y_true_orig = self.preprocessor.inverse_transform_target(y_val_np)
                y_pred_orig = self.preprocessor.inverse_transform_target(val_predictions.ravel())
                
                mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
                rmse_orig = np.sqrt(mse_orig)
                mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
                mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
                r2_orig = r2_score(y_true_orig, y_pred_orig)
                
                # Log detailed metrics
                logging.info("\nValidation Metrics (Original Scale):")
                logging.info(f"MSE: {mse_orig:.2f}")
                logging.info(f"RMSE: {rmse_orig:.2f}")
                logging.info(f"MAE: {mae_orig:.2f}")
                logging.info(f"MAPE: {mape:.4f}")
                logging.info(f"R²: {r2_orig:.4f}")
                
                return {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mse_orig': mse_orig,
                    'rmse_orig': rmse_orig,
                    'mae_orig': mae_orig,
                    'mape': mape,
                    'r2_orig': r2_orig
                }
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
    
    def _plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.plots_dir / 'training_history.png')
        plt.close()
    
    def _plot_predictions(self, X_val, y_val):
        """Plot predicted vs actual values."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            
            if self.preprocessor is not None:
                y_true = self.preprocessor.inverse_transform_target(y_val_np)
                y_pred = self.preprocessor.inverse_transform_target(predictions.ravel())
            else:
                y_true = y_val_np
                y_pred = predictions.ravel()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Chunk Size')
            plt.ylabel('Predicted Chunk Size')
            plt.title('Predicted vs Actual Chunk Sizes')
            plt.savefig(self.plots_dir / 'predictions_scatter.png')
            plt.close()
            
            # Plot error distribution
            errors = y_pred - y_true
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, bins=50)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Error')
            plt.ylabel('Count')
            plt.savefig(self.plots_dir / 'error_distribution.png')
            plt.close()
    
    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu().numpy().reshape(-1) 