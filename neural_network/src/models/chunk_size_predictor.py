import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
        
        # Output layer for binary classification
        layers.append(nn.Linear(self.hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
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
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        
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
                
                # Calculate metrics
                val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int)
                val_true = y_val.cpu().numpy()
                
                accuracy = accuracy_score(val_true, val_preds)
                precision = precision_score(val_true, val_preds)
                recall = recall_score(val_true, val_preds)
                f1 = f1_score(val_true, val_preds)
                
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
                    logging.info(f"Epoch {epoch+1}/{epochs}")
                    logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
                    logging.info(f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        metrics = self._calculate_metrics(X_val, y_val)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(X_val, y_val)
        
        return metrics['loss'], metrics['accuracy'], metrics
    
    def _calculate_metrics(self, X_val, y_val):
        """Calculate classification metrics."""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_loss = nn.BCELoss()(val_outputs, y_val.reshape(-1, 1))
            
            # Get predictions
            val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int)
            val_true = y_val.cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(val_true, val_preds)
            precision = precision_score(val_true, val_preds)
            recall = recall_score(val_true, val_preds)
            f1 = f1_score(val_true, val_preds)
            
            metrics = {
                'loss': val_loss.item(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            return metrics
    
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
    
    def _plot_confusion_matrix(self, X_val, y_val):
        """Plot confusion matrix of predictions."""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val)
            val_preds = (val_outputs.cpu().numpy() > 0.5).astype(int)
            val_true = y_val.cpu().numpy()
            
            cm = confusion_matrix(val_true, val_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(self.plots_dir / 'confusion_matrix.png')
            plt.close()
    
    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
            # Convert probabilities to binary predictions
            binary_predictions = (predictions.cpu().numpy() > 0.5).astype(int)
        return binary_predictions 