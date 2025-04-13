import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np
from ..config import config

class ChunkSizePredictor(nn.Module):
    def __init__(self, input_size: int = config['model'].INPUT_FEATURES):
        super(ChunkSizePredictor, self).__init__()
        
        # Define network architecture
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in config['model'].HIDDEN_LAYERS:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Final prediction layer
        self.prediction_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.ReLU()  # Ensure positive chunk sizes
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        chunk_size = self.prediction_layer(features)
        
        # Scale output to be within valid chunk size range
        chunk_size = torch.clamp(
            chunk_size,
            min=config['model'].MIN_CHUNK_SIZE,
            max=config['model'].MAX_CHUNK_SIZE
        )
        
        return chunk_size

class ChunkSizeTrainer:
    def __init__(self, model: ChunkSizePredictor):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['model'].LEARNING_RATE
        )
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_step(self, 
                   features: torch.Tensor, 
                   targets: torch.Tensor) -> float:
        """Perform one training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(features)
        loss = self.criterion(predictions, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, 
                features: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[float, bool]:
        """Validate model and check for early stopping"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
        # Early stopping check
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
            should_stop = False
        else:
            self.patience_counter += 1
            should_stop = self.patience_counter >= config['model'].EARLY_STOPPING_PATIENCE
            
        return loss.item(), should_stop
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        """Make predictions for new data"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)
        return predictions.numpy()

    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss'] 