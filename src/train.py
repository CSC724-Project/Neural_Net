import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List
import os
import logging
from datetime import datetime

from data.collector import MetricCollector
from data.processor import FeatureProcessor
from models.chunk_predictor import ChunkSizePredictor, ChunkSizeTrainer
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_training_data(collector: MetricCollector,
                         processor: FeatureProcessor,
                         file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Collect and process training data"""
    features_list = []
    targets_list = []
    
    for file_path in file_paths:
        try:
            # Collect metrics
            system_metrics = collector.collect_system_metrics()
            file_metrics = collector.collect_file_metrics(file_path)
            
            if not file_metrics:  # Skip if file metrics collection failed
                continue
                
            # Process features
            features = processor.engineer_features(system_metrics, file_metrics)
            features_list.append(features)
            
            # Calculate target (optimal chunk size) for training
            target = processor.get_optimal_chunk_size(features)
            targets_list.append([target])
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    return np.array(features_list), np.array(targets_list)

def prepare_data_loaders(features: np.ndarray,
                        targets: np.ndarray,
                        processor: FeatureProcessor) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders"""
    # Process features
    features_tensor = processor.process_features(features, training=True)
    targets_tensor = torch.FloatTensor(targets)
    
    # Split into train and validation
    dataset_size = len(features_tensor)
    indices = list(range(dataset_size))
    split = int(np.floor(config['model'].VALIDATION_SPLIT * dataset_size))
    
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    # Create data loaders
    train_dataset = TensorDataset(
        features_tensor[train_indices],
        targets_tensor[train_indices]
    )
    val_dataset = TensorDataset(
        features_tensor[val_indices],
        targets_tensor[val_indices]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model'].BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model'].BATCH_SIZE
    )
    
    return train_loader, val_loader

def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                model_dir: str = 'models') -> ChunkSizePredictor:
    """Train the model"""
    model = ChunkSizePredictor()
    trainer = ChunkSizeTrainer(model)
    
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    for epoch in range(config['model'].EPOCHS):
        # Training
        train_losses = []
        for features, targets in train_loader:
            loss = trainer.train_step(features, targets)
            train_losses.append(loss)
        
        # Validation
        val_losses = []
        for features, targets in val_loader:
            loss, should_stop = trainer.validate(features, targets)
            val_losses.append(loss)
        
        # Logging
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        logger.info(
            f"Epoch {epoch+1}/{config['model'].EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        # Save best model
        if avg_val_loss < trainer.best_loss:
            trainer.save_model(best_model_path)
            logger.info(f"Saved new best model with validation loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if should_stop:
            logger.info("Early stopping triggered")
            break
    
    # Load best model
    trainer.load_model(best_model_path)
    return model

def main():
    """Main training function"""
    # Initialize components
    collector = MetricCollector()
    processor = FeatureProcessor()
    
    # Get list of files to process (you'll need to implement this)
    file_paths = []  # TODO: Implement file collection logic
    
    logger.info("Collecting training data...")
    features, targets = collect_training_data(collector, processor, file_paths)
    
    if len(features) == 0:
        logger.error("No training data collected!")
        return
    
    logger.info(f"Collected {len(features)} training samples")
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(features, targets, processor)
    
    # Train model
    logger.info("Starting model training...")
    model = train_model(train_loader, val_loader)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 