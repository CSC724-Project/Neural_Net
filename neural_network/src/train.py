import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List
import os
import logging
from datetime import datetime
from sklearn.model_selection import KFold
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.data.collector import MetricCollector
from src.data.processor import FeatureProcessor
from src.models.chunk_predictor import ChunkSizePredictor, ChunkSizeTrainer
from config import config
from src.data.preprocessor import DataPreprocessor
from src.models.chunk_size_predictor import ModelTrainer

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

def train_model(data_path, n_splits=5, epochs=200, batch_size=64, learning_rate=0.001):
    """
    Train the chunk size predictor using k-fold cross validation.
    
    Args:
        data_path (str): Path to the CSV data file
        n_splits (int): Number of folds for cross-validation
        epochs (int): Number of training epochs per fold
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimization
    """
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_data(data_path)
    
    # Initialize K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics
    loss_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Initialize trainer with preprocessor
    trainer = ModelTrainer(input_size=X.shape[1], preprocessor=preprocessor)
    
    # Perform k-fold training
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logging.info(f"\nTraining Fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        loss, accuracy, metrics = trainer.train_fold(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Store metrics
        loss_scores.append(loss)
        accuracy_scores.append(accuracy)
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        
        logging.info(f"Fold {fold} Results:")
        logging.info(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Calculate and log average metrics
    avg_loss = np.mean(loss_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    
    std_loss = np.std(loss_scores)
    std_accuracy = np.std(accuracy_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_f1 = np.std(f1_scores)
    
    logging.info("\nOverall Results:")
    logging.info(f"Average Loss: {avg_loss:.4f} (±{std_loss:.4f})")
    logging.info(f"Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logging.info(f"Average Precision: {avg_precision:.4f} (±{std_precision:.4f})")
    logging.info(f"Average Recall: {avg_recall:.4f} (±{std_recall:.4f})")
    logging.info(f"Average F1 Score: {avg_f1:.4f} (±{std_f1:.4f})")
    
    # Save results
    results = {
        "metrics": {
            "loss": {
                "mean": float(avg_loss),
                "std": float(std_loss),
                "folds": [float(score) for score in loss_scores]
            },
            "accuracy": {
                "mean": float(avg_accuracy),
                "std": float(std_accuracy),
                "folds": [float(score) for score in accuracy_scores]
            },
            "precision": {
                "mean": float(avg_precision),
                "std": float(std_precision),
                "folds": [float(score) for score in precision_scores]
            },
            "recall": {
                "mean": float(avg_recall),
                "std": float(std_recall),
                "folds": [float(score) for score in recall_scores]
            },
            "f1": {
                "mean": float(avg_f1),
                "std": float(std_f1),
                "folds": [float(score) for score in f1_scores]
            }
        },
        "parameters": {
            "n_splits": n_splits,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "features": preprocessor.get_feature_names()
        }
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results to JSON
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return trainer

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
    model = train_model(features, targets)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    data_path = "src/data/combined_logs.csv"
    model = train_model(data_path) 