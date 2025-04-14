import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import KFold
from pathlib import Path
import json
import os

from src.data.preprocessor import DataPreprocessor
from src.models.xgb_predictor import XGBChunkPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgb_model(data_path, n_splits=5, params=None):
    """
    Train the XGBoost chunk size predictor using k-fold cross validation.
    
    Args:
        data_path (str): Path to the CSV data file
        n_splits (int): Number of folds for cross-validation
        params (dict): XGBoost parameters
    """
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_data(data_path)
    
    # Initialize K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics
    mse_scores = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    # Default XGBoost parameters
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'eta': 0.01,  # Changed from learning_rate to eta
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': 42
        }
    
    # Initialize trainer with preprocessor
    trainer = XGBChunkPredictor(preprocessor=preprocessor)
    
    # Perform k-fold training
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logging.info(f"\nTraining Fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        mse, r2, metrics = trainer.train_fold(
            X_train, y_train,
            X_val, y_val,
            params=params
        )
        
        # Store metrics
        mse_scores.append(mse)
        r2_scores.append(r2)
        if 'rmse_orig' in metrics:
            rmse_scores.append(metrics['rmse_orig'])
            mae_scores.append(metrics['mae_orig'])
            mape_scores.append(metrics['mape'])
        else:
            rmse_scores.append(metrics['rmse'])
            mae_scores.append(metrics['mae'])
        
        logging.info(f"Fold {fold} - MSE: {mse:.4f}, RMSE: {metrics.get('rmse_orig', metrics['rmse']):.4f}")
        logging.info(f"         MAE: {metrics.get('mae_orig', metrics['mae']):.4f}, R²: {r2:.4f}")
        if 'mape' in metrics:
            logging.info(f"         MAPE: {metrics['mape']:.4%}")
    
    # Calculate and log average metrics
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    std_mse = np.std(mse_scores)
    std_r2 = np.std(r2_scores)
    std_rmse = np.std(rmse_scores)
    std_mae = np.std(mae_scores)
    
    logging.info("\nOverall Results:")
    logging.info(f"Average MSE: {avg_mse:.4f} (±{std_mse:.4f})")
    logging.info(f"Average RMSE: {avg_rmse:.4f} (±{std_rmse:.4f})")
    logging.info(f"Average MAE: {avg_mae:.4f} (±{std_mae:.4f})")
    logging.info(f"Average R2 Score: {avg_r2:.4f} (±{std_r2:.4f})")
    if mape_scores:
        avg_mape = np.mean(mape_scores)
        std_mape = np.std(mape_scores)
        logging.info(f"Average MAPE: {avg_mape:.4%} (±{std_mape:.4%})")
    
    # Save results
    results = {
        "metrics": {
            "mse": {
                "mean": float(avg_mse),
                "std": float(std_mse),
                "folds": [float(score) for score in mse_scores]
            },
            "rmse": {
                "mean": float(avg_rmse),
                "std": float(std_rmse),
                "folds": [float(score) for score in rmse_scores]
            },
            "mae": {
                "mean": float(avg_mae),
                "std": float(std_mae),
                "folds": [float(score) for score in mae_scores]
            },
            "r2": {
                "mean": float(avg_r2),
                "std": float(std_r2),
                "folds": [float(score) for score in r2_scores]
            }
        },
        "parameters": {
            "n_splits": n_splits,
            "model_params": params,
            "features": preprocessor.get_feature_names()
        }
    }
    
    if mape_scores:
        results["metrics"]["mape"] = {
            "mean": float(avg_mape),
            "std": float(std_mape),
            "folds": [float(score) for score in mape_scores]
        }
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save results to JSON
    with open(results_dir / "xgboost_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return trainer

if __name__ == "__main__":
    # Get the script's directory
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "src" / "data" / "beegfs_test_results4.csv"
    train_xgb_model(data_path) 