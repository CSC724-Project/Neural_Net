import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class XGBChunkPredictor:
    def __init__(self, preprocessor=None):
        self.model = None
        self.preprocessor = preprocessor
        self.best_model = None
        self.feature_importance = None
        
        # Create plots directory
        self.plots_dir = Path("results/plots/xgboost")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def train_fold(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model on a single fold."""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae'],
                'eta': 0.01,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'seed': 42
            }
        elif 'learning_rate' in params:
            # Convert learning_rate to eta if present
            params['eta'] = params.pop('learning_rate')
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with early stopping
        num_boost_round = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=100  # Print evaluation every 100 rounds
        )
        
        # Store feature importance
        self.feature_importance = model.get_score(importance_type='gain')
        
        # Plot feature importance
        self._plot_feature_importance()
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_val, y_val)
        
        # Plot predictions vs actual
        self._plot_predictions(model, X_val, y_val)
        
        # Store best model
        self.best_model = model
        
        # Return metrics in the same format as the neural network for consistency
        if self.preprocessor is not None:
            return metrics['mse_orig'], metrics['r2_orig'], metrics
        return metrics['mse'], metrics['r2'], metrics
    
    def _calculate_metrics(self, model, X_val, y_val):
        """Calculate regression metrics."""
        # Get predictions
        dval = xgb.DMatrix(X_val)
        val_pred = model.predict(dval)
        
        # Calculate metrics on scaled data
        mse = mean_squared_error(y_val, val_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # If preprocessor is available, calculate metrics on original scale
        if self.preprocessor is not None:
            y_true_orig = self.preprocessor.inverse_transform_target(y_val)
            y_pred_orig = self.preprocessor.inverse_transform_target(val_pred)
            
            mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
            rmse_orig = np.sqrt(mse_orig)
            mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
            mape = mean_absolute_percentage_error(y_true_orig, y_pred_orig)
            r2_orig = r2_score(y_true_orig, y_pred_orig)
            
            # Log metrics
            logging.info("\nValidation Metrics (Original Scale):")
            logging.info(f"MSE: {mse_orig:.2f}")
            logging.info(f"RMSE: {rmse_orig:.2f}")
            logging.info(f"MAE: {mae_orig:.2f}")
            logging.info(f"MAPE: {mape:.4f}")
            logging.info(f"R²: {r2_orig:.4f}")
            
            metrics.update({
                'mse_orig': mse_orig,
                'rmse_orig': rmse_orig,
                'mae_orig': mae_orig,
                'mape': mape,
                'r2_orig': r2_orig
            })
        
        return metrics
    
    def _plot_predictions(self, model, X_val, y_val):
        """Plot predicted vs actual values."""
        dval = xgb.DMatrix(X_val)
        predictions = model.predict(dval)
        
        if self.preprocessor is not None:
            y_true = self.preprocessor.inverse_transform_target(y_val)
            y_pred = self.preprocessor.inverse_transform_target(predictions)
        else:
            y_true = y_val
            y_pred = predictions
        
        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Chunk Size')
        plt.ylabel('Predicted Chunk Size')
        plt.title('XGBoost: Predicted vs Actual Chunk Sizes')
        plt.savefig(self.plots_dir / 'predictions_scatter.png')
        plt.close()
        
        # Error distribution
        errors = y_pred - y_true
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50)
        plt.title('XGBoost: Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.savefig(self.plots_dir / 'error_distribution.png')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance scores."""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance (gain)')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png')
        plt.close()
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet")
        
        dtest = xgb.DMatrix(X)
        return self.best_model.predict(dtest) 