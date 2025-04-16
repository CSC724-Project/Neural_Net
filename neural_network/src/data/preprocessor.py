import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_scaler = StandardScaler()
        self.feature_names = None
        self.feature_stats = {}
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        df = pd.read_csv(csv_path)
        self._analyze_data(df)  # Analyze data distribution
        return self._extract_features(df)
    
    def _analyze_data(self, df: pd.DataFrame) -> None:
        """Analyze data distribution and log statistics."""
        logging.info("\nData Analysis:")
        
        # Analyze OT distribution
        ot_stats = df['OT'].value_counts()
        self.target_stats = {
            'positive_samples': int(ot_stats.get(1, 0)),
            'negative_samples': int(ot_stats.get(0, 0))
        }
        
        # Log basic statistics
        logging.info("\nOptimal Throughput Statistics:")
        total_samples = sum(self.target_stats.values())
        logging.info(f"Total samples: {total_samples}")
        logging.info(f"Optimal samples: {self.target_stats['positive_samples']} ({self.target_stats['positive_samples']/total_samples*100:.2f}%)")
        logging.info(f"Non-optimal samples: {self.target_stats['negative_samples']} ({self.target_stats['negative_samples']/total_samples*100:.2f}%)")
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='OT')
        plt.title('Optimal Throughput Distribution')
        plt.xlabel('Is Optimal')
        plt.ylabel('Count')
        plt.savefig(self.plots_dir / 'ot_distribution.png')
        plt.close()
        
        # Analyze feature correlations with OT
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['OT'].sort_values(ascending=False)
        
        logging.info("\nFeature correlations with Optimal Throughput:")
        for feat, corr in correlations.items():
            if feat != 'OT':
                logging.info(f"{feat}: {corr:.3f}")
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and engineer features from raw data."""
        # Log transform size-related features
        df['log_file_size'] = np.log1p(df['file_size'])
        df['log_chunk_size'] = np.log1p(df['chunk_size'])
        
        # Domain-specific feature engineering
        df['size_ratio'] = df['chunk_size'] / df['file_size']
        df['ops_per_chunk'] = df['access_count'] / (df['file_size'] / df['chunk_size'])
        
        # Normalize features by file size
        for col in ['avg_read_size', 'avg_write_size', 'max_read_size', 'max_write_size']:
            df[f'norm_{col}'] = df[col] / df['file_size']
        
        # Operation patterns
        total_ops = df['read_count'] + df['write_count']
        df['read_ratio'] = df['read_count'] / total_ops
        df['write_ratio'] = df['write_count'] / total_ops
        df['ops_density'] = total_ops / (df['file_size'] / (1024 * 1024))  # ops per MB
        
        # Performance metrics
        df['throughput_per_op'] = df['throughput_mbps'] / total_ops
        df['throughput_density'] = df['throughput_mbps'] / (df['file_size'] / (1024 * 1024))
        df['log_throughput'] = np.log1p(df['throughput_mbps'])
        
        # Access count label features
        df['access_density'] = df['access_count_label'] / (df['file_size'] / (1024 * 1024))
        df['log_access_count'] = np.log1p(df['access_count_label'])
        
        # Selected features
        features = [
            'log_file_size',
            'size_ratio',
            'ops_per_chunk',
            'ops_density',
            'read_ratio',
            'write_ratio',
            'norm_avg_read_size',
            'norm_avg_write_size',
            'norm_max_read_size',
            'norm_max_write_size',
            'throughput_density',
            'throughput_per_op',
            'log_throughput',
            'access_density',
            'log_access_count',
            'combination'  # New feature from dataset
        ]
        
        self.feature_names = features
        
        # Analyze feature correlations
        correlation_matrix = df[features + ['OT']].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_correlations.png')
        plt.close()
        
        X = df[features].values
        y = df['OT'].values  # Binary classification target
        
        # Clean data
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        return X_scaled, y
    
    def get_feature_names(self):
        """Return the list of feature names."""
        return self.feature_names
    
    def get_target_stats(self):
        """Return target variable statistics."""
        return self.target_stats
    
    def prepare_k_fold(self, X, y, n_splits=5):
        """Prepare K-fold cross validation splits."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return kf.split(X)
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler."""
        X_train_scaled = self.feature_scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.feature_scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
            
        return X_train_scaled 