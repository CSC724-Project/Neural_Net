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
        self.target_scaler = StandardScaler()
        self.feature_names = None
        self.feature_stats = {}
        self.target_stats = {}
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Store target transformation parameters
        self.target_min = None
        self.target_max = None
        self.target_mean = None
        self.target_std = None
        
    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        df = pd.read_csv(csv_path)
        self._analyze_data(df)  # Analyze data distribution
        return self._extract_features(df)
    
    def _analyze_data(self, df: pd.DataFrame) -> None:
        """Analyze data distribution and log statistics."""
        logging.info("\nData Analysis:")
        
        # Analyze chunk size distribution
        chunk_stats = df['chunk_size'].describe()
        self.target_stats = {
            'mean': chunk_stats['mean'],
            'std': chunk_stats['std'],
            'min': chunk_stats['min'],
            'max': chunk_stats['max'],
            'median': chunk_stats['50%'],
            'q1': chunk_stats['25%'],
            'q3': chunk_stats['75%']
        }
        
        # Log basic statistics
        logging.info("\nChunk Size Statistics:")
        for key, value in self.target_stats.items():
            logging.info(f"{key}: {value:.2f}")
        
        # Plot original distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='chunk_size', bins=50)
        plt.title('Original Chunk Size Distribution')
        plt.savefig(self.plots_dir / 'chunk_size_original_dist.png')
        plt.close()
        
        # Plot log distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=np.log1p(df['chunk_size']), bins=50)
        plt.title('Log-transformed Chunk Size Distribution')
        plt.savefig(self.plots_dir / 'chunk_size_log_dist.png')
        plt.close()
        
        # Analyze outliers
        Q1 = chunk_stats['25%']
        Q3 = chunk_stats['75%']
        IQR = Q3 - Q1
        outlier_low = Q1 - 1.5 * IQR
        outlier_high = Q3 + 1.5 * IQR
        outliers = df[(df['chunk_size'] < outlier_low) | (df['chunk_size'] > outlier_high)]
        
        logging.info(f"\nOutlier Analysis:")
        logging.info(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        logging.info(f"Outlier threshold (low): {outlier_low:.2f}")
        logging.info(f"Outlier threshold (high): {outlier_high:.2f}")
        
        if len(outliers) > 0:
            logging.info("\nOutlier Examples:")
            logging.info(outliers[['chunk_size', 'file_size', 'access_count', 'throughput_mbps']].head())
        
        # Determine best scaling approach based on data distribution
        skewness = df['chunk_size'].skew()
        logging.info(f"\nTarget Skewness: {skewness:.2f}")
        
        # Store scaling approach decision
        self.target_stats['skewness'] = skewness
        self.target_stats['scaling_method'] = 'log_minmax' if abs(skewness) > 1 else 'robust'
        
        logging.info(f"Selected scaling method: {self.target_stats['scaling_method']}")
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and engineer features from raw data."""
        # Handle outliers in chunk_size using domain-specific approach
        df = self._handle_outliers(df)
        
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
            'log_throughput'
        ]
        
        self.feature_names = features
        
        # Analyze feature correlations
        correlation_matrix = df[features + ['log_chunk_size']].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_correlations.png')
        plt.close()
        
        # Log correlations with target
        correlations = correlation_matrix['log_chunk_size'].sort_values(ascending=False)
        logging.info("\nFeature correlations with log_chunk_size:")
        for feat, corr in correlations.items():
            if feat != 'log_chunk_size':
                logging.info(f"{feat}: {corr:.3f}")
        
        X = df[features].values
        y = df['log_chunk_size'].values if self.target_stats['scaling_method'] == 'log_minmax' else df['chunk_size'].values
        
        # Clean data
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features and target
        X_scaled = self.feature_scaler.fit_transform(X)
        
        if self.target_stats['scaling_method'] == 'log_minmax':
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.target_scaler = RobustScaler()
        
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        return X_scaled, y_scaled
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using domain-specific rules."""
        # Calculate basic statistics
        Q1 = df['chunk_size'].quantile(0.25)
        Q3 = df['chunk_size'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define more conservative bounds for chunk sizes
        lower_bound = max(Q1 - 2 * IQR, 4096)  # Minimum 4KB
        upper_bound = min(Q3 + 2 * IQR, 8 * 1024 * 1024)  # Maximum 8MB
        
        # Filter outliers
        mask = (df['chunk_size'] >= lower_bound) & (df['chunk_size'] <= upper_bound)
        filtered_df = df[mask].copy()
        
        removed = len(df) - len(filtered_df)
        if removed > 0:
            logging.info(f"\nRemoved {removed} samples ({removed/len(df)*100:.2f}%) as outliers")
            logging.info(f"Chunk size bounds: [{lower_bound/1024:.2f}KB, {upper_bound/1024/1024:.2f}MB]")
        
        return filtered_df
    
    def _transform_target(self, y):
        """Transform target values using log transformation and standardization."""
        # Store original range for inverse transformation
        self.target_min = float(np.min(y))
        self.target_max = float(np.max(y))
        
        # Log transform (use natural log as it's more interpretable)
        y_log = np.log(y)
        
        # Standardize log-transformed values
        y_scaled = self.target_scaler.fit_transform(y_log)
        
        # Store transformation parameters
        self.target_mean = float(self.target_scaler.mean_[0])
        self.target_std = float(self.target_scaler.scale_[0])
        
        # Analyze transformed distribution
        self._analyze_target_distribution(y_scaled.ravel(), "transformed")
        
        return y_scaled
    
    def _analyze_target_distribution(self, values, label):
        """Analyze and plot the distribution of values."""
        # Calculate statistics
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'skewness': float(pd.Series(values).skew())
        }
        
        # Log statistics
        logging.info(f"\n{label.title()} Distribution Statistics:")
        for metric, value in stats.items():
            logging.info(f"{metric}: {value:.2f}")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        plt.title(f'{label.title()} Chunk Size Distribution')
        plt.xlabel('Chunk Size')
        plt.ylabel('Count')
        plt.savefig(self.plots_dir / f'chunk_size_dist_{label}.png')
        plt.close()
        
        # Plot Q-Q plot to check normality
        plt.figure(figsize=(10, 6))
        from scipy import stats
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f'{label.title()} Q-Q Plot')
        plt.savefig(self.plots_dir / f'chunk_size_qq_{label}.png')
        plt.close()
    
    def inverse_transform_target(self, y_scaled):
        """Inverse transform the target values back to original scale."""
        # Inverse standardization
        y_log = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        
        # Inverse log transformation
        y_orig = np.exp(y_log)
        
        return y_orig.ravel()
    
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