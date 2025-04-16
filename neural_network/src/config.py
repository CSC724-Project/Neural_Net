from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BeeGFSConfig:
    """BeeGFS-specific configuration"""
    INFLUXDB_URL: str = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    INFLUXDB_TOKEN: str = os.getenv('INFLUXDB_TOKEN', '')
    INFLUXDB_ORG: str = os.getenv('INFLUXDB_ORG', 'beegfs')
    INFLUXDB_BUCKET: str = os.getenv('INFLUXDB_BUCKET', 'beegfs_metrics')
    
    # BeeGFS specific settings
    STORAGE_TARGETS: List[str] = field(default_factory=list)
    METADATA_SERVERS: List[str] = field(default_factory=list)
    CLIENT_NODES: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Neural network model configuration"""
    INPUT_FEATURES: int = 16  # Updated for new feature set
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [64, 32, 16])
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32  # Reduced batch size for better generalization
    EPOCHS: int = 150  # Increased epochs for binary classification
    
    # Chunk size constraints (in KB)
    MIN_CHUNK_SIZE: int = 64    # 64KB
    MAX_CHUNK_SIZE: int = 8192  # 8MB
    
    # Training settings
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 15  # Increased patience for binary classification
    
@dataclass
class MetricsConfig:
    """Configuration for metric collection"""
    COLLECTION_INTERVAL: int = 60  # seconds
    HISTORY_WINDOW: str = '1h'     # InfluxDB time format
    METRICS_TO_COLLECT: List[str] = field(default_factory=lambda: [
        'io_throughput',
        'network_latency',
        'metadata_ops_rate',
        'storage_usage',
        'cpu_usage',
        'memory_usage',
        'network_bandwidth',
        'concurrent_ops'
    ])

config = {
    'beegfs': BeeGFSConfig(),
    'model': ModelConfig(),
    'metrics': MetricsConfig()
} 