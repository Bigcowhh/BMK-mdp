"""
Configuration file for the MaiMai difficulty prediction project.
"""
import os
import torch
from dataclasses import dataclass
from typing import Optional


# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SERIALIZED_DIR = os.path.join(DATA_DIR, 'serialized')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data files
SONG_INFO_PATH = os.path.join(DATA_DIR, 'song_info.csv')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
EXCLUDED_SONGS_PATH = os.path.join(DATA_DIR, 'excluded_songs.csv')
SONGS_JSON_PATH = os.path.join(DATA_DIR, 'maimai-songs', 'songs.json')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingConfig:
    """训练配置类，管理所有训练相关的超参数"""
    
    # Model configuration
    input_size: int = 21
    hidden_size: int = 64
    output_size: int = 1
    num_layers: int = 2
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 50
    weight_decay: float = 1e-5
    
    # Optimizer configuration
    optimizer_type: str = 'Adam'
    
    # Scheduler configuration
    scheduler_type: str = 'ReduceLROnPlateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_mode: str = 'min'
    
    # Early stopping configuration
    early_stop_patience: int = 10
    
    # Gradient clipping
    grad_clip_max_norm: float = 1.0
    
    # Progress logging
    log_interval: int = 10  # 每多少个batch输出一次进度
    
    # TensorBoard configuration
    log_parameters: bool = True
    log_gradients: bool = True
    log_histograms: bool = True
    
    # Model checkpointing
    save_best_model: bool = True
    model_save_name: str = "best_model.pth"
    
    # Evaluation thresholds
    accuracy_thresholds: tuple = (0.1, 0.2, 0.5)
    
    # Visualization
    max_scatter_points: int = 1000  # 散点图最大点数限制
    
    def to_dict(self) -> dict:
        """将配置转换为字典格式"""
        return {
            'model_type': 'SimpleLSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'early_stop_patience': self.early_stop_patience,
            'grad_clip_max_norm': self.grad_clip_max_norm,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_factor': self.scheduler_factor,
        }


# 默认训练配置实例
DEFAULT_TRAINING_CONFIG = TrainingConfig()

# 向后兼容的配置变量
BATCH_SIZE = DEFAULT_TRAINING_CONFIG.batch_size
LEARNING_RATE = DEFAULT_TRAINING_CONFIG.learning_rate
NUM_EPOCHS = DEFAULT_TRAINING_CONFIG.num_epochs

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)