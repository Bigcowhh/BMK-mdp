"""
Configuration file for the MaiMai difficulty prediction project.
"""
import os
import torch

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

# Model parameters


# Training parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)